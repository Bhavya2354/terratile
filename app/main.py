import hashlib
import io
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, Protocol

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel

from app.classifier import Classifier, Prediction
from app.config import Settings
from app.db import Store

Status = Literal["accepted", "needs_review", "reviewed"]


class TileClassifier(Protocol):
    version: str
    review_threshold: float
    input_size: tuple[int, int]
    classes: list[str]

    def predict(self, image: Image.Image) -> Prediction: ...


class PredictionOut(BaseModel):
    id: int
    tile_sha256: str
    filename: str | None
    source: str | None
    width: int
    height: int
    model_version: str
    label: str
    confidence: float
    margin: float
    probabilities: dict[str, float]
    status: Status
    review_reason: str | None
    reviewed_label: str | None
    latency_ms: float
    created_at: str
    reviewed_at: str | None


class PredictionPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PredictionOut]


class ReviewIn(BaseModel):
    label: str


def decode_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except OSError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "not a readable image")
    return image


def review_reason(
    pred: Prediction, size: tuple[int, int], sha256: str, clf: TileClassifier, audit_rate: float
) -> str | None:
    if size != clf.input_size:
        return "unexpected_size"
    if pred.confidence < clf.review_threshold:
        return "low_confidence"
    if int(sha256[:8], 16) / 0x100000000 < audit_rate:
        return "audit_sample"
    return None


def create_app(settings: Settings | None = None, classifier: TileClassifier | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier or Classifier(settings.model_dir)
        app.state.store = Store(settings.db_path)
        settings.tile_dir.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(title="terratile", lifespan=lifespan)

    def get_store(request: Request) -> Store:
        return request.app.state.store

    def get_classifier(request: Request) -> TileClassifier:
        return request.app.state.classifier

    @app.get("/health")
    def health(clf: TileClassifier = Depends(get_classifier)) -> dict[str, Any]:
        return {"status": "ok", "model_version": clf.version, "review_threshold": clf.review_threshold}

    @app.post("/tiles", response_model=PredictionOut, status_code=status.HTTP_201_CREATED)
    async def classify_tile(
        response: Response,
        file: UploadFile = File(...),
        source: str | None = Form(None),
        store: Store = Depends(get_store),
        clf: TileClassifier = Depends(get_classifier),
    ):
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "tile too large")

        sha256 = hashlib.sha256(data).hexdigest()
        existing = store.find(sha256, clf.version)
        if existing:
            response.status_code = status.HTTP_200_OK
            return existing

        image = decode_image(data)
        started = time.perf_counter()
        pred = await run_in_threadpool(clf.predict, image)
        latency_ms = (time.perf_counter() - started) * 1000

        path = settings.tile_dir / sha256[:2] / f"{sha256}.{(image.format or 'bin').lower()}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        reason = review_reason(pred, image.size, sha256, clf, settings.audit_rate)
        record = {
            "tile_sha256": sha256,
            "filename": file.filename,
            "source": source,
            "width": image.width,
            "height": image.height,
            "stored_path": str(path.relative_to(settings.data_dir)),
            "model_version": clf.version,
            "label": pred.label,
            "confidence": pred.confidence,
            "margin": pred.margin,
            "probabilities": pred.probabilities,
            "status": "needs_review" if reason else "accepted",
            "review_reason": reason,
            "latency_ms": round(latency_ms, 2),
        }
        try:
            return store.insert(record)
        except sqlite3.IntegrityError:
            response.status_code = status.HTTP_200_OK
            return store.find(sha256, clf.version)

    @app.get("/tiles", response_model=PredictionPage)
    def list_tiles(
        label: str | None = None,
        status_: Status | None = Query(None, alias="status"),
        model_version: str | None = None,
        source: str | None = None,
        min_confidence: float | None = Query(None, ge=0, le=1),
        max_confidence: float | None = Query(None, ge=0, le=1),
        since: str | None = None,
        until: str | None = None,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        store: Store = Depends(get_store),
    ):
        total, items = store.query(
            limit, offset,
            label=label, status=status_, model_version=model_version, source=source,
            min_confidence=min_confidence, max_confidence=max_confidence, since=since, until=until,
        )
        return {"total": total, "limit": limit, "offset": offset, "items": items}

    @app.get("/tiles/{prediction_id}", response_model=PredictionOut)
    def get_tile(prediction_id: int, store: Store = Depends(get_store)):
        record = store.get(prediction_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "prediction not found")
        return record

    @app.post("/tiles/{prediction_id}/review", response_model=PredictionOut)
    def review_tile(
        prediction_id: int,
        body: ReviewIn,
        store: Store = Depends(get_store),
        clf: TileClassifier = Depends(get_classifier),
    ):
        if body.label not in clf.classes:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"label must be one of {clf.classes}")
        record = store.review(prediction_id, body.label)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "prediction not found")
        return record

    @app.get("/stats")
    def stats(model_version: str | None = None, store: Store = Depends(get_store)) -> dict[str, Any]:
        return store.stats(model_version)

    return app


app = create_app()
