import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.classifier import Prediction
from app.config import Settings
from app.main import create_app

CLASSES = ["Forest", "SeaLake"]


class FakeClassifier:
    version = "fake-1"
    review_threshold = 0.6
    input_size = (64, 64)
    classes = CLASSES

    def predict(self, image: Image.Image) -> Prediction:
        green = image.convert("RGB").getpixel((0, 0))[1] / 255
        probs = {"Forest": green, "SeaLake": 1 - green}
        label = max(probs, key=probs.get)
        return Prediction(label, probs[label], abs(green - (1 - green)), probs)


def png(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    settings = Settings(model_dir=tmp_path, data_dir=tmp_path / "data", audit_rate=0.0)
    with TestClient(create_app(settings, FakeClassifier())) as c:
        yield c


def upload(client, data: bytes, name: str = "tile.png", **form):
    return client.post("/tiles", files={"file": (name, data, "image/png")}, data=form)


def test_classifies_and_stores(client):
    r = upload(client, png((0, 230, 0)), source="pass-42")
    assert r.status_code == 201
    body = r.json()
    assert body["label"] == "Forest"
    assert body["status"] == "accepted"
    assert body["source"] == "pass-42"
    assert body["model_version"] == "fake-1"
    assert client.get(f"/tiles/{body['id']}").json() == body


def test_duplicate_tile_is_idempotent(client):
    first = upload(client, png((0, 230, 0)))
    second = upload(client, png((0, 230, 0)), name="renamed.png")
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert client.get("/tiles").json()["total"] == 1


def test_low_confidence_goes_to_review(client):
    body = upload(client, png((0, 140, 0))).json()
    assert body["status"] == "needs_review"
    assert body["review_reason"] == "low_confidence"


def test_unexpected_size_goes_to_review(client):
    body = upload(client, png((0, 250, 0), size=(256, 256))).json()
    assert body["status"] == "needs_review"
    assert body["review_reason"] == "unexpected_size"


def test_audit_sample_routes_confident_tiles_to_review(tmp_path):
    settings = Settings(model_dir=tmp_path, data_dir=tmp_path / "data", audit_rate=1.0)
    with TestClient(create_app(settings, FakeClassifier())) as c:
        body = upload(c, png((0, 240, 0))).json()
    assert body["status"] == "needs_review"
    assert body["review_reason"] == "audit_sample"


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_rejects_unreadable_upload(client, data):
    assert upload(client, data).status_code == 400


def test_query_filters(client):
    upload(client, png((0, 240, 0)))
    upload(client, png((0, 10, 0)))
    upload(client, png((0, 140, 0)))
    assert client.get("/tiles", params={"label": "SeaLake"}).json()["total"] == 1
    assert client.get("/tiles", params={"status": "needs_review"}).json()["total"] == 1
    assert client.get("/tiles", params={"min_confidence": 0.9}).json()["total"] == 2
    page = client.get("/tiles", params={"limit": 2}).json()
    assert page["total"] == 3
    assert len(page["items"]) == 2


def test_review_overrides_label(client):
    tile = upload(client, png((0, 140, 0))).json()
    r = client.post(f"/tiles/{tile['id']}/review", json={"label": "SeaLake"})
    assert r.json()["status"] == "reviewed"
    assert client.get("/tiles", params={"label": "SeaLake"}).json()["total"] == 1
    stats = client.get("/stats").json()
    assert stats["reviewed"] == 1
    assert stats["reviewed_agreement"] == 0
    assert client.post(f"/tiles/{tile['id']}/review", json={"label": "Moon"}).status_code == 422
    assert client.post("/tiles/999/review", json={"label": "Forest"}).status_code == 404
