import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY,
    tile_sha256     TEXT    NOT NULL,
    filename        TEXT,
    source          TEXT,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    stored_path     TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,
    label           TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    margin          REAL    NOT NULL,
    probabilities   TEXT    NOT NULL,
    status          TEXT    NOT NULL CHECK (status IN ('accepted', 'needs_review', 'reviewed')),
    review_reason   TEXT,
    reviewed_label  TEXT,
    latency_ms      REAL    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    reviewed_at     TEXT,
    UNIQUE (tile_sha256, model_version)
);
CREATE INDEX IF NOT EXISTS ix_predictions_label ON predictions (label);
CREATE INDEX IF NOT EXISTS ix_predictions_status ON predictions (status);
CREATE INDEX IF NOT EXISTS ix_predictions_created_at ON predictions (created_at);
"""

FILTERS = {
    "label": "COALESCE(reviewed_label, label) = ?",
    "status": "status = ?",
    "model_version": "model_version = ?",
    "source": "source = ?",
    "min_confidence": "confidence >= ?",
    "max_confidence": "confidence < ?",
    "since": "created_at >= ?",
    "until": "created_at < ?",
}


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def find(self, sha256: str, model_version: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE tile_sha256 = ? AND model_version = ?",
                (sha256, model_version),
            ).fetchone()
        return _row(row)

    def insert(self, record: dict[str, Any]) -> dict[str, Any]:
        record = {**record, "probabilities": json.dumps(record["probabilities"])}
        columns = ", ".join(record)
        params = ", ".join(f":{k}" for k in record)
        with self._connect() as conn:
            cur = conn.execute(f"INSERT INTO predictions ({columns}) VALUES ({params})", record)
        return self.get(cur.lastrowid)

    def get(self, prediction_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
        return _row(row)

    def query(self, limit: int, offset: int, **filters: Any) -> tuple[int, list[dict[str, Any]]]:
        active = {k: v for k, v in filters.items() if v is not None}
        where = " AND ".join(FILTERS[k] for k in active) or "1=1"
        args = list(active.values())
        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM predictions WHERE {where}", args).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM predictions WHERE {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        return total, [_row(r) for r in rows]

    def review(self, prediction_id: int, label: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE predictions SET reviewed_label = ?, status = 'reviewed', "
                "reviewed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (label, prediction_id),
            )
        return self.get(prediction_id)

    def stats(self, model_version: str | None = None) -> dict[str, Any]:
        where, args = ("WHERE model_version = ?", [model_version]) if model_version else ("", [])
        with self._connect() as conn:
            by_label = conn.execute(
                f"SELECT COALESCE(reviewed_label, label) AS label, COUNT(*) AS n, AVG(confidence) AS mean_confidence "
                f"FROM predictions {where} GROUP BY 1 ORDER BY n DESC",
                args,
            ).fetchall()
            by_status = conn.execute(
                f"SELECT status, COUNT(*) AS n FROM predictions {where} GROUP BY status", args
            ).fetchall()
            reviewed = conn.execute(
                f"SELECT COUNT(*) AS n, SUM(reviewed_label = label) AS agreed FROM predictions "
                f"{where} {'AND' if where else 'WHERE'} reviewed_label IS NOT NULL",
                args,
            ).fetchone()
        return {
            "total": sum(r["n"] for r in by_status),
            "by_status": {r["status"]: r["n"] for r in by_status},
            "by_label": [dict(r) for r in by_label],
            "reviewed": reviewed["n"],
            "reviewed_agreement": (reviewed["agreed"] / reviewed["n"]) if reviewed["n"] else None,
        }


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    data["probabilities"] = json.loads(data["probabilities"])
    return data
