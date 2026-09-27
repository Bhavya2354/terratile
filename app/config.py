import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    model_dir: Path = field(default_factory=lambda: _path("TERRATILE_MODEL_DIR", ROOT / "models"))
    data_dir: Path = field(default_factory=lambda: _path("TERRATILE_DATA_DIR", ROOT / "data"))
    max_upload_bytes: int = int(os.getenv("TERRATILE_MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
    audit_rate: float = float(os.getenv("TERRATILE_AUDIT_RATE", 0.02))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "terratile.db"

    @property
    def tile_dir(self) -> Path:
        return self.data_dir / "tiles"
