import csv
from pathlib import Path

import numpy as np
from PIL import Image

from app.classifier import embed


def labelled_folders(root: Path) -> list[tuple[Path, str]]:
    return [(p, d.name) for d in sorted(root.iterdir()) if d.is_dir() for p in sorted(d.glob("*.png"))]


def labelled_csv(image_dir: Path, labels_csv: Path) -> list[tuple[Path, str]]:
    with labels_csv.open(newline="") as f:
        return [(image_dir / row["filename"], row["true_label"]) for row in csv.DictReader(f)]


def load_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
    return images


def embed_paths(backbone, paths: list[Path], batch_size: int = 64) -> np.ndarray:
    chunks = [embed(backbone, load_images(paths[i:i + batch_size])) for i in range(0, len(paths), batch_size)]
    return np.concatenate(chunks)
