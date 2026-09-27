import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

BACKBONE_FILE = "resnet18_imagenet.pt"
HEAD_FILE = "head.joblib"
META_FILE = "model.json"

_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_backbone(model_dir: Path) -> torch.nn.Module:
    net = models.resnet18(weights=None)
    state = torch.load(model_dir / BACKBONE_FILE, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.fc = torch.nn.Identity()
    return net.eval()


@torch.inference_mode()
def embed(backbone: torch.nn.Module, images: list[Image.Image]) -> np.ndarray:
    batch = torch.stack([_preprocess(im.convert("RGB")) for im in images])
    return backbone(batch).numpy()


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    margin: float
    probabilities: dict[str, float]


class Classifier:
    def __init__(self, model_dir: Path):
        self.backbone = load_backbone(model_dir)
        self.head = joblib.load(model_dir / HEAD_FILE)
        meta = json.loads((model_dir / META_FILE).read_text())
        self.version: str = meta["version"]
        self.review_threshold: float = meta["review_threshold"]
        self.input_size: tuple[int, int] = tuple(meta["input_size"])
        self.classes: list[str] = [str(c) for c in self.head.classes_]

    def predict_many(self, images: list[Image.Image]) -> list[Prediction]:
        probs = self.head.predict_proba(embed(self.backbone, images))
        return [self._to_prediction(p) for p in probs]

    def predict(self, image: Image.Image) -> Prediction:
        return self.predict_many([image])[0]

    def _to_prediction(self, probs: np.ndarray) -> Prediction:
        top, second = np.argsort(probs)[::-1][:2]
        return Prediction(
            label=self.classes[top],
            confidence=float(probs[top]),
            margin=float(probs[top] - probs[second]),
            probabilities={c: round(float(p), 6) for c, p in zip(self.classes, probs)},
        )
