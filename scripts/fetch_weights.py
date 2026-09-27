import argparse
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

from app.classifier import BACKBONE_FILE
from app.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time online step: save ImageNet ResNet-18 weights locally.")
    parser.add_argument("--out", type=Path, default=Settings().model_dir)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    target = args.out / BACKBONE_FILE
    torch.save(net.state_dict(), target)
    print(f"saved {target}")


if __name__ == "__main__":
    main()
