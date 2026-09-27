import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.classifier import HEAD_FILE, META_FILE, load_backbone
from app.config import Settings
from scripts.dataset import embed_paths, labelled_folders

INPUT_SIZE = (64, 64)


def pick_threshold(confidence: np.ndarray, correct: np.ndarray, target: float) -> float:
    for t in np.round(np.arange(0.0, 1.0, 0.01), 2):
        kept = confidence >= t
        if kept.any() and correct[kept].mean() >= target:
            return float(t)
    return 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a logistic-regression head on frozen ResNet-18 features.")
    parser.add_argument("--dataset", type=Path, required=True, help="path to be-mlsys-assignment-dataset")
    parser.add_argument("--model-dir", type=Path, default=Settings().model_dir)
    parser.add_argument("--target-accuracy", type=float, default=0.95,
                        help="accuracy required on auto-accepted predictions; sets the review threshold")
    args = parser.parse_args()

    samples = labelled_folders(args.dataset / "candidate_tiles")
    paths, labels = [p for p, _ in samples], np.array([y for _, y in samples])
    print(f"embedding {len(paths)} candidate tiles")
    features = embed_paths(load_backbone(args.model_dir), paths)

    head = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = cross_val_predict(head, features, labels, cv=cv, method="predict_proba")
    classes = np.unique(labels)
    confidence = oof.max(axis=1)
    correct = classes[oof.argmax(axis=1)] == labels
    threshold = pick_threshold(confidence, correct, args.target_accuracy)
    kept = confidence >= threshold

    head.fit(features, labels)
    head_path = args.model_dir / HEAD_FILE
    joblib.dump(head, head_path)

    meta = {
        "version": f"r18-logreg-{hashlib.sha256(head_path.read_bytes()).hexdigest()[:8]}",
        "backbone": "resnet18-imagenet1k-v1 (frozen)",
        "head": "standard-scaler + logistic-regression(C=0.1)",
        "classes": classes.tolist(),
        "input_size": list(INPUT_SIZE),
        "review_threshold": threshold,
        "trained_on": len(paths),
        "cv_accuracy": round(float(correct.mean()), 4),
        "cv_accepted_accuracy": round(float(correct[kept].mean()), 4),
        "cv_coverage": round(float(kept.mean()), 4),
        "target_accuracy": args.target_accuracy,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.model_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
