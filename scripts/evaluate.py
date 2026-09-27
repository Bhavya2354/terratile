import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from app.classifier import Classifier
from app.config import ROOT, Settings
from scripts.dataset import labelled_csv, load_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the classifier on the held-out eval_set.")
    parser.add_argument("--dataset", type=Path, required=True, help="path to be-mlsys-assignment-dataset")
    parser.add_argument("--model-dir", type=Path, default=Settings().model_dir)
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    clf = Classifier(args.model_dir)
    samples = labelled_csv(args.dataset / "eval_set", args.dataset / "eval_labels.csv")
    truth = np.array([y for _, y in samples])
    preds = [p for i in range(0, len(samples), 64)
             for p in clf.predict_many(load_images([s for s, _ in samples[i:i + 64]]))]
    predicted = np.array([p.label for p in preds])
    confidence = np.array([p.confidence for p in preds])
    correct = predicted == truth
    kept = confidence >= clf.review_threshold

    report = {
        "model_version": clf.version,
        "review_threshold": clf.review_threshold,
        "n": len(samples),
        "accuracy": round(float(correct.mean()), 4),
        "coverage": round(float(kept.mean()), 4),
        "accepted_accuracy": round(float(correct[kept].mean()), 4) if kept.any() else None,
        "needs_review_accuracy": round(float(correct[~kept].mean()), 4) if (~kept).any() else None,
        "per_class": classification_report(truth, predicted, labels=clf.classes, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": clf.classes,
                             "rows_true_cols_pred": confusion_matrix(truth, predicted, labels=clf.classes).tolist()},
        "mistakes": [{"file": s.name, "true": t, "pred": p.label, "confidence": round(p.confidence, 4)}
                     for (s, t), p, ok in zip(samples, preds, correct) if not ok],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval.json").write_text(json.dumps(report, indent=2))
    print(classification_report(truth, predicted, labels=clf.classes, zero_division=0))
    print({k: report[k] for k in ("model_version", "accuracy", "review_threshold", "coverage",
                                  "accepted_accuracy", "needs_review_accuracy")})


if __name__ == "__main__":
    main()
