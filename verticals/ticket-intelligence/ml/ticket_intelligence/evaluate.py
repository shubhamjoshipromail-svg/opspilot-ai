"""Print a compact evaluation report for trained ticket intelligence artifacts."""

from __future__ import annotations

import json
from pathlib import Path


METRICS_PATH = Path(__file__).resolve().parent / "outputs" / "metrics.json"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("No metrics found. Run train_baseline.py first.")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    summary = {
        "model_version": metrics["model_version"],
        "baseline": metrics["baseline"],
        "data": metrics.get("data"),
        "category_accuracy": metrics["category"]["accuracy"],
        "category_macro_f1": metrics["category"]["macro_f1"],
        "error_analysis": "outputs/error_analysis.csv",
        "confusion_matrix": "outputs/confusion_matrix.png",
    }
    if metrics.get("priority"):
        summary["priority_accuracy"] = metrics["priority"]["accuracy"]
        summary["priority_macro_f1"] = metrics["priority"]["macro_f1"]
    else:
        summary["priority"] = "not_trained_missing_or_single_class_labels"
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
