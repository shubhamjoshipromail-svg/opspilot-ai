"""Build a model leaderboard for OpsPilot Ticket Intelligence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_utils import ARTIFACT_DIR, OUTPUT_DIR


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def add_row(
    rows: list[dict[str, Any]],
    task: str,
    model_name: str,
    metrics: dict[str, Any] | None,
    notes: str,
    artifact_path: Path | str | None,
    metrics_path: Path,
) -> None:
    if not metrics:
        return
    rows.append(
        {
            "task": task,
            "model_name": model_name,
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "notes": notes,
            "artifact_path": str(artifact_path) if artifact_path else "",
            "metrics_path": str(metrics_path),
        }
    )


def build_leaderboard() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    baseline_path = OUTPUT_DIR / "metrics.json"
    baseline = read_json(baseline_path)
    if baseline:
        add_row(
            rows,
            "parent_queue",
            "tfidf_logistic_regression",
            baseline.get("category"),
            "V1 interpretable baseline using customer_message.",
            ARTIFACT_DIR / "category_model.pkl",
            baseline_path,
        )
        add_row(
            rows,
            "priority",
            "tfidf_logistic_regression",
            baseline.get("priority"),
            "V1 interpretable priority baseline using customer_message.",
            ARTIFACT_DIR / "priority_model.pkl",
            baseline_path,
        )

    tensorflow_specs = [
        ("parent_queue", "tensorflow_cnn_category", OUTPUT_DIR / "tensorflow_category_metrics.json", ARTIFACT_DIR / "tensorflow_category_model.keras"),
        ("priority", "tensorflow_cnn_priority", OUTPUT_DIR / "tensorflow_priority_metrics.json", ARTIFACT_DIR / "tensorflow_priority_model.keras"),
        ("weak_policy_risk", "tensorflow_cnn_weak_risk", OUTPUT_DIR / "tensorflow_risk_metrics.json", ARTIFACT_DIR / "tensorflow_risk_model.keras"),
    ]
    for task, model_name, metrics_path, artifact_path in tensorflow_specs:
        metrics = read_json(metrics_path)
        add_row(
            rows,
            task,
            model_name,
            metrics,
            "V1 TensorFlow/Keras CNN text classifier; risk labels are weak policy-derived when present.",
            artifact_path,
            metrics_path,
        )

    modernbert_specs = [
        ("parent_queue", OUTPUT_DIR / "modernbert_parent_queue_metrics.json", ARTIFACT_DIR / "modernbert_parent_queue"),
        ("priority", OUTPUT_DIR / "modernbert_priority_metrics.json", ARTIFACT_DIR / "modernbert_priority"),
        ("ticket_type", OUTPUT_DIR / "modernbert_ticket_type_metrics.json", ARTIFACT_DIR / "modernbert_ticket_type"),
        ("raw_queue", OUTPUT_DIR / "modernbert_raw_queue_metrics.json", ARTIFACT_DIR / "modernbert_raw_queue"),
    ]
    for task, metrics_path, artifact_path in modernbert_specs:
        metrics = read_json(metrics_path)
        add_row(
            rows,
            task,
            metrics.get("model_name", "answerdotai/ModernBERT-base") if metrics else "answerdotai/ModernBERT-base",
            metrics,
            "V2 transformer sequence classifier using fixed splits; answer/reference_answer excluded.",
            artifact_path,
            metrics_path,
        )

    leaderboard = pd.DataFrame(
        rows,
        columns=["task", "model_name", "accuracy", "macro_f1", "weighted_f1", "notes", "artifact_path", "metrics_path"],
    )
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(["task", "macro_f1", "weighted_f1"], ascending=[True, False, False])
    return leaderboard


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = build_leaderboard()
    csv_path = OUTPUT_DIR / "model_leaderboard.csv"
    json_path = OUTPUT_DIR / "model_leaderboard.json"
    leaderboard.to_csv(csv_path, index=False)
    payload = leaderboard.to_dict(orient="records")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": len(payload)}, indent=2))


if __name__ == "__main__":
    main()
