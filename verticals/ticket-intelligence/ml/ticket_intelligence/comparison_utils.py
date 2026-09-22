"""Utilities for comparing baseline and TensorFlow ticket models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_utils import OUTPUT_DIR


def _metric_summary(payload: dict[str, Any], target: str | None = None) -> dict[str, Any] | None:
    source = payload.get(target, payload) if target else payload
    if not isinstance(source, dict):
        return None
    if "accuracy" not in source:
        return None
    return {
        "accuracy": source.get("accuracy"),
        "macro_f1": source.get("macro_f1"),
        "weighted_f1": source.get("weighted_f1"),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_comparison() -> dict[str, Any]:
    baseline_metrics = _read_json(OUTPUT_DIR / "metrics.json") or {}
    tensorflow_category = _read_json(OUTPUT_DIR / "tensorflow_category_metrics.json")
    tensorflow_priority = _read_json(OUTPUT_DIR / "tensorflow_priority_metrics.json")
    tensorflow_risk = _read_json(OUTPUT_DIR / "tensorflow_risk_metrics.json")

    comparison: dict[str, Any] = {
        "notes": {
            "baseline": "TF-IDF + Logistic Regression benchmark on fixed splits.",
            "tensorflow": "Keras TextVectorization + Embedding text classifiers trained on model_text.",
            "leakage_prevention": "TextVectorization is adapted on train text only; validation is used for early stopping; test is used only for final evaluation.",
            "risk_label_quality": "Risk labels are weak policy-derived labels, not observed escalation outcomes.",
        },
        "baseline_category": _metric_summary(baseline_metrics, "category"),
        "baseline_priority": _metric_summary(baseline_metrics, "priority"),
        "tensorflow_category": _metric_summary(tensorflow_category or {}),
        "tensorflow_priority": _metric_summary(tensorflow_priority or {}),
        "tensorflow_risk": _metric_summary(tensorflow_risk or {}),
    }

    baseline_category = comparison.get("baseline_category")
    tf_category = comparison.get("tensorflow_category")
    if baseline_category and tf_category:
        comparison["delta_tensorflow_category_minus_baseline"] = {
            key: round(float(tf_category[key]) - float(baseline_category[key]), 4)
            for key in ["accuracy", "macro_f1", "weighted_f1"]
            if baseline_category.get(key) is not None and tf_category.get(key) is not None
        }

    baseline_priority = comparison.get("baseline_priority")
    tf_priority = comparison.get("tensorflow_priority")
    if baseline_priority and tf_priority:
        comparison["delta_tensorflow_priority_minus_baseline"] = {
            key: round(float(tf_priority[key]) - float(baseline_priority[key]), 4)
            for key in ["accuracy", "macro_f1", "weighted_f1"]
            if baseline_priority.get(key) is not None and tf_priority.get(key) is not None
        }

    return comparison


def write_model_comparison() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "model_comparison.json"
    path.write_text(json.dumps(build_model_comparison(), indent=2), encoding="utf-8")
    return path
