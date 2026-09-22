"""Structured prediction demo for OpsPilot Ticket Intelligence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any

from data_utils import ARTIFACT_DIR, MODEL_VERSION, OUTPUT_DIR
from routing import route_ticket


CATEGORY_MODEL_PATH = ARTIFACT_DIR / "category_model.pkl"
PRIORITY_MODEL_PATH = ARTIFACT_DIR / "priority_model.pkl"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"
DEFAULT_DEMO_TEXT = (
    "I was charged twice and support keeps closing my tickets. "
    "If this refund is not handled today I am escalating to legal."
)


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}. Run train_baseline.py first.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_with_confidence(model, text: str) -> tuple[str, float, dict[str, float]]:
    prediction = str(model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        distribution = {
            str(class_label): round(float(probability), 4)
            for class_label, probability in zip(model.classes_, probabilities)
        }
        return prediction, round(float(probabilities.max()), 4), distribution
    return prediction, 0.5, {prediction: 0.5}


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {
        "model_version": MODEL_VERSION,
        "model_type": "tfidf_logistic_regression_baseline",
    }


def predict_ticket(text: str, ticket_id: str = "demo-001", save: bool = True) -> dict[str, Any]:
    category_model = load_model(CATEGORY_MODEL_PATH)
    priority_model = load_model(PRIORITY_MODEL_PATH)
    metadata = load_metadata()

    raw_category, category_confidence, category_distribution = predict_with_confidence(category_model, text)
    priority, priority_confidence, priority_distribution = predict_with_confidence(priority_model, text)
    routing = route_ticket(
        text=text,
        category=raw_category,
        priority=priority,
        category_confidence=category_confidence,
        priority_confidence=priority_confidence,
    )

    result = {
        "ticket_id": ticket_id,
        "customer_message": text,
        "category": {
            "prediction": raw_category,
            "confidence": category_confidence,
            "raw_model_prediction": raw_category,
            "fallback_applied": False,
            "distribution": category_distribution,
        },
        "priority": {
            "prediction": priority,
            "confidence": priority_confidence,
            "distribution": priority_distribution,
        },
        "risk": {
            "escalation_risk": routing.escalation_risk,
            "risk_signals": routing.risk_signals,
            "policy_sensitive": routing.policy_sensitive,
        },
        "routing": {
            "decision": routing.routing_decision,
            "reason": routing.reason,
        },
        "model_metadata": {
            "model_version": metadata.get("model_version", MODEL_VERSION),
            "model_type": metadata.get("model_type", "tfidf_logistic_regression_baseline"),
            "dataset_source": metadata.get("data", {}).get("dataset_source"),
        },
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default=DEFAULT_DEMO_TEXT)
    parser.add_argument("--ticket-id", default="demo-001")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    print(json.dumps(predict_ticket(args.text, args.ticket_id, save=not args.no_save), indent=2))


if __name__ == "__main__":
    main()
