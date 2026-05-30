"""Prediction entry point for OpsPilot Ticket Intelligence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any

from routing import MODEL_VERSION, route_ticket


MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_DIR / "outputs"
CATEGORY_MODEL_PATH = OUTPUT_DIR / "baseline_category.pkl"
PRIORITY_MODEL_PATH = OUTPUT_DIR / "baseline_priority.pkl"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"

CATEGORY_HINTS = {
    "billing_dispute": {"refund", "charged", "charge", "invoice", "billing", "dispute", "credit", "reimbursement"},
    "technical_issue": {"api", "error", "failed", "crash", "500", "webhook", "integration", "dashboard", "sync"},
    "account_access": {"login", "locked", "password", "mfa", "sso", "admin", "invite", "user", "portal"},
    "cancellation": {"cancel", "cancelled", "cancellation", "downgrade", "renewal", "subscription", "leaving"},
    "shipping_delay": {"shipment", "package", "tracking", "delivery", "delivered", "carrier", "parcel", "replacement"},
    "compliance_request": {"legal", "compliance", "gdpr", "hipaa", "soc 2", "dpa", "privacy", "audit", "breach"},
}


def _load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}. Run train_baseline.py first.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _predict_with_confidence(model, text: str) -> tuple[str, float, dict[str, float]]:
    label = str(model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        classes = [str(value) for value in model.classes_]
        distribution = {klass: round(float(prob), 4) for klass, prob in zip(classes, probabilities)}
        return label, round(float(max(probabilities)), 4), distribution
    return label, 0.5, {label: 0.5}


def _category_hint(text: str) -> tuple[str | None, int]:
    normalized = text.lower()
    scores: dict[str, int] = {}
    for category, terms in CATEGORY_HINTS.items():
        scores[category] = sum(1 for term in terms if term in normalized)
    category, score = max(scores.items(), key=lambda item: item[1])
    return (category, score) if score >= 2 else (None, score)


def predict_ticket(text: str, ticket_id: str | None = None, save: bool = True) -> dict[str, Any]:
    category_model = _load_model(CATEGORY_MODEL_PATH)
    priority_model = _load_model(PRIORITY_MODEL_PATH)

    raw_category, category_confidence, category_distribution = _predict_with_confidence(category_model, text)
    priority, priority_confidence, priority_distribution = _predict_with_confidence(priority_model, text)
    hinted_category, hint_score = _category_hint(text)
    category_source = "model"
    category = raw_category
    if category_confidence < 0.4 and hinted_category and hinted_category != raw_category:
        category = hinted_category
        category_source = "low_confidence_lexical_fallback"
    confidence = round(min(category_confidence, priority_confidence), 4)
    routing = route_ticket(text=text, category=category, priority=priority, confidence=confidence)

    result: dict[str, Any] = {
        "ticket_id": ticket_id,
        "category": category,
        "priority": priority,
        "escalation_risk": routing.escalation_risk,
        "confidence": confidence,
        "routing_decision": routing.routing_decision,
        "reason": routing.reason,
        "risk_signals": routing.risk_signals,
        "model_version": MODEL_VERSION,
        "model_family": "tfidf_logistic_regression_baseline",
        "metadata": {
            "raw_model_category": raw_category,
            "category_source": category_source,
            "category_hint_score": hint_score,
            "category_confidence": category_confidence,
            "priority_confidence": priority_confidence,
            "category_distribution": category_distribution,
            "priority_distribution": priority_distribution,
        },
    }
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Raw customer ticket message")
    parser.add_argument("--ticket-id", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    print(json.dumps(predict_ticket(args.text, ticket_id=args.ticket_id, save=not args.no_save), indent=2))


if __name__ == "__main__":
    main()
