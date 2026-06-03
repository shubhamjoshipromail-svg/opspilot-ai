"""Generate richer test-set error analysis for Ticket Intelligence models."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from data_utils import ARTIFACT_DIR, DEFAULT_TEST_PATH, OUTPUT_DIR, canonicalize_ticket_frame


AMBIGUOUS_TERMS = {"help", "issue", "problem", "question", "request", "support", "not working"}
MULTIPLE_INTENT_TERMS = {"and", "also", "another", "refund", "login", "billing", "order", "account"}
METADATA_PRIORITY_TERMS = {"priority", "urgent", "asap", "today", "deadline", "blocked"}


def load_model(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_with_confidence(model, text: str) -> tuple[str | None, float | None]:
    if model is None:
        return None, None
    prediction = str(model.predict([text])[0])
    confidence = 0.5
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba([text])[0].max())
    return prediction, round(confidence, 4)


def likely_reason(row: pd.Series) -> str:
    text = str(row.get("customer_message", "")).lower()
    category_confidence = row.get("category_confidence")
    priority_confidence = row.get("priority_confidence")
    if category_confidence is not None and category_confidence < 0.45:
        return "low_confidence"
    if priority_confidence is not None and priority_confidence < 0.45:
        return "priority_depends_on_metadata"
    if sum(term in text for term in MULTIPLE_INTENT_TERMS) >= 3:
        return "multiple_intents"
    if any(term in text for term in AMBIGUOUS_TERMS) and len(text.split()) < 45:
        return "ambiguous_text"
    if row.get("category") in {"customer_service", "product_support", "technical_support"}:
        return "label_overlap"
    if any(term in text for term in METADATA_PRIORITY_TERMS):
        return "priority_depends_on_metadata"
    if len(text.split()) < 20:
        return "insufficient_examples"
    return "possible_label_noise"


def build_error_analysis() -> pd.DataFrame:
    test_df = canonicalize_ticket_frame(pd.read_csv(DEFAULT_TEST_PATH))
    category_model = load_model(ARTIFACT_DIR / "category_model.pkl")
    priority_model = load_model(ARTIFACT_DIR / "priority_model.pkl")

    rows: list[dict] = []
    for _, row in test_df.iterrows():
        text = row["customer_message"]
        predicted_category, category_confidence = predict_with_confidence(category_model, text)
        predicted_priority, priority_confidence = predict_with_confidence(priority_model, text)
        category_correct = predicted_category == row["category"] if predicted_category is not None else True
        priority_correct = predicted_priority == row["priority"] if predicted_priority is not None else True
        if category_correct and priority_correct:
            continue

        output = {
            "external_id": row.get("external_id", row.get("ticket_id")),
            "ticket_id": row.get("ticket_id", row.get("external_id")),
            "customer_message": text,
            "true_category": row.get("true_category", row.get("category")),
            "predicted_category": predicted_category,
            "category_confidence": category_confidence,
            "true_priority": row.get("true_priority", row.get("priority")),
            "predicted_priority": predicted_priority,
            "priority_confidence": priority_confidence,
            "category_correct": category_correct,
            "priority_correct": priority_correct,
        }
        output["error_type"] = likely_reason(pd.Series({**row.to_dict(), **output}))
        output["likely_reason"] = output["error_type"]
        rows.append(output)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = build_error_analysis()
    path = OUTPUT_DIR / "error_analysis.csv"
    errors.to_csv(path, index=False)
    print(json.dumps({"error_analysis": str(path), "rows": int(len(errors))}, indent=2))


if __name__ == "__main__":
    main()
