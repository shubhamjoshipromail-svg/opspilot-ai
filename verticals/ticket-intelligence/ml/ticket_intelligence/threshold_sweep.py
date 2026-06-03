"""Sweep confidence thresholds for auto-triage versus human review tradeoffs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data_utils import ARTIFACT_DIR, DEFAULT_TEST_PATH, OUTPUT_DIR, canonicalize_ticket_frame
from routing import detect_policy_sensitivity, score_escalation_risk


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}. Run train_baseline.py first.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_frame() -> pd.DataFrame:
    test_df = canonicalize_ticket_frame(pd.read_csv(DEFAULT_TEST_PATH))
    category_model = load_model(ARTIFACT_DIR / "category_model.pkl")
    priority_model = load_model(ARTIFACT_DIR / "priority_model.pkl")

    category_predictions = category_model.predict(test_df["customer_message"])
    category_probabilities = category_model.predict_proba(test_df["customer_message"])
    priority_predictions = priority_model.predict(test_df["customer_message"])
    priority_probabilities = priority_model.predict_proba(test_df["customer_message"])

    rows = test_df.copy()
    rows["predicted_category"] = category_predictions
    rows["category_confidence"] = category_probabilities.max(axis=1)
    rows["predicted_priority"] = priority_predictions
    rows["priority_confidence"] = priority_probabilities.max(axis=1)

    risks = []
    policy_flags = []
    for _, row in rows.iterrows():
        risk, _ = score_escalation_risk(
            row["customer_message"],
            row["predicted_category"],
            row["predicted_priority"],
            row["category_confidence"],
            row["priority_confidence"],
        )
        policy_sensitive, _ = detect_policy_sensitivity(row["customer_message"], row["predicted_category"])
        risks.append(risk)
        policy_flags.append(policy_sensitive)
    rows["escalation_risk"] = risks
    rows["policy_sensitive"] = policy_flags
    return rows


def sweep_thresholds() -> pd.DataFrame:
    df = predict_frame()
    rows: list[dict] = []
    for threshold in [round(value / 100, 2) for value in range(50, 100, 5)]:
        auto_mask = (
            (df["category_confidence"] >= threshold)
            & (df["escalation_risk"] < 0.80)
            & (~df["policy_sensitive"])
        )
        auto_df = df[auto_mask]
        if len(auto_df) > 0:
            accuracy = accuracy_score(auto_df["category"], auto_df["predicted_category"])
            macro_f1 = f1_score(auto_df["category"], auto_df["predicted_category"], average="macro")
        else:
            accuracy = None
            macro_f1 = None
        rows.append(
            {
                "threshold": threshold,
                "auto_triage_count": int(auto_mask.sum()),
                "human_review_count": int((~auto_mask).sum()),
                "auto_triage_rate": round(float(auto_mask.mean()), 4),
                "human_review_rate": round(float((~auto_mask).mean()), 4),
                "accuracy_on_auto_triaged": round(float(accuracy), 4) if accuracy is not None else None,
                "macro_f1_on_auto_triaged": round(float(macro_f1), 4) if macro_f1 is not None else None,
                "high_risk_auto_triaged_count": int((auto_mask & (df["escalation_risk"] >= 0.80)).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = sweep_thresholds()
    path = OUTPUT_DIR / "threshold_sweep.csv"
    results.to_csv(path, index=False)
    print(json.dumps({"threshold_sweep": str(path), "rows": int(len(results))}, indent=2))


if __name__ == "__main__":
    main()
