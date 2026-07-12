"""Create weak policy-derived risk labels for ticket-intelligence experiments.

These labels are not observed escalation outcomes. They are candidate labels
derived from ticket priority/type/tags/text signals for controlled experiments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_utils import DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, DEFAULT_VAL_PATH, OUTPUT_DIR


TAG_RISK_TERMS = {
    "security",
    "outage",
    "disruption",
    "data breach",
    "fraud",
    "compliance",
    "legal",
}
TEXT_RISK_TERMS = {
    "legal",
    "lawsuit",
    "chargeback",
    "fraud",
    "regulator",
    "complaint",
    "urgent",
    "outage",
    "data breach",
    "account locked",
    "cannot access",
    "safety",
    "compliance",
    "refund",
    "duplicate charge",
    "charged twice",
}


def _contains_any(text: str, terms: Iterable[str]) -> list[str]:
    normalized = str(text).lower()
    matches: list[str] = []
    for term in terms:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, normalized):
            matches.append(term)
    return sorted(matches)


def add_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "model_text" not in output.columns:
        raise ValueError("Risk labeling requires model_text. Rebuild the dataset and recreate splits first.")

    priority = output.get("priority", output.get("true_priority", "")).fillna("").astype(str).str.lower()
    ticket_type = output.get("type", output.get("ticket_type", "")).fillna("").astype(str).str.lower()
    tags = output.get("combined_tags", output.get("tags", "")).fillna("").astype(str)
    model_text = output["model_text"].fillna("").astype(str)

    urgency_signal = []
    policy_sensitive_signal = []
    escalation_keyword_signal = []
    high_risk_policy_label = []
    risk_signal_terms = []

    for idx in output.index:
        tag_matches = _contains_any(tags.loc[idx], TAG_RISK_TERMS)
        text_matches = _contains_any(model_text.loc[idx], TEXT_RISK_TERMS)
        is_high_priority = priority.loc[idx] == "high"
        is_incident = ticket_type.loc[idx] == "incident"
        has_policy_tag = bool(set(tag_matches) & {"security", "data breach", "fraud", "compliance", "legal"})
        has_policy_text = bool(set(text_matches) & {"legal", "lawsuit", "chargeback", "fraud", "regulator", "safety", "compliance"})
        has_escalation_text = bool(text_matches)

        urgency = bool(is_high_priority or is_incident or "urgent" in text_matches or "outage" in text_matches)
        policy = bool(has_policy_tag or has_policy_text)
        escalation = bool(has_escalation_text or tag_matches)
        high_risk = bool(policy or escalation or (is_high_priority and is_incident))

        urgency_signal.append(int(urgency))
        policy_sensitive_signal.append(int(policy))
        escalation_keyword_signal.append(int(escalation))
        high_risk_policy_label.append(int(high_risk))
        terms = []
        if is_high_priority:
            terms.append("priority:high")
        if is_incident:
            terms.append("type:incident")
        terms.extend([f"tag:{term}" for term in tag_matches])
        terms.extend([f"text:{term}" for term in text_matches])
        risk_signal_terms.append("; ".join(sorted(set(terms))))

    output["urgency_signal"] = urgency_signal
    output["policy_sensitive_signal"] = policy_sensitive_signal
    output["escalation_keyword_signal"] = escalation_keyword_signal
    output["high_risk_policy_label"] = high_risk_policy_label
    output["risk_signal_terms"] = risk_signal_terms
    return output


def load_all_splits() -> pd.DataFrame:
    frames = []
    for split_name, path in [("train", DEFAULT_TRAIN_PATH), ("val", DEFAULT_VAL_PATH), ("test", DEFAULT_TEST_PATH)]:
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}. Run create_splits.py first.")
        df = pd.read_csv(path)
        df["split"] = split_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def create_risk_label_outputs() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labeled = add_risk_labels(load_all_splits())
    audit_columns = [
        column
        for column in [
            "split",
            "ticket_id",
            "external_id",
            "customer_message",
            "model_text",
            "category",
            "priority",
            "type",
            "combined_tags",
            "urgency_signal",
            "policy_sensitive_signal",
            "escalation_keyword_signal",
            "high_risk_policy_label",
            "risk_signal_terms",
        ]
        if column in labeled.columns
    ]
    audit_path = OUTPUT_DIR / "risk_label_audit.csv"
    labeled[audit_columns].to_csv(audit_path, index=False)

    summary = {
        "label_type": "weak_policy_derived",
        "not_ground_truth": True,
        "rows": int(len(labeled)),
        "split_counts": labeled["split"].value_counts().to_dict(),
        "high_risk_policy_label_distribution": labeled["high_risk_policy_label"].value_counts().sort_index().to_dict(),
        "urgency_signal_distribution": labeled["urgency_signal"].value_counts().sort_index().to_dict(),
        "policy_sensitive_signal_distribution": labeled["policy_sensitive_signal"].value_counts().sort_index().to_dict(),
        "escalation_keyword_signal_distribution": labeled["escalation_keyword_signal"].value_counts().sort_index().to_dict(),
        "rules": {
            "urgency_signal": "priority == high, type == Incident, or text contains urgent/outage.",
            "policy_sensitive_signal": "tags/text contain security, data breach, fraud, compliance, legal, lawsuit, chargeback, regulator, safety.",
            "escalation_keyword_signal": "tags/text contain configured risk keywords.",
            "high_risk_policy_label": "policy-sensitive, escalation keyword, or high-priority incident.",
        },
    }
    summary_path = OUTPUT_DIR / "risk_label_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"risk_label_audit": str(audit_path), "risk_label_summary": str(summary_path), **summary}


def main() -> None:
    print(json.dumps(create_risk_label_outputs(), indent=2))


if __name__ == "__main__":
    main()
