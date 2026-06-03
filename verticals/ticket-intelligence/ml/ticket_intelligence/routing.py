"""Escalation-risk scoring and human-in-the-loop routing for ticket triage."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from data_utils import MODEL_VERSION


RISK_TERMS = {
    "legal",
    "lawsuit",
    "chargeback",
    "fraud",
    "regulator",
    "complaint",
    "angry",
    "cancel",
    "refund",
    "charged twice",
    "duplicate charge",
    "manager",
    "escalate",
    "escalating",
    "urgent",
    "today",
    "account locked",
    "cannot access",
    "safety",
    "compliance",
}
POLICY_TERMS = {
    "legal",
    "lawsuit",
    "regulator",
    "compliance",
    "privacy",
    "gdpr",
    "hipaa",
    "fraud",
    "chargeback",
    "safety",
    "contract",
}
SENSITIVE_CATEGORIES = {"billing_and_payments", "returns_and_exchanges", "human_resources"}
HIGH_PRIORITY_VALUES = {"high", "urgent", "critical"}


@dataclass(frozen=True)
class RoutingResult:
    escalation_risk: float
    routing_decision: str
    reason: str
    risk_signals: list[str]
    policy_sensitive: bool


def _contains_any(text: str, terms: Iterable[str]) -> list[str]:
    found: list[str] = []
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text):
            found.append(term)
    return sorted(found)


def detect_policy_sensitivity(text: str, category: str | None = None) -> tuple[bool, list[str]]:
    normalized = text.lower()
    signals = _contains_any(normalized, POLICY_TERMS)
    if category in SENSITIVE_CATEGORIES:
        signals.append(f"sensitive_category:{category}")
    return bool(signals), signals


def score_escalation_risk(
    text: str,
    category: str | None = None,
    priority: str | None = None,
    category_confidence: float | None = None,
    priority_confidence: float | None = None,
) -> tuple[float, list[str]]:
    normalized = text.lower()
    signals: list[str] = []
    score = 0.1

    risk_matches = _contains_any(normalized, RISK_TERMS)
    if risk_matches:
        score += min(0.45, 0.08 * len(risk_matches))
        signals.extend([f"risk_keyword:{term}" for term in risk_matches[:8]])

    policy_sensitive, policy_signals = detect_policy_sensitivity(text, category)
    if policy_sensitive:
        score += 0.18
        signals.extend([f"policy:{signal}" for signal in policy_signals[:5]])

    if priority in HIGH_PRIORITY_VALUES:
        score += 0.16
        signals.append(f"high_priority:{priority}")
    elif priority == "medium":
        score += 0.06
        signals.append("medium_priority")

    if category_confidence is not None and category_confidence < 0.65:
        score += 0.08
        signals.append("low_category_confidence")
    if priority_confidence is not None and priority_confidence < 0.55:
        score += 0.05
        signals.append("low_priority_confidence")

    return round(min(score, 0.99), 2), sorted(set(signals))


def route_ticket(
    text: str,
    category: str,
    priority: str,
    confidence: float | None = None,
    category_confidence: float | None = None,
    priority_confidence: float | None = None,
    low_confidence_threshold: float = 0.65,
    high_risk_threshold: float = 0.80,
) -> RoutingResult:
    if category_confidence is None:
        category_confidence = confidence
    risk, signals = score_escalation_risk(text, category, priority, category_confidence, priority_confidence)
    policy_sensitive, policy_signals = detect_policy_sensitivity(text, category)

    if risk >= high_risk_threshold:
        decision = "human_review"
        reason = "High escalation risk requires human review before action"
    elif policy_sensitive:
        decision = "supervisor_review"
        reason = "Policy-sensitive language or category requires supervisor review"
    elif category_confidence is not None and category_confidence < low_confidence_threshold:
        decision = "human_review"
        reason = "Low category confidence requires human validation"
    elif priority in HIGH_PRIORITY_VALUES and (priority_confidence is None or priority_confidence >= 0.55):
        decision = "priority_queue"
        reason = "High-priority prediction should be prioritized for operations review"
    else:
        decision = "auto_triage_suggestion"
        reason = "Confidence and risk are acceptable for an auto-triage suggestion"

    return RoutingResult(
        escalation_risk=risk,
        routing_decision=decision,
        reason=reason,
        risk_signals=sorted(set(signals + [f"policy:{signal}" for signal in policy_signals])),
        policy_sensitive=policy_sensitive,
    )
