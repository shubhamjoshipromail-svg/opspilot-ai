"""Escalation-risk scoring and human-in-the-loop routing for ticket triage."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


MODEL_VERSION = "ticket-intelligence-v1"

ANGRY_TERMS = {
    "angry",
    "furious",
    "unacceptable",
    "ignored",
    "terrible",
    "escalating",
    "escalate",
    "vp",
    "executive",
}
BILLING_TERMS = {"refund", "charged", "charge", "billing", "invoice", "dispute", "credit", "reimbursement"}
URGENCY_TERMS = {"today", "immediately", "urgent", "blocked", "blocking", "production", "deadline", "one hour"}
REPEAT_TERMS = {"again", "second", "twice", "three calls", "repeat", "keeps", "already"}
LEGAL_TERMS = {"legal", "attorney", "compliance", "gdpr", "hipaa", "breach", "contract", "dpa", "soc 2"}
SENSITIVE_CATEGORIES = {"compliance_request", "billing_dispute", "cancellation"}


@dataclass(frozen=True)
class RoutingResult:
    escalation_risk: float
    routing_decision: str
    reason: str
    risk_signals: list[str]


def _contains_any(text: str, terms: Iterable[str]) -> list[str]:
    found: list[str] = []
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text):
            found.append(term)
    return sorted(found)


def score_escalation_risk(text: str, category: str | None = None, priority: str | None = None) -> tuple[float, list[str]]:
    normalized = text.lower()
    signals: list[str] = []
    score = 0.12

    signal_groups = [
        ("angry language", ANGRY_TERMS, 0.19),
        ("billing/refund dispute", BILLING_TERMS, 0.15),
        ("urgency/business impact", URGENCY_TERMS, 0.18),
        ("repeat issue", REPEAT_TERMS, 0.14),
        ("legal/compliance sensitivity", LEGAL_TERMS, 0.2),
    ]
    for label, terms, weight in signal_groups:
        matches = _contains_any(normalized, terms)
        if matches:
            score += weight
            signals.append(f"{label}: {', '.join(matches[:3])}")

    if category in SENSITIVE_CATEGORIES:
        score += 0.08
        signals.append(f"sensitive category: {category}")
    if priority == "high":
        score += 0.12
        signals.append("high predicted priority")
    elif priority == "medium":
        score += 0.05
        signals.append("medium predicted priority")

    return round(min(score, 0.99), 2), signals


def route_ticket(
    text: str,
    category: str,
    priority: str,
    confidence: float,
    low_confidence_threshold: float = 0.58,
    high_risk_threshold: float = 0.72,
) -> RoutingResult:
    risk, signals = score_escalation_risk(text, category, priority)

    if category == "compliance_request" and risk >= 0.55:
        decision = "supervisor_review"
        reason = "Policy-sensitive ticket with compliance/legal signals"
    elif confidence < low_confidence_threshold:
        decision = "human_review"
        reason = "Low model confidence requires human validation"
    elif risk >= high_risk_threshold:
        decision = "priority_queue"
        reason = "High escalation risk based on language, impact, or sensitive terms"
    elif priority == "high":
        decision = "human_review"
        reason = "High-priority ticket should be checked before action"
    else:
        decision = "auto_triage_suggestion"
        reason = "High enough confidence with low escalation risk"

    return RoutingResult(
        escalation_risk=risk,
        routing_decision=decision,
        reason=reason,
        risk_signals=signals,
    )
