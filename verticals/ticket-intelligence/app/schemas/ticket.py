from __future__ import annotations

from pydantic import BaseModel, Field


class TicketTriageRequest(BaseModel):
    customer_message: str = Field(..., min_length=3)


class TicketTriageResponse(BaseModel):
    ticket_id: str | None
    category: str
    priority: str
    escalation_risk: float
    confidence: float
    routing_decision: str
    reason: str
    risk_signals: list[str]
    model_version: str
    model_family: str
    metadata: dict
