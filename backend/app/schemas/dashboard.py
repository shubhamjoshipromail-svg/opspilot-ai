from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.ticket_intelligence import QueueLabel, RoutingDecision


class DashboardSummary(BaseModel):
    total_tickets: int = Field(..., ge=0)
    queue_distribution: dict[QueueLabel, int]
    decision_counts: dict[RoutingDecision, int]
    average_confidence: float | None
    total_predictions: int = Field(..., ge=0)


class RecentRoutingPrediction(BaseModel):
    id: int
    ticket_id: int | None
    subject: str
    body: str
    predicted_queue: QueueLabel
    confidence: float
    threshold: float
    decision: RoutingDecision
    probabilities: dict[QueueLabel, float]
    model_id: str
    latency_ms: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
