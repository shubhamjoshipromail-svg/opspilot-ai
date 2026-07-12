from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


EXPECTED_QUEUE_LABELS = (
    "billing_and_payments",
    "customer_general",
    "human_resources",
    "returns_and_exchanges",
    "sales_and_pre_sales",
    "service_outages_and_maintenance",
    "technical_product_support",
)

QueueLabel = Literal[
    "billing_and_payments",
    "customer_general",
    "human_resources",
    "returns_and_exchanges",
    "sales_and_pre_sales",
    "service_outages_and_maintenance",
    "technical_product_support",
]
RoutingDecision = Literal["auto_route", "human_review"]


class TicketRouteRequest(BaseModel):
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class TicketRouteResponse(BaseModel):
    predicted_queue: QueueLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    decision: RoutingDecision
    threshold: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[QueueLabel, float]
    model_id: str
    latency_ms: int = Field(..., ge=0)
    timestamp: datetime

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(
        cls,
        probabilities: dict[QueueLabel, float],
    ) -> dict[QueueLabel, float]:
        if set(probabilities) != set(EXPECTED_QUEUE_LABELS):
            raise ValueError("probabilities must contain exactly the seven supported queue labels")
        if any(probability < 0.0 or probability > 1.0 for probability in probabilities.values()):
            raise ValueError("probability values must be between 0.0 and 1.0")
        return probabilities
