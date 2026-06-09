from pydantic import BaseModel, Field


class TicketRouteRequest(BaseModel):
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class TicketRouteResponse(BaseModel):
    predicted_queue: str
    confidence: float
    decision: str
    threshold: float
    class_probabilities: dict[str, float]
