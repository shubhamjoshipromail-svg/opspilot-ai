from sqlalchemy.orm import Session

from backend.app.models import TicketRoutingPrediction
from backend.app.services.ticket_intelligence import RoutePrediction


def save_routing_prediction(
    db: Session,
    *,
    subject: str,
    body: str,
    prediction: RoutePrediction,
    ticket_id: int | None = None,
) -> TicketRoutingPrediction:
    """Persist a successful routing prediction."""
    record = TicketRoutingPrediction(
        ticket_id=ticket_id,
        subject=subject,
        body=body,
        predicted_queue=prediction.predicted_queue,
        confidence=prediction.confidence,
        threshold=prediction.threshold,
        decision=prediction.decision,
        probabilities=dict(prediction.probabilities),
        model_id=prediction.model_id,
        latency_ms=prediction.latency_ms,
        created_at=prediction.timestamp,
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception:
        db.rollback()
        raise
    return record
