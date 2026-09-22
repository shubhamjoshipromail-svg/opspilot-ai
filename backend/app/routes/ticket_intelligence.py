from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.ticket_intelligence import TicketRouteRequest, TicketRouteResponse
from backend.app.services.routing_predictions import save_routing_prediction
from backend.app.services.ticket_intelligence import TicketIntelligenceUnavailableError, predict_route


router = APIRouter(prefix="/api/ticket-intelligence", tags=["ticket-intelligence"])


@router.post("/route", response_model=TicketRouteResponse)
def route_ticket(
    request: TicketRouteRequest,
    db: Session = Depends(get_db),
) -> TicketRouteResponse:
    try:
        prediction = predict_route(request.subject, request.body)
    except TicketIntelligenceUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ticket Intelligence service unavailable: {exc}",
        ) from exc

    save_routing_prediction(
        db,
        subject=request.subject,
        body=request.body,
        prediction=prediction,
    )

    return TicketRouteResponse(
        predicted_queue=prediction.predicted_queue,
        confidence=prediction.confidence,
        decision=prediction.decision,
        threshold=prediction.threshold,
        probabilities=prediction.probabilities,
        model_id=prediction.model_id,
        latency_ms=prediction.latency_ms,
        timestamp=prediction.timestamp,
    )
