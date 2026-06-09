from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.ticket_intelligence import TicketRouteRequest, TicketRouteResponse
from backend.app.services.ticket_intelligence import predict_route


router = APIRouter(prefix="/api/ticket-intelligence", tags=["ticket-intelligence"])


@router.post("/route", response_model=TicketRouteResponse)
def route_ticket(request: TicketRouteRequest) -> TicketRouteResponse:
    try:
        prediction = predict_route(request.subject, request.body)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return TicketRouteResponse(
        predicted_queue=prediction.predicted_queue,
        confidence=prediction.confidence,
        decision=prediction.decision,
        threshold=prediction.threshold,
        class_probabilities=prediction.class_probabilities,
    )
