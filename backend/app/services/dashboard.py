from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models import Ticket, TicketRoutingPrediction
from backend.app.schemas.dashboard import DashboardSummary
from backend.app.schemas.ticket_intelligence import EXPECTED_QUEUE_LABELS


def get_dashboard_summary(db: Session) -> DashboardSummary:
    total_tickets = db.scalar(select(func.count(Ticket.id))) or 0
    total_predictions = db.scalar(select(func.count(TicketRoutingPrediction.id))) or 0
    average_confidence = db.scalar(select(func.avg(TicketRoutingPrediction.confidence)))

    queue_counts = {
        queue: count
        for queue, count in db.execute(
            select(
                TicketRoutingPrediction.predicted_queue,
                func.count(TicketRoutingPrediction.id),
            ).group_by(TicketRoutingPrediction.predicted_queue)
        )
    }
    decision_rows = {
        decision: count
        for decision, count in db.execute(
            select(
                TicketRoutingPrediction.decision,
                func.count(TicketRoutingPrediction.id),
            ).group_by(TicketRoutingPrediction.decision)
        )
    }

    return DashboardSummary(
        total_tickets=total_tickets,
        total_predictions=total_predictions,
        queue_distribution={
            label: int(queue_counts.get(label, 0))
            for label in EXPECTED_QUEUE_LABELS
        },
        decision_counts={
            "auto_route": int(decision_rows.get("auto_route", 0)),
            "human_review": int(decision_rows.get("human_review", 0)),
        },
        average_confidence=(
            round(float(average_confidence), 6)
            if average_confidence is not None
            else None
        ),
    )


def get_recent_predictions(
    db: Session,
    *,
    limit: int,
) -> list[TicketRoutingPrediction]:
    return list(
        db.scalars(
            select(TicketRoutingPrediction)
            .order_by(
                TicketRoutingPrediction.created_at.desc(),
                TicketRoutingPrediction.id.desc(),
            )
            .limit(limit)
        )
    )
