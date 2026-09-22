"""ORM models exposed for SQLAlchemy metadata discovery."""

from backend.app.models.ticket import Ticket
from backend.app.models.ticket_routing_prediction import TicketRoutingPrediction

__all__ = ["Ticket", "TicketRoutingPrediction"]
