"""API routers exposed for application startup."""

from backend.app.routes.ticket_intelligence import router as ticket_intelligence_router
from backend.app.routes.tickets import router as tickets_router

__all__ = ["ticket_intelligence_router", "tickets_router"]
