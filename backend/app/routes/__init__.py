"""API routers exposed for application startup."""

from backend.app.routes.dashboard import router as dashboard_router
from backend.app.routes.ticket_intelligence import router as ticket_intelligence_router
from backend.app.routes.tickets import router as tickets_router

__all__ = ["dashboard_router", "ticket_intelligence_router", "tickets_router"]
