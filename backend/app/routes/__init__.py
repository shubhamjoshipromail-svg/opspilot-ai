"""API routers exposed for application startup."""

from backend.app.routes.tickets import router as tickets_router

__all__ = ["tickets_router"]
