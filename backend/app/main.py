from fastapi import FastAPI

from backend.app.routes import dashboard_router, ticket_intelligence_router, tickets_router


app = FastAPI(
    title="OpsPilot AI API",
    description="Backend API for the OpsPilot AI operations copilot.",
    version="0.1.0",
)

app.include_router(tickets_router)
app.include_router(ticket_intelligence_router)
app.include_router(dashboard_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Minimal scaffold health check."""
    return {"status": "ok", "service": "opspilot-ai"}


# TODO: Add service orchestration for triage, risk, retrieval, recommendations, and evaluation.
