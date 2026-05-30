from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[2]
TICKET_INTELLIGENCE_DIR = ROOT / "ml" / "ticket_intelligence"
sys.path.insert(0, str(TICKET_INTELLIGENCE_DIR))
sys.path.insert(0, str(ROOT))

from app.schemas.ticket import TicketTriageRequest, TicketTriageResponse
from predict import predict_ticket
from routing import MODEL_VERSION


app = FastAPI(
    title="OpsPilot Ticket Intelligence",
    version=MODEL_VERSION,
    description="AI-assisted triage and escalation-risk routing for messy operations tickets.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=TicketTriageResponse)
def predict(request: TicketTriageRequest) -> dict:
    return predict_ticket(request.customer_message)


@app.post("/tickets/{ticket_id}/triage", response_model=TicketTriageResponse)
def triage_ticket(ticket_id: str, request: TicketTriageRequest) -> dict:
    return predict_ticket(request.customer_message, ticket_id=ticket_id)
