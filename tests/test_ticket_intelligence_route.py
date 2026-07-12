from datetime import datetime, timezone

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from backend.app.main import app
from backend.app.routes import ticket_intelligence
from backend.app.schemas.ticket_intelligence import EXPECTED_QUEUE_LABELS, TicketRouteResponse
from backend.app.services.ticket_intelligence import (
    DEFAULT_ROUTE_THRESHOLD,
    MODEL_ID,
    RoutePrediction,
    TicketIntelligenceUnavailableError,
    route_threshold,
    routing_decision,
)


client = TestClient(app)
PREDICTION_TIMESTAMP = datetime(2026, 6, 10, 12, 30, tzinfo=timezone.utc)
PROBABILITIES = {
    "billing_and_payments": 0.91,
    "customer_general": 0.01,
    "human_resources": 0.01,
    "returns_and_exchanges": 0.01,
    "sales_and_pre_sales": 0.01,
    "service_outages_and_maintenance": 0.01,
    "technical_product_support": 0.04,
}


def test_ticket_intelligence_route(monkeypatch) -> None:
    def fake_predict_route(subject: str, body: str) -> RoutePrediction:
        assert subject == "Refund needed"
        assert body == "I was charged twice and need help today."
        return RoutePrediction(
            predicted_queue="billing_and_payments",
            confidence=0.91,
            decision="auto_route",
            threshold=0.8,
            probabilities=PROBABILITIES,
            model_id=MODEL_ID,
            latency_ms=42,
            timestamp=PREDICTION_TIMESTAMP,
        )

    monkeypatch.setattr(ticket_intelligence, "predict_route", fake_predict_route)

    response = client.post(
        "/api/ticket-intelligence/route",
        json={
            "subject": "Refund needed",
            "body": "I was charged twice and need help today.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "predicted_queue": "billing_and_payments",
        "confidence": 0.91,
        "decision": "auto_route",
        "threshold": 0.8,
        "probabilities": PROBABILITIES,
        "model_id": MODEL_ID,
        "latency_ms": 42,
        "timestamp": "2026-06-10T12:30:00Z",
    }

    recent_response = client.get("/api/dashboard/recent-predictions")
    assert recent_response.status_code == 200
    recent = recent_response.json()
    assert len(recent) == 1
    assert recent[0]["subject"] == "Refund needed"
    assert recent[0]["body"] == "I was charged twice and need help today."
    assert recent[0]["predicted_queue"] == "billing_and_payments"
    assert recent[0]["probabilities"] == PROBABILITIES


def test_auto_route_at_threshold() -> None:
    assert routing_decision(0.80, 0.80) == "auto_route"


def test_human_review_below_threshold() -> None:
    assert routing_decision(0.799999, 0.80) == "human_review"


def test_default_route_threshold_is_080(monkeypatch) -> None:
    monkeypatch.delenv("OPSPILOT_ROUTING_THRESHOLD", raising=False)

    assert DEFAULT_ROUTE_THRESHOLD == 0.80
    assert route_threshold() == 0.80


def test_response_schema_rejects_unsupported_queue_and_incomplete_probabilities() -> None:
    with pytest.raises(ValidationError):
        TicketRouteResponse(
            predicted_queue="unsupported_queue",
            confidence=0.91,
            decision="auto_route",
            threshold=0.8,
            probabilities={"billing_and_payments": 1.0},
            model_id=MODEL_ID,
            latency_ms=1,
            timestamp=PREDICTION_TIMESTAMP,
        )


def test_route_response_has_exact_probability_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        ticket_intelligence,
        "predict_route",
        lambda subject, body: RoutePrediction(
            predicted_queue="technical_product_support",
            confidence=0.79,
            decision="human_review",
            threshold=0.8,
            probabilities=PROBABILITIES,
            model_id=MODEL_ID,
            latency_ms=7,
            timestamp=PREDICTION_TIMESTAMP,
        ),
    )

    response = client.post(
        "/api/ticket-intelligence/route",
        json={"subject": "Login issue", "body": "I cannot sign in."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["probabilities"]) == set(EXPECTED_QUEUE_LABELS)
    assert len(payload["probabilities"]) == 7
    assert "class_probabilities" not in payload
    assert payload["model_id"] == MODEL_ID
    assert payload["latency_ms"] == 7
    assert payload["timestamp"] == "2026-06-10T12:30:00Z"


def test_model_unavailable_returns_503(monkeypatch) -> None:
    def unavailable(subject: str, body: str) -> RoutePrediction:
        raise TicketIntelligenceUnavailableError("model could not be loaded")

    monkeypatch.setattr(ticket_intelligence, "predict_route", unavailable)

    response = client.post(
        "/api/ticket-intelligence/route",
        json={"subject": "Refund needed", "body": "Please help."},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Ticket Intelligence service unavailable: model could not be loaded"
    }

    recent_response = client.get("/api/dashboard/recent-predictions")
    assert recent_response.json() == []
