from datetime import datetime, timezone

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routes import ticket_intelligence
from backend.app.services.ticket_intelligence import MODEL_ID, RoutePrediction


client = TestClient(app)
PROBABILITIES = {
    "billing_and_payments": 0.81,
    "customer_general": 0.03,
    "human_resources": 0.02,
    "returns_and_exchanges": 0.02,
    "sales_and_pre_sales": 0.02,
    "service_outages_and_maintenance": 0.02,
    "technical_product_support": 0.08,
}


def prediction(
    *,
    queue: str,
    confidence: float,
    decision: str,
    minute: int,
) -> RoutePrediction:
    probabilities = dict(PROBABILITIES)
    probabilities["billing_and_payments"] = 0.01
    probabilities[queue] = confidence
    return RoutePrediction(
        predicted_queue=queue,
        confidence=confidence,
        decision=decision,
        threshold=0.8,
        probabilities=probabilities,
        model_id=MODEL_ID,
        latency_ms=10 + minute,
        timestamp=datetime(2026, 6, 10, 12, minute, tzinfo=timezone.utc),
    )


def test_dashboard_summary_uses_persisted_tickets_and_predictions(monkeypatch) -> None:
    client.post(
        "/tickets",
        json={"customer_message": "First demo ticket", "source": "test"},
    )
    client.post(
        "/tickets",
        json={"customer_message": "Second demo ticket", "source": "test"},
    )

    predictions = iter(
        [
            prediction(
                queue="billing_and_payments",
                confidence=0.91,
                decision="auto_route",
                minute=1,
            ),
            prediction(
                queue="technical_product_support",
                confidence=0.69,
                decision="human_review",
                minute=2,
            ),
        ]
    )
    monkeypatch.setattr(
        ticket_intelligence,
        "predict_route",
        lambda subject, body: next(predictions),
    )

    client.post(
        "/api/ticket-intelligence/route",
        json={"subject": "Billing", "body": "Charged twice"},
    )
    client.post(
        "/api/ticket-intelligence/route",
        json={"subject": "Login", "body": "Cannot sign in"},
    )

    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_tickets"] == 2
    assert payload["total_predictions"] == 2
    assert payload["queue_distribution"]["billing_and_payments"] == 1
    assert payload["queue_distribution"]["technical_product_support"] == 1
    assert sum(payload["queue_distribution"].values()) == 2
    assert payload["decision_counts"] == {
        "auto_route": 1,
        "human_review": 1,
    }
    assert payload["average_confidence"] == 0.8


def test_recent_predictions_are_newest_first_and_limit_is_applied(monkeypatch) -> None:
    predictions = iter(
        [
            prediction(
                queue="billing_and_payments",
                confidence=0.91,
                decision="auto_route",
                minute=1,
            ),
            prediction(
                queue="technical_product_support",
                confidence=0.69,
                decision="human_review",
                minute=2,
            ),
        ]
    )
    monkeypatch.setattr(
        ticket_intelligence,
        "predict_route",
        lambda subject, body: next(predictions),
    )

    client.post(
        "/api/ticket-intelligence/route",
        json={"subject": "Older", "body": "First prediction"},
    )
    client.post(
        "/api/ticket-intelligence/route",
        json={"subject": "Newer", "body": "Second prediction"},
    )

    response = client.get("/api/dashboard/recent-predictions", params={"limit": 1})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["subject"] == "Newer"


def test_empty_dashboard_summary() -> None:
    response = client.get("/api/dashboard/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_tickets"] == 0
    assert payload["total_predictions"] == 0
    assert payload["average_confidence"] is None
    assert sum(payload["queue_distribution"].values()) == 0
    assert payload["decision_counts"] == {
        "auto_route": 0,
        "human_review": 0,
    }
