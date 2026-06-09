from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routes import ticket_intelligence
from backend.app.services.ticket_intelligence import RoutePrediction


client = TestClient(app)


def test_ticket_intelligence_route(monkeypatch) -> None:
    def fake_predict_route(subject: str, body: str) -> RoutePrediction:
        assert subject == "Refund needed"
        assert body == "I was charged twice and need help today."
        return RoutePrediction(
            predicted_queue="billing_and_payments",
            confidence=0.91,
            decision="auto_route",
            threshold=0.65,
            class_probabilities={
                "billing_and_payments": 0.91,
                "technical_product_support": 0.09,
            },
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
        "threshold": 0.65,
        "class_probabilities": {
            "billing_and_payments": 0.91,
            "technical_product_support": 0.09,
        },
    }
