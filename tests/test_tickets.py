from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_create_and_list_tickets() -> None:
    create_response = client.post(
        "/tickets",
        json={
            "external_id": "TICKET-001",
            "customer_message": "My order never arrived and I want a refund.",
            "channel": "email",
            "source": "sample",
            "true_category": "shipping_delay",
            "true_priority": "high",
        },
    )

    assert create_response.status_code == 201
    created_ticket = create_response.json()
    assert created_ticket["id"] == 1
    assert created_ticket["status"] == "new"
    assert created_ticket["external_id"] == "TICKET-001"

    list_response = client.get("/tickets")

    assert list_response.status_code == 200
    tickets = list_response.json()
    assert len(tickets) == 1
    assert tickets[0]["customer_message"] == "My order never arrived and I want a refund."
