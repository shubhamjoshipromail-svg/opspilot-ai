from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database import Base, get_db
from backend.app.main import app


engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db() -> Generator[Session, None, None]:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def setup_function() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


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
