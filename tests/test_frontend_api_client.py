from typing import Any

import pytest
import requests

from frontend.api_client import (
    BackendOfflineError,
    ModelUnavailableError,
    OpsPilotAPIClient,
    UnexpectedResponseError,
    ValidationAPIError,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        json_error: bool = False,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.json_error = json_error
        self.ok = 200 <= status_code < 300

    def json(self) -> Any:
        if self.json_error:
            raise ValueError("not json")
        return self.payload


class FakeSession:
    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response


def test_route_ticket_uses_canonical_endpoint() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {
                "predicted_queue": "billing_and_payments",
                "confidence": 0.91,
            },
        )
    )
    client = OpsPilotAPIClient(
        base_url="http://api.example",
        timeout_seconds=10,
        session=session,  # type: ignore[arg-type]
    )

    payload = client.route_ticket("Invoice issue", "I was charged twice.")

    assert payload["predicted_queue"] == "billing_and_payments"
    assert session.calls == [
        {
            "method": "POST",
            "url": "http://api.example/api/ticket-intelligence/route",
            "timeout": 10,
            "json": {
                "subject": "Invoice issue",
                "body": "I was charged twice.",
            },
        }
    ]


def test_backend_connection_error_is_structured() -> None:
    client = OpsPilotAPIClient(
        session=FakeSession(error=requests.ConnectionError("offline")),  # type: ignore[arg-type]
    )

    with pytest.raises(BackendOfflineError):
        client.health_check()


def test_model_unavailable_error_preserves_detail() -> None:
    client = OpsPilotAPIClient(
        session=FakeSession(
            FakeResponse(503, {"detail": "model is warming up"})
        ),  # type: ignore[arg-type]
    )

    with pytest.raises(ModelUnavailableError, match="warming up"):
        client.route_ticket("Subject", "Body")


def test_validation_error_is_structured() -> None:
    client = OpsPilotAPIClient(
        session=FakeSession(
            FakeResponse(422, {"detail": [{"msg": "Field required"}]})
        ),  # type: ignore[arg-type]
    )

    with pytest.raises(ValidationAPIError, match="Field required"):
        client.route_ticket("", "")


def test_non_json_response_is_unexpected() -> None:
    client = OpsPilotAPIClient(
        session=FakeSession(FakeResponse(500, None, json_error=True)),  # type: ignore[arg-type]
    )

    with pytest.raises(UnexpectedResponseError, match="non-JSON"):
        client.get_dashboard_summary()


def test_success_with_wrong_response_shape_is_unexpected() -> None:
    client = OpsPilotAPIClient(
        session=FakeSession(FakeResponse(200, ["not", "a", "summary"])),  # type: ignore[arg-type]
    )

    with pytest.raises(UnexpectedResponseError, match="dashboard summary"):
        client.get_dashboard_summary()
