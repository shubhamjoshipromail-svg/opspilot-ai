from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 120.0


class OpsPilotAPIError(RuntimeError):
    """Base error raised by the OpsPilot frontend API client."""


class BackendOfflineError(OpsPilotAPIError):
    """Raised when the FastAPI backend cannot be reached."""


class ModelUnavailableError(OpsPilotAPIError):
    """Raised when Ticket Intelligence is unavailable or warming up."""


class ValidationAPIError(OpsPilotAPIError):
    """Raised when the backend rejects request input."""


class UnexpectedResponseError(OpsPilotAPIError):
    """Raised for unexpected HTTP responses or malformed JSON."""


class OpsPilotAPIClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OPSPILOT_API_URL", DEFAULT_API_URL)
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("OPSPILOT_API_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
        )
        self.session = session or requests.Session()

    def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout_seconds,
                **kwargs,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise BackendOfflineError(
                f"Could not reach the OpsPilot backend at {self.base_url}."
            ) from exc
        except requests.RequestException as exc:
            raise UnexpectedResponseError(f"Backend request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise UnexpectedResponseError(
                f"Backend returned a non-JSON response with status {response.status_code}."
            ) from exc

        detail = self._error_detail(payload)
        if response.status_code == 503:
            raise ModelUnavailableError(
                detail
                or "Ticket Intelligence is unavailable. The model may still be starting."
            )
        if response.status_code in {400, 422}:
            raise ValidationAPIError(detail or "The backend rejected the request.")
        if not response.ok:
            raise UnexpectedResponseError(
                detail or f"Backend returned HTTP {response.status_code}."
            )
        return payload

    @staticmethod
    def _error_detail(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            messages = [
                str(item.get("msg", item))
                if isinstance(item, dict)
                else str(item)
                for item in detail
            ]
            return "; ".join(messages)
        return None

    def health_check(self) -> dict[str, Any]:
        return self._expect_dict(self._request_json("GET", "/health"), "health check")

    def get_dashboard_summary(self) -> dict[str, Any]:
        return self._expect_dict(
            self._request_json("GET", "/api/dashboard/summary"),
            "dashboard summary",
        )

    def get_recent_predictions(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._expect_list(
            self._request_json(
                "GET",
                "/api/dashboard/recent-predictions",
                params={"limit": limit},
            ),
            "recent predictions",
        )

    def route_ticket(self, subject: str, body: str) -> dict[str, Any]:
        return self._expect_dict(
            self._request_json(
                "POST",
                "/api/ticket-intelligence/route",
                json={"subject": subject, "body": body},
            ),
            "ticket routing",
        )

    def list_tickets(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._expect_list(
            self._request_json(
                "GET",
                "/tickets",
                params={"skip": skip, "limit": limit},
            ),
            "ticket list",
        )

    @staticmethod
    def _expect_dict(payload: Any, context: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise UnexpectedResponseError(
                f"Backend returned an unexpected {context} response."
            )
        return payload

    @staticmethod
    def _expect_list(payload: Any, context: str) -> list[dict[str, Any]]:
        if not isinstance(payload, list) or any(
            not isinstance(item, dict) for item in payload
        ):
            raise UnexpectedResponseError(
                f"Backend returned an unexpected {context} response."
            )
        return payload
