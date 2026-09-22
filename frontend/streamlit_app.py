from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import pandas as pd
import streamlit as st

from api_client import (
    BackendOfflineError,
    ModelUnavailableError,
    OpsPilotAPIClient,
    OpsPilotAPIError,
    UnexpectedResponseError,
    ValidationAPIError,
)


st.set_page_config(
    page_title="OpsPilot",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)

QUEUE_LABELS = {
    "billing_and_payments": "Billing & Payments",
    "customer_general": "Customer General",
    "human_resources": "Human Resources",
    "returns_and_exchanges": "Returns & Exchanges",
    "sales_and_pre_sales": "Sales & Pre-Sales",
    "service_outages_and_maintenance": "Service Outages & Maintenance",
    "technical_product_support": "Technical Product Support",
}


def get_api_client() -> OpsPilotAPIClient:
    return OpsPilotAPIClient()


def humanize(value: str | None) -> str:
    if not value:
        return "Unknown"
    return QUEUE_LABELS.get(value, value.replace("_", " ").title())


def format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def format_timestamp(value: str | None) -> str:
    if not value:
        return "N/A"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def preview(value: str | None, length: int = 90) -> str:
    text = (value or "").strip().replace("\n", " ")
    return text if len(text) <= length else f"{text[: length - 1]}..."


def load_resource(
    loader: Callable[[], Any],
    *,
    unavailable_message: str,
) -> tuple[Any | None, OpsPilotAPIError | None]:
    try:
        return loader(), None
    except OpsPilotAPIError as exc:
        return None, exc
    except Exception as exc:
        return None, UnexpectedResponseError(
            f"{unavailable_message}: {exc}"
        )


def render_api_error(error: OpsPilotAPIError, *, context: str) -> None:
    if isinstance(error, BackendOfflineError):
        st.error(
            f"{context} is unavailable because the FastAPI backend is offline. "
            "Start the backend and refresh this page."
        )
    elif isinstance(error, ModelUnavailableError):
        st.warning(
            "Ticket Intelligence is currently unavailable. The cloud model may "
            "be cold-starting; wait a moment and try again."
        )
        with st.expander("Service detail"):
            st.write(str(error))
    elif isinstance(error, ValidationAPIError):
        st.warning(f"The request could not be validated: {error}")
    else:
        st.error(f"{context} could not be loaded: {error}")


def render_queue_chart(queue_distribution: dict[str, int]) -> None:
    rows = [
        {"Queue": humanize(queue), "Tickets": count}
        for queue, count in queue_distribution.items()
    ]
    frame = pd.DataFrame(rows).set_index("Queue")
    st.bar_chart(frame, horizontal=True, color="#4F7CAC")


def render_decision_chart(decision_counts: dict[str, int]) -> None:
    frame = pd.DataFrame(
        [
            {"Decision": "Auto-route", "Predictions": decision_counts.get("auto_route", 0)},
            {
                "Decision": "Human review",
                "Predictions": decision_counts.get("human_review", 0),
            },
        ]
    ).set_index("Decision")
    st.bar_chart(frame, color="#68A691")


def recent_predictions_frame(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Timestamp": format_timestamp(item.get("created_at") or item.get("timestamp")),
                "Queue": humanize(item.get("predicted_queue")),
                "Confidence": item.get("confidence"),
                "Decision": humanize(item.get("decision")),
                "Threshold": item.get("threshold"),
                "Latency (ms)": item.get("latency_ms"),
                "Subject": preview(item.get("subject")),
            }
            for item in predictions
        ]
    )


def render_recent_table(predictions: list[dict[str, Any]]) -> None:
    if not predictions:
        st.info("No routing predictions have been stored yet.")
        return
    st.dataframe(
        recent_predictions_frame(predictions),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                format="%.1%%",
            ),
            "Threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
        },
    )


def render_dashboard(
    summary: dict[str, Any] | None,
    summary_error: OpsPilotAPIError | None,
    recent: list[dict[str, Any]] | None,
    recent_error: OpsPilotAPIError | None,
) -> None:
    st.subheader("Operations Dashboard")
    st.caption("Live metrics derived from canonical database records.")
    if summary_error:
        render_api_error(summary_error, context="Dashboard")
        return
    if summary is None:
        st.info("Dashboard data is not available.")
        return

    decisions = summary.get("decision_counts", {})
    auto_route = int(decisions.get("auto_route", 0))
    human_review = int(decisions.get("human_review", 0))
    total_predictions = int(summary.get("total_predictions", 0))
    review_rate = human_review / total_predictions if total_predictions else None

    columns = st.columns(6)
    columns[0].metric("Total tickets", summary.get("total_tickets", 0))
    columns[1].metric("Predictions", total_predictions)
    columns[2].metric("Auto-routed", auto_route)
    columns[3].metric("Human review", human_review)
    columns[4].metric(
        "Average confidence",
        format_percent(summary.get("average_confidence")),
    )
    columns[5].metric("Review rate", format_percent(review_rate))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Queue distribution")
        render_queue_chart(summary.get("queue_distribution", {}))
    with right:
        st.markdown("#### Routing decisions")
        render_decision_chart(decisions)

    st.markdown("#### Recent predictions")
    if recent_error:
        render_api_error(recent_error, context="Recent predictions")
    else:
        render_recent_table(recent or [])


def render_prediction_result(result: dict[str, Any]) -> None:
    decision = result.get("decision")
    if decision == "auto_route":
        st.success(
            f"Auto-route to {humanize(result.get('predicted_queue'))}. "
            "The prediction meets the configured confidence threshold."
        )
    else:
        st.warning(
            f"Send to human review. The predicted queue is "
            f"{humanize(result.get('predicted_queue'))}, but confidence is below "
            "the auto-route threshold."
        )

    columns = st.columns(4)
    columns[0].metric("Predicted queue", humanize(result.get("predicted_queue")))
    columns[1].metric("Confidence", format_percent(result.get("confidence")))
    columns[2].metric("Threshold", format_percent(result.get("threshold")))
    columns[3].metric("Latency", f"{result.get('latency_ms', 0)} ms")

    metadata_left, metadata_right = st.columns(2)
    metadata_left.caption(f"Model: `{result.get('model_id', 'Unknown')}`")
    metadata_right.caption(
        f"Prediction time: {format_timestamp(result.get('timestamp'))}"
    )

    probabilities = result.get("probabilities", {})
    probability_frame = pd.DataFrame(
        [
            {"Queue": humanize(queue), "Probability": probability}
            for queue, probability in sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
    ).set_index("Queue")
    st.markdown("#### Probability breakdown")
    st.bar_chart(probability_frame, horizontal=True, color="#4F7CAC")


def render_routing_demo(client: OpsPilotAPIClient) -> None:
    st.subheader("Ticket Routing Demo")
    st.caption(
        "Submit a service ticket to the canonical FastAPI routing endpoint. "
        "Successful predictions are stored automatically."
    )
    with st.form("ticket-routing-form"):
        subject = st.text_input(
            "Subject",
            placeholder="Invoice payment issue",
        )
        body = st.text_area(
            "Body",
            placeholder=(
                "I was charged twice for my subscription and need help fixing "
                "the billing problem."
            ),
            height=180,
        )
        submitted = st.form_submit_button(
            "Route ticket",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not subject.strip() or not body.strip():
            st.session_state["routing_error"] = {
                "kind": "validation",
                "message": "Enter both a subject and ticket body before routing.",
            }
            st.rerun()
        else:
            with st.spinner(
                "Running Ticket Intelligence. A cloud model cold start may take a moment..."
            ):
                try:
                    st.session_state["latest_prediction"] = client.route_ticket(
                        subject.strip(),
                        body.strip(),
                    )
                    st.session_state.pop("routing_error", None)
                    st.rerun()
                except OpsPilotAPIError as exc:
                    st.session_state["routing_error"] = {
                        "kind": exc.__class__.__name__,
                        "message": str(exc),
                    }
                    st.rerun()

    latest_prediction = st.session_state.get("latest_prediction")
    if latest_prediction:
        st.divider()
        render_prediction_result(latest_prediction)


def render_global_routing_status() -> None:
    routing_error = st.session_state.get("routing_error")
    latest_prediction = st.session_state.get("latest_prediction")
    if routing_error:
        if routing_error.get("kind") == "ModelUnavailableError":
            st.warning(
                "Ticket Intelligence is currently unavailable. The cloud model "
                "may be cold-starting; wait a moment and try again."
            )
        elif routing_error.get("kind") == "validation":
            st.warning(routing_error.get("message"))
        else:
            st.error(f"Ticket routing failed: {routing_error.get('message')}")
        return
    if latest_prediction:
        decision = latest_prediction.get("decision")
        message = (
            f"Latest routing result: {humanize(latest_prediction.get('predicted_queue'))} "
            f"at {format_percent(latest_prediction.get('confidence'))} confidence."
        )
        if decision == "auto_route":
            st.success(message)
        else:
            st.warning(f"{message} Human review is required.")


def render_recent_predictions(
    recent: list[dict[str, Any]] | None,
    error: OpsPilotAPIError | None,
) -> None:
    st.subheader("Recent Predictions")
    st.caption("The 20 newest persisted Ticket Intelligence routing records.")
    if error:
        render_api_error(error, context="Recent predictions")
        return
    render_recent_table(recent or [])


def render_ticket_inbox(
    tickets: list[dict[str, Any]] | None,
    error: OpsPilotAPIError | None,
) -> None:
    st.subheader("Ticket Inbox")
    st.caption("Canonical ticket records stored by the FastAPI ticket CRUD service.")
    if error:
        render_api_error(error, context="Ticket inbox")
        return
    if not tickets:
        st.info(
            "No tickets are stored yet. Prediction demo submissions are logged "
            "separately and do not currently create ticket records."
        )
        return

    frame = pd.DataFrame(
        [
            {
                "ID": ticket.get("id"),
                "Status": humanize(ticket.get("status")),
                "Category": humanize(ticket.get("true_category")),
                "Priority": humanize(ticket.get("true_priority")),
                "Source / Channel": " / ".join(
                    value
                    for value in [ticket.get("source"), ticket.get("channel")]
                    if value
                )
                or "Unknown",
                "Message": preview(ticket.get("customer_message"), 110),
                "Created": format_timestamp(ticket.get("created_at")),
                "Updated": format_timestamp(ticket.get("updated_at")),
            }
            for ticket in tickets
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_model_analytics(
    summary: dict[str, Any] | None,
    summary_error: OpsPilotAPIError | None,
    recent: list[dict[str, Any]] | None,
    recent_error: OpsPilotAPIError | None,
) -> None:
    st.subheader("Model Analytics")
    st.caption(
        "Current demo analytics. Confidence and latency distributions use only "
        "the bounded recent-predictions API, not the full prediction history."
    )
    if summary_error:
        render_api_error(summary_error, context="Model analytics")
        return

    predictions = recent or []
    if recent_error:
        render_api_error(recent_error, context="Recent model data")
    summary = summary or {}
    decisions = summary.get("decision_counts", {})
    total_predictions = int(summary.get("total_predictions", 0))
    auto_route = int(decisions.get("auto_route", 0))
    human_review = int(decisions.get("human_review", 0))

    metrics = st.columns(4)
    metrics[0].metric(
        "Auto-route rate",
        format_percent(auto_route / total_predictions if total_predictions else None),
    )
    metrics[1].metric(
        "Human-review rate",
        format_percent(human_review / total_predictions if total_predictions else None),
    )
    metrics[2].metric(
        "Average confidence",
        format_percent(summary.get("average_confidence")),
    )
    latencies = [
        float(item["latency_ms"])
        for item in predictions
        if item.get("latency_ms") is not None
    ]
    metrics[3].metric(
        "Recent avg latency",
        f"{sum(latencies) / len(latencies):.0f} ms" if latencies else "N/A",
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Queue mix")
        render_queue_chart(summary.get("queue_distribution", {}))
    with right:
        st.markdown("#### Decision mix")
        render_decision_chart(decisions)

    if len(predictions) >= 2:
        confidence_frame = pd.DataFrame(
            {
                "Prediction": list(range(1, len(predictions) + 1)),
                "Confidence": [
                    item.get("confidence", 0.0)
                    for item in reversed(predictions)
                ],
            }
        ).set_index("Prediction")
        st.markdown("#### Recent confidence distribution")
        st.bar_chart(confidence_frame, color="#4F7CAC")
    else:
        st.info(
            "At least two recent prediction records are needed for a confidence chart."
        )

    if latencies:
        latency_columns = st.columns(3)
        latency_columns[0].metric("Minimum latency", f"{min(latencies):.0f} ms")
        latency_columns[1].metric(
            "Average latency",
            f"{sum(latencies) / len(latencies):.0f} ms",
        )
        latency_columns[2].metric("Maximum latency", f"{max(latencies):.0f} ms")


def render_sidebar(client: OpsPilotAPIClient) -> None:
    with st.sidebar:
        st.markdown("### Runtime")
        st.caption(f"API: `{client.base_url}`")
        health, error = load_resource(
            client.health_check,
            unavailable_message="Health check failed",
        )
        if error:
            st.error("Backend offline")
        elif health and health.get("status") == "ok":
            st.success("Backend connected")
        else:
            st.warning("Backend status unknown")
        st.divider()
        st.markdown("### Product boundary")
        st.caption(
            "This console calls the canonical FastAPI backend. It does not load "
            "models or research artifacts in Streamlit."
        )


def main() -> None:
    client = get_api_client()
    st.title("OpsPilot")
    st.subheader("AI operations console for ticket routing, review, and analytics")
    st.write(
        "OpsPilot is a modular AI operations platform that connects operational "
        "data, specialized models, controlled workflows, and human review. It is "
        "an operations backbone, not a simple chatbot or standalone classifier."
    )
    render_sidebar(client)

    summary, summary_error = load_resource(
        client.get_dashboard_summary,
        unavailable_message="Dashboard summary failed",
    )
    recent, recent_error = load_resource(
        lambda: client.get_recent_predictions(limit=20),
        unavailable_message="Recent predictions failed",
    )
    tickets, tickets_error = load_resource(
        lambda: client.list_tickets(limit=100),
        unavailable_message="Ticket inbox failed",
    )
    render_global_routing_status()

    dashboard_tab, routing_tab, recent_tab, inbox_tab, analytics_tab = st.tabs(
        [
            "Dashboard",
            "Ticket Routing Demo",
            "Recent Predictions",
            "Ticket Inbox",
            "Model Analytics",
        ]
    )
    with dashboard_tab:
        render_dashboard(summary, summary_error, recent, recent_error)
    with routing_tab:
        render_routing_demo(client)
    with recent_tab:
        render_recent_predictions(recent, recent_error)
    with inbox_tab:
        render_ticket_inbox(tickets, tickets_error)
    with analytics_tab:
        render_model_analytics(summary, summary_error, recent, recent_error)


if __name__ == "__main__":
    main()
