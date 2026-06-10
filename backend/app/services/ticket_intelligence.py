from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import inspect
import os
from time import perf_counter
from typing import Any

from backend.app.schemas.ticket_intelligence import (
    EXPECTED_QUEUE_LABELS,
    QueueLabel,
    RoutingDecision,
)


MODEL_ID = "shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1"
DEFAULT_ROUTE_THRESHOLD = 0.80


class TicketIntelligenceUnavailableError(RuntimeError):
    """Raised when the routing model cannot be loaded or used for inference."""


@dataclass(frozen=True)
class RoutePrediction:
    predicted_queue: QueueLabel
    confidence: float
    decision: RoutingDecision
    threshold: float
    probabilities: dict[QueueLabel, float]
    model_id: str
    latency_ms: int
    timestamp: datetime


def route_threshold() -> float:
    raw_threshold = os.getenv("OPSPILOT_ROUTING_THRESHOLD")
    if raw_threshold is None:
        return DEFAULT_ROUTE_THRESHOLD
    try:
        threshold = float(raw_threshold)
    except ValueError:
        return DEFAULT_ROUTE_THRESHOLD
    return min(max(threshold, 0.0), 1.0)


@lru_cache(maxsize=1)
def load_routing_model() -> tuple[Any, Any, Any, str]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise TicketIntelligenceUnavailableError(
            "Ticket intelligence routing requires torch and transformers. "
            "Install them with `pip install torch transformers`."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except ValueError as exc:
        if "Tokenizer class" not in str(exc):
            raise TicketIntelligenceUnavailableError(
                f"Could not load routing tokenizer from {MODEL_ID}: {exc}"
            ) from exc
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)
        except Exception as fallback_exc:  # pragma: no cover - depends on remote model state
            raise TicketIntelligenceUnavailableError(
                f"Could not load routing tokenizer from {MODEL_ID}: {fallback_exc}"
            ) from fallback_exc
    except Exception as exc:  # pragma: no cover - depends on remote model state
        raise TicketIntelligenceUnavailableError(
            f"Could not load routing tokenizer from {MODEL_ID}: {exc}"
        ) from exc
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    except Exception as exc:  # pragma: no cover - depends on remote model state
        raise TicketIntelligenceUnavailableError(
            f"Could not load routing model from {MODEL_ID}: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return tokenizer, model, torch, device


def build_model_text(subject: str, body: str) -> str:
    return f"{subject.strip()}\n\n{body.strip()}".strip()


def routing_decision(confidence: float, threshold: float) -> RoutingDecision:
    return "auto_route" if confidence >= threshold else "human_review"


def _label_for_index(id2label: dict[Any, Any], index: int) -> str:
    return str(id2label.get(index, id2label.get(str(index), f"LABEL_{index}")))


def predict_route(subject: str, body: str) -> RoutePrediction:
    started_at = perf_counter()
    try:
        tokenizer, model, torch, device = load_routing_model()
        model_text = build_model_text(subject, body)
        encoded = tokenizer(model_text, truncation=True, max_length=512, return_tensors="pt")

        accepts_token_type_ids = "token_type_ids" in inspect.signature(model.forward).parameters
        if not accepts_token_type_ids and "token_type_ids" in encoded:
            encoded.pop("token_type_ids")

        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            model_probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()

        id2label = getattr(model.config, "id2label", {})
        probabilities_by_label = {
            _label_for_index(id2label, index): float(probability)
            for index, probability in enumerate(model_probabilities.tolist())
        }
        if set(probabilities_by_label) != set(EXPECTED_QUEUE_LABELS):
            raise TicketIntelligenceUnavailableError(
                "Routing model labels do not match the seven supported queue labels."
            )

        probabilities = {
            label: round(probabilities_by_label[label], 6)
            for label in EXPECTED_QUEUE_LABELS
        }
        predicted_index = int(model_probabilities.argmax().item())
        predicted_queue = _label_for_index(id2label, predicted_index)
        confidence = round(float(model_probabilities[predicted_index].item()), 6)
        threshold = route_threshold()

        return RoutePrediction(
            predicted_queue=predicted_queue,
            confidence=confidence,
            decision=routing_decision(confidence, threshold),
            threshold=threshold,
            probabilities=probabilities,
            model_id=MODEL_ID,
            latency_ms=max(0, round((perf_counter() - started_at) * 1000)),
            timestamp=datetime.now(timezone.utc),
        )
    except TicketIntelligenceUnavailableError:
        raise
    except Exception as exc:  # pragma: no cover - exact failure depends on model runtime
        raise TicketIntelligenceUnavailableError(
            f"Ticket routing model is unavailable: {exc}"
        ) from exc
