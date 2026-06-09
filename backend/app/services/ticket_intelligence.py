from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import inspect
import os
from typing import Any

MODEL_NAME = "shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1"
DEFAULT_ROUTE_THRESHOLD = 0.65


@dataclass(frozen=True)
class RoutePrediction:
    predicted_queue: str
    confidence: float
    decision: str
    threshold: float
    class_probabilities: dict[str, float]


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
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Ticket intelligence routing requires torch and transformers. "
            "Install them with `pip install torch transformers`."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, torch, device


def build_model_text(subject: str, body: str) -> str:
    return f"{subject.strip()}\n\n{body.strip()}".strip()


def predict_route(subject: str, body: str) -> RoutePrediction:
    tokenizer, model, torch, device = load_routing_model()
    model_text = build_model_text(subject, body)
    encoded = tokenizer(model_text, truncation=True, max_length=512, return_tensors="pt")

    accepts_token_type_ids = "token_type_ids" in inspect.signature(model.forward).parameters
    if not accepts_token_type_ids and "token_type_ids" in encoded:
        encoded.pop("token_type_ids")

    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()

    id2label = getattr(model.config, "id2label", {})
    class_probabilities = {
        str(id2label.get(idx, f"LABEL_{idx}")): round(float(probability), 6)
        for idx, probability in enumerate(probabilities.tolist())
    }
    predicted_index = int(probabilities.argmax().item())
    predicted_queue = str(id2label.get(predicted_index, f"LABEL_{predicted_index}"))
    confidence = round(float(probabilities[predicted_index].item()), 6)
    threshold = route_threshold()
    decision = "auto_route" if confidence >= threshold else "human_review"

    return RoutePrediction(
        predicted_queue=predicted_queue,
        confidence=confidence,
        decision=decision,
        threshold=threshold,
        class_probabilities=class_probabilities,
    )
