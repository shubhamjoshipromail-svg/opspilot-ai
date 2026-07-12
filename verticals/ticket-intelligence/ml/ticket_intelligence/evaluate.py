"""Evaluate saved Ticket Intelligence baseline models on the fixed test split."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data_utils import ARTIFACT_DIR, DEFAULT_TEST_PATH, MODEL_VERSION, OUTPUT_DIR, canonicalize_ticket_frame


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}. Run train_baseline.py first.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_with_confidence(model, texts: pd.Series) -> tuple[list[str], list[float]]:
    predictions = [str(value) for value in model.predict(texts)]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        confidences = [round(float(row.max()), 4) for row in probabilities]
    else:
        confidences = [0.5] * len(predictions)
    return predictions, confidences


def evaluate_target(target: str, model, test_df: pd.DataFrame, label_column: str) -> tuple[dict, list[str], list[float]]:
    y_true = test_df[label_column]
    predictions, confidences = predict_with_confidence(model, test_df["customer_message"])
    labels = sorted(y_true.dropna().unique().tolist())
    report = classification_report(y_true, predictions, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "macro_f1": round(float(f1_score(y_true, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, predictions, average="weighted")), 4),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
    }
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / f"{target}_classification_report.csv")
    write_confusion_matrix_png(metrics, OUTPUT_DIR / f"{target}_confusion_matrix.png", f"{target.title()} Confusion Matrix")
    return metrics, predictions, confidences


def write_confusion_matrix_png(metrics: dict, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    labels = metrics["labels"]
    matrix = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if value:
                ax.text(col_idx, row_idx, value, ha="center", va="center", color="black", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_error_frame(
    test_df: pd.DataFrame,
    category_predictions: list[str] | None,
    category_confidences: list[float] | None,
    priority_predictions: list[str] | None,
    priority_confidences: list[float] | None,
) -> pd.DataFrame:
    errors = test_df.copy()
    if category_predictions is not None:
        errors["predicted_category"] = category_predictions
        errors["category_confidence"] = category_confidences
        errors["category_correct"] = errors["category"] == errors["predicted_category"]
    if priority_predictions is not None:
        errors["predicted_priority"] = priority_predictions
        errors["priority_confidence"] = priority_confidences
        errors["priority_correct"] = errors["priority"] == errors["predicted_priority"]

    masks = []
    if "category_correct" in errors:
        masks.append(~errors["category_correct"])
    if "priority_correct" in errors:
        masks.append(~errors["priority_correct"])
    if not masks:
        return errors.head(0)
    mask = masks[0]
    for extra in masks[1:]:
        mask = mask | extra
    return errors[mask].copy()


def evaluate() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_df = canonicalize_ticket_frame(pd.read_csv(DEFAULT_TEST_PATH))
    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    metrics: dict = {
        "model_version": metadata.get("model_version", MODEL_VERSION),
        "model_type": metadata.get("model_type", "tfidf_logistic_regression_baseline"),
        "data": metadata.get("data", {}),
        "test_rows": int(len(test_df)),
    }

    category_predictions = category_confidences = None
    category_model_path = ARTIFACT_DIR / "category_model.pkl"
    if category_model_path.exists():
        category_model = load_model(category_model_path)
        category_metrics, category_predictions, category_confidences = evaluate_target(
            "category", category_model, test_df, "category"
        )
        metrics["category"] = category_metrics

    priority_predictions = priority_confidences = None
    priority_model_path = ARTIFACT_DIR / "priority_model.pkl"
    if priority_model_path.exists():
        priority_model = load_model(priority_model_path)
        priority_metrics, priority_predictions, priority_confidences = evaluate_target(
            "priority", priority_model, test_df, "priority"
        )
        metrics["priority"] = priority_metrics

    error_df = build_error_frame(test_df, category_predictions, category_confidences, priority_predictions, priority_confidences)
    error_df.to_csv(OUTPUT_DIR / "error_analysis.csv", index=False)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = evaluate()
    summary = {
        "metrics_path": str(OUTPUT_DIR / "metrics.json"),
        "category_macro_f1": metrics.get("category", {}).get("macro_f1"),
        "priority_macro_f1": metrics.get("priority", {}).get("macro_f1"),
        "outputs_dir": str(OUTPUT_DIR),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
