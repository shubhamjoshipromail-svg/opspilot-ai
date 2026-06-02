"""Train TF-IDF + logistic regression baselines for ticket intelligence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from data_utils import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    SYNTHETIC_DATA_PATH,
    canonicalize_ticket_frame,
    load_default_splits,
    load_smoke_split,
)


ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_DIR / "outputs"


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def _evaluate(name: str, model: Pipeline, x_eval: pd.Series, y_eval: pd.Series) -> dict:
    predictions = model.predict(x_eval)
    labels = sorted(y_eval.unique().tolist())
    report = classification_report(y_eval, predictions, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_eval, predictions)), 4),
        "macro_f1": round(float(f1_score(y_eval, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_eval, predictions, average="weighted")), 4),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_eval, predictions, labels=labels).tolist(),
        "predictions": predictions.tolist(),
        "target": name,
    }


def _write_confusion_matrix_png(metrics: dict, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = metrics["labels"]
    matrix = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Ticket Category Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, value, ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_data(
    train_path: Path | None = None,
    val_path: Path | None = None,
    test_path: Path | None = None,
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, dict]:
    if train_path and test_path:
        train_df = canonicalize_ticket_frame(pd.read_csv(train_path))
        val_df = canonicalize_ticket_frame(pd.read_csv(val_path)) if val_path and val_path.exists() else None
        test_df = canonicalize_ticket_frame(pd.read_csv(test_path))
        metadata = {
            "dataset_type": "explicit_splits",
            "train_path": str(train_path),
            "val_path": str(val_path) if val_path else None,
            "test_path": str(test_path),
        }
        return train_df, val_df, test_df, metadata

    if smoke_test:
        train_df, val_df, test_df = load_smoke_split()
        return train_df, val_df, test_df, {
            "dataset_type": "synthetic_smoke_test",
            "source_dataset": str(SYNTHETIC_DATA_PATH),
        }

    splits = load_default_splits()
    if splits is None:
        raise FileNotFoundError(
            "Missing real train/test splits. Run ml/ticket_intelligence/data_audit.py and "
            "ml/ticket_intelligence/create_splits.py after adding a real normalized ticket dataset. "
            f"Expected {DEFAULT_TRAIN_PATH}, {DEFAULT_VAL_PATH}, and {DEFAULT_TEST_PATH}. "
            "Use --smoke-test only for synthetic smoke-test training."
        )
    train_df, val_df, test_df = splits
    return train_df, val_df, test_df, {
        "dataset_type": "real_normalized_splits",
        "train_path": str(DEFAULT_TRAIN_PATH),
        "val_path": str(DEFAULT_VAL_PATH) if DEFAULT_VAL_PATH.exists() else None,
        "test_path": str(DEFAULT_TEST_PATH),
    }


def train(
    train_path: Path | None = None,
    val_path: Path | None = None,
    test_path: Path | None = None,
    smoke_test: bool = False,
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df, data_metadata = _load_data(train_path, val_path, test_path, smoke_test)

    category_model = _build_pipeline()
    category_model.fit(train_df["customer_message"], train_df["category"])
    category_metrics = _evaluate("category", category_model, test_df["customer_message"], test_df["category"])

    priority_model = None
    priority_metrics = None
    if train_df["priority"].nunique() >= 2:
        priority_model = _build_pipeline()
        priority_model.fit(train_df["customer_message"], train_df["priority"])
        priority_metrics = _evaluate("priority", priority_model, test_df["customer_message"], test_df["priority"])
    validation_metrics = None
    if val_df is not None and len(val_df) > 0:
        validation_metrics = {
            "category": _evaluate("category", category_model, val_df["customer_message"], val_df["category"]),
        }
        if priority_model is not None:
            validation_metrics["priority"] = _evaluate("priority", priority_model, val_df["customer_message"], val_df["priority"])
        validation_metrics["category"].pop("predictions", None)
        if "priority" in validation_metrics:
            validation_metrics["priority"].pop("predictions", None)

    with (OUTPUT_DIR / "baseline_category.pkl").open("wb") as handle:
        pickle.dump(category_model, handle)
    if priority_model is not None:
        with (OUTPUT_DIR / "baseline_priority.pkl").open("wb") as handle:
            pickle.dump(priority_model, handle)

    category_predictions = category_metrics.pop("predictions")
    priority_predictions = priority_metrics.pop("predictions") if priority_metrics else ["unknown"] * len(test_df)
    error_df = test_df[["ticket_id", "customer_message", "category", "priority"]].copy()
    error_df["predicted_category"] = category_predictions
    error_df["predicted_priority"] = priority_predictions
    error_df["category_correct"] = error_df["category"] == error_df["predicted_category"]
    error_df["priority_correct"] = error_df["priority"] == error_df["predicted_priority"]
    error_df = error_df[(~error_df["category_correct"]) | (~error_df["priority_correct"])]
    error_df.to_csv(OUTPUT_DIR / "error_analysis.csv", index=False)

    metrics = {
        "model_version": "ticket-intelligence-v1",
        "baseline": "tfidf_logistic_regression",
        "data": data_metadata,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)) if val_df is not None else 0,
        "test_rows": int(len(test_df)),
        "category": category_metrics,
        "priority": priority_metrics,
        "validation": validation_metrics,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_confusion_matrix_png(category_metrics, OUTPUT_DIR / "confusion_matrix.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=None, help="Optional explicit train.csv path.")
    parser.add_argument("--val", type=Path, default=None, help="Optional explicit val.csv path.")
    parser.add_argument("--test", type=Path, default=None, help="Optional explicit test.csv path.")
    parser.add_argument("--smoke-test", action="store_true", help="Use synthetic_tickets.csv only for smoke testing.")
    args = parser.parse_args()
    metrics = train(args.train, args.val, args.test, args.smoke_test)
    print(json.dumps({"metrics_path": str(OUTPUT_DIR / "metrics.json"), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
