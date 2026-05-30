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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
DATA_PATH = MODULE_DIR / "data" / "synthetic_tickets.csv"
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


def _evaluate(name: str, model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    labels = sorted(y_test.unique().tolist())
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
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


def train(data_path: Path = DATA_PATH) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["category"],
    )

    category_model = _build_pipeline()
    category_model.fit(train_df["customer_message"], train_df["category"])
    category_metrics = _evaluate("category", category_model, test_df["customer_message"], test_df["category"])

    priority_model = _build_pipeline()
    priority_model.fit(train_df["customer_message"], train_df["priority"])
    priority_metrics = _evaluate("priority", priority_model, test_df["customer_message"], test_df["priority"])

    with (OUTPUT_DIR / "baseline_category.pkl").open("wb") as handle:
        pickle.dump(category_model, handle)
    with (OUTPUT_DIR / "baseline_priority.pkl").open("wb") as handle:
        pickle.dump(priority_model, handle)

    category_predictions = category_metrics.pop("predictions")
    priority_predictions = priority_metrics.pop("predictions")
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
        "dataset": str(data_path.relative_to(ROOT)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "category": category_metrics,
        "priority": priority_metrics,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_confusion_matrix_png(category_metrics, OUTPUT_DIR / "confusion_matrix.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    metrics = train(args.data)
    print(json.dumps({"metrics_path": str(OUTPUT_DIR / "metrics.json"), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
