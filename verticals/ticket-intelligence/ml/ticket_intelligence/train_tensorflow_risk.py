"""Train an optional TensorFlow model for weak policy-derived risk labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from comparison_utils import write_model_comparison
from create_risk_labels import add_risk_labels, create_risk_label_outputs
from data_utils import ARTIFACT_DIR, DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, DEFAULT_VAL_PATH, OUTPUT_DIR, RANDOM_STATE


INPUT_FEATURE = "model_text"
LABEL_COLUMN = "high_risk_policy_label"
MAX_TOKENS = 30_000
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 128
BATCH_SIZE = 64
DEFAULT_EPOCHS = 10


def import_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for this script. Install it with "
            "`pip install tensorflow` or `pip install -r verticals/ticket-intelligence/requirements.txt`."
        ) from exc
    return tf


def load_labeled_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}. Run create_splits.py first.")
    df = pd.read_csv(path)
    if INPUT_FEATURE not in df.columns:
        raise ValueError(
            f"{path} must include {INPUT_FEATURE}. Rebuild the dataset with scripts/build_ticket_dataset.py "
            "and recreate splits before TensorFlow training."
        )
    df = add_risk_labels(df)
    df[INPUT_FEATURE] = df[INPUT_FEATURE].fillna("").astype(str).str.strip()
    df = df[df[INPUT_FEATURE].str.len() > 0].copy()
    return df


def build_model(tf):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        name="text_vectorization",
    )
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(), dtype=tf.string, name="model_text"),
            vectorizer,
            tf.keras.layers.Embedding(input_dim=MAX_TOKENS, output_dim=EMBEDDING_DIM, name="embedding"),
            tf.keras.layers.Conv1D(128, 5, activation="relu", name="conv1d"),
            tf.keras.layers.GlobalMaxPooling1D(name="global_max_pooling"),
            tf.keras.layers.Dropout(0.3, name="dropout_1"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_relu"),
            tf.keras.layers.Dropout(0.3, name="dropout_2"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="risk_sigmoid"),
        ],
        name="opspilot_tensorflow_weak_risk_classifier",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model, vectorizer


def write_confusion_matrix_png(labels: list[str], matrix: list[list[int]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True weak label",
        xlabel="Predicted weak label",
        title="TensorFlow Weak Risk Confusion Matrix",
    )
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if value:
                ax.text(col_idx, row_idx, value, ha="center", va="center", color="black", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def train_tensorflow_risk(epochs: int = DEFAULT_EPOCHS, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    print("Creating weak risk label audit...", flush=True)
    create_risk_label_outputs()

    print("Importing TensorFlow...", flush=True)
    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading fixed train/validation/test splits with weak labels...", flush=True)
    train_df = load_labeled_split(DEFAULT_TRAIN_PATH)
    val_df = load_labeled_split(DEFAULT_VAL_PATH)
    test_df = load_labeled_split(DEFAULT_TEST_PATH)

    train_texts = train_df[INPUT_FEATURE].astype(str).to_numpy()
    val_texts = val_df[INPUT_FEATURE].astype(str).to_numpy()
    test_texts = test_df[INPUT_FEATURE].astype(str).to_numpy()
    y_train = train_df[LABEL_COLUMN].astype("int32").to_numpy()
    y_val = val_df[LABEL_COLUMN].astype("int32").to_numpy()
    y_test = test_df[LABEL_COLUMN].astype("int32").to_numpy()

    print("Building weak-risk model...", flush=True)
    model, vectorizer = build_model(tf)
    print("Adapting TextVectorization on training text only...", flush=True)
    vectorizer.adapt(train_texts)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]
    print("Training TensorFlow weak-risk model...", flush=True)
    history = model.fit(
        train_texts,
        y_train,
        validation_data=(val_texts, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    print("Evaluating once on the held-out test split...", flush=True)
    probabilities = model.predict(test_texts, batch_size=batch_size, verbose=0).reshape(-1)
    predicted_ids = (probabilities >= 0.5).astype("int32")
    labels = ["not_high_risk_policy_label", "high_risk_policy_label"]
    report = classification_report(y_test, predicted_ids, labels=[0, 1], target_names=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predicted_ids, labels=[0, 1]).tolist()
    metrics = {
        "model_version": "ticket-intelligence-tensorflow-risk-v1",
        "model_type": "tensorflow_keras_binary_text_classifier",
        "label_type": "weak_policy_derived",
        "not_ground_truth": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "train_path": str(DEFAULT_TRAIN_PATH),
            "val_path": str(DEFAULT_VAL_PATH),
            "test_path": str(DEFAULT_TEST_PATH),
            "input_feature": INPUT_FEATURE,
            "input_feature_description": "subject + body + type + tags; answer/reference_answer is excluded to prevent leakage",
            "label": LABEL_COLUMN,
            "label_quality_note": "This predicts a policy-derived weak label, not observed customer escalation.",
        },
        "row_counts": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "preprocessing": {
            "text_vectorization_adapted_on": "train.model_text only",
            "max_tokens": MAX_TOKENS,
            "output_sequence_length": SEQUENCE_LENGTH,
        },
        "architecture": [
            "TextVectorization",
            f"Embedding(input_dim={MAX_TOKENS}, output_dim={EMBEDDING_DIM})",
            "Conv1D(128, kernel_size=5, activation='relu')",
            "GlobalMaxPooling1D",
            "Dropout(0.3)",
            "Dense(64, activation='relu')",
            "Dropout(0.3)",
            "Dense(1, activation='sigmoid')",
        ],
        "training": {
            "optimizer": "Adam",
            "loss": "binary_crossentropy",
            "metric": "accuracy",
            "batch_size": batch_size,
            "max_epochs": epochs,
            "epochs_completed": len(history.history.get("loss", [])),
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 2,
                "restore_best_weights": True,
            },
            "history": {key: [round(float(value), 6) for value in values] for key, values in history.history.items()},
        },
        "accuracy": round(float(accuracy_score(y_test, predicted_ids)), 4),
        "macro_f1": round(float(f1_score(y_test, predicted_ids, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, predicted_ids, average="weighted")), 4),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix,
    }

    model_path = ARTIFACT_DIR / "tensorflow_risk_model.keras"
    print(f"Saving TensorFlow weak-risk model to {model_path}...", flush=True)
    model.save(model_path)

    metrics_path = OUTPUT_DIR / "tensorflow_risk_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / "tensorflow_risk_classification_report.csv")
    write_confusion_matrix_png(labels, matrix, OUTPUT_DIR / "tensorflow_risk_confusion_matrix.png")

    base_error_columns = [
        column
        for column in ["ticket_id", "external_id", "customer_message", INPUT_FEATURE, LABEL_COLUMN, "risk_signal_terms"]
        if column in test_df.columns
    ]
    error_df = test_df[base_error_columns].copy()
    error_df["true_high_risk_policy_label"] = y_test
    error_df["predicted_high_risk_policy_label"] = predicted_ids
    error_df["risk_confidence"] = [round(float(max(value, 1 - value)), 4) for value in probabilities]
    error_df["risk_correct"] = error_df["true_high_risk_policy_label"] == error_df["predicted_high_risk_policy_label"]
    error_df = error_df[~error_df["risk_correct"]].copy()
    error_df.to_csv(OUTPUT_DIR / "tensorflow_risk_error_analysis.csv", index=False)

    comparison_path = write_model_comparison()
    return {
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "comparison_path": str(comparison_path),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "label_quality_note": metrics["data"]["label_quality_note"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train optional TensorFlow weak-risk classifier for Ticket Intelligence.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    summary = train_tensorflow_risk(epochs=args.epochs, batch_size=args.batch_size)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
