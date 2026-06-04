"""Fine-tune a transformer classifier for OpsPilot Ticket Intelligence v2.

Default task:
    parent_queue classification from subject + body.

The script intentionally excludes answer/reference_answer from model inputs
because those fields are post-resolution information and would leak labels.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import random
import re
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from data_utils import ARTIFACT_DIR, DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, DEFAULT_VAL_PATH, OUTPUT_DIR, RANDOM_STATE


DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"
FALLBACK_MODELS = [
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-small",
    "distilbert-base-uncased",
]

TASKS = {
    "parent_queue": {
        "label_candidates": ["category", "true_category", "parent_queue"],
        "input_preference": "subject_body",
        "description": "Normalized parent queue/category routing.",
    },
    "priority": {
        "label_candidates": ["priority", "true_priority"],
        "input_preference": "subject_body",
        "description": "Source priority label prediction.",
    },
    "ticket_type": {
        "label_candidates": ["ticket_type", "type"],
        "input_preference": "subject_body",
        "description": "Ticket type prediction.",
    },
    "raw_queue": {
        "label_candidates": ["queue", "true_category", "category"],
        "input_preference": "subject_body",
        "description": "Raw queue label prediction.",
    },
}


def require_transformer_dependencies() -> None:
    missing: list[str] = []
    for package in ["torch", "transformers", "datasets", "evaluate", "accelerate"]:
        try:
            __import__(package)
        except ModuleNotFoundError:
            missing.append(package)
    if missing:
        raise ModuleNotFoundError(
            "Missing transformer dependencies: "
            + ", ".join(missing)
            + ". Install with `pip install torch transformers datasets evaluate accelerate sentencepiece` "
            "or `pip install -r verticals/ticket-intelligence/requirements.txt`."
        )


def normalize_label(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def subject_body_input(df: pd.DataFrame) -> pd.Series | None:
    if {"subject", "body"}.issubset(df.columns):
        subject = df["subject"].fillna("").astype(str).str.strip()
        body = df["body"].fillna("").astype(str).str.strip()
        text = (subject + "\n\n" + body).str.strip()
        if text.str.len().gt(0).any():
            return text
    return None


def select_input_series(df: pd.DataFrame, task: str) -> tuple[pd.Series, str]:
    preferred = TASKS[task]["input_preference"]
    if preferred == "subject_body":
        text = subject_body_input(df)
        if text is not None:
            return text, "subject + body"
    for column in ["model_text", "customer_message"]:
        if column in df.columns:
            text = df[column].fillna("").astype(str).str.strip()
            if text.str.len().gt(0).any():
                return text, column
    raise ValueError("Could not find usable input text. Expected subject/body, model_text, or customer_message.")


def select_label_series(df: pd.DataFrame, task: str) -> tuple[pd.Series, str]:
    for column in TASKS[task]["label_candidates"]:
        if column in df.columns:
            labels = df[column].map(normalize_label)
            if labels.str.len().gt(0).any():
                return labels, column
    raise ValueError(f"Could not find label column for task {task!r}. Tried {TASKS[task]['label_candidates']}.")


def load_split(path: Path, task: str) -> tuple[pd.DataFrame, str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}. Run create_splits.py first.")
    df = pd.read_csv(path)
    text, input_source = select_input_series(df, task)
    labels, label_source = select_label_series(df, task)
    output = df.copy()
    output["input_text"] = text.map(clean_text)
    output["label_text"] = labels
    output = output[(output["input_text"].str.len() > 0) & (output["label_text"].str.len() > 0)].copy()
    return output, input_source, label_source


def sample_frame(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if not sample_size or sample_size <= 0 or sample_size >= len(df):
        return df
    if df["label_text"].nunique() > sample_size:
        return df.sample(n=sample_size, random_state=seed).copy()
    try:
        sampled, _ = train_test_split(
            df,
            train_size=sample_size,
            random_state=seed,
            stratify=df["label_text"],
        )
        return sampled.copy()
    except ValueError:
        return df.sample(n=sample_size, random_state=seed).copy()


def build_label_maps(labels: pd.Series) -> tuple[dict[str, int], dict[int, str]]:
    unique_labels = sorted(labels.dropna().astype(str).unique().tolist())
    if len(unique_labels) < 2:
        raise ValueError("Need at least two labels to train a classifier.")
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_labels(df: pd.DataFrame, label2id: dict[str, int], split_name: str) -> pd.DataFrame:
    unknown = sorted(set(df["label_text"]) - set(label2id))
    if unknown:
        raise ValueError(f"{split_name} has labels not present in train split: {unknown}")
    output = df.copy()
    output["label"] = output["label_text"].map(label2id).astype("int64")
    return output


def estimate_token_lengths(texts: pd.Series) -> dict[str, float]:
    counts = texts.fillna("").astype(str).map(lambda value: len(re.findall(r"\S+", value)))
    return {
        "mean_whitespace_tokens": round(float(counts.mean()), 2),
        "max_whitespace_tokens": int(counts.max()),
        "p95_whitespace_tokens": round(float(counts.quantile(0.95)), 2),
    }


def print_split_audit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    input_source: str,
    label_source: str,
    task: str,
) -> None:
    print("=== Fixed Split Audit ===")
    print(f"Task: {task}")
    print(f"Selected input: {input_source}")
    print(f"Selected target: {label_source}")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n[{name}] rows: {len(df)}")
        print(f"[{name}] columns: {list(df.columns)}")
        print(f"[{name}] target counts: {df['label_text'].value_counts().to_dict()}")
        key_columns = [column for column in ["input_text", "label_text", "subject", "body", "model_text", "customer_message"] if column in df.columns]
        print(f"[{name}] missing values: {df[key_columns].isna().sum().to_dict()}")
    print("\nSample input text:")
    print(train_df["input_text"].iloc[0][:1000])
    print("\nToken length estimate:")
    print(json.dumps(estimate_token_lengths(train_df["input_text"]), indent=2))


def softmax(logits):
    import numpy as np

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def write_confusion_matrix_png(labels: list[str], matrix: list[list[int]], output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

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


def training_args_kwargs(transformers_module, args, output_dir: Path) -> dict[str, Any]:
    TrainingArguments = transformers_module.TrainingArguments
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "save_total_limit": 2,
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if "fp16" in signature:
        kwargs["fp16"] = bool(args.fp16)
    return kwargs


def train(args: argparse.Namespace) -> dict[str, Any]:
    require_transformer_dependencies()

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    import transformers

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("No CUDA GPU detected. Use Colab GPU for full ModernBERT training.")

    train_df, input_source, label_source = load_split(DEFAULT_TRAIN_PATH, args.task)
    val_df, val_input_source, val_label_source = load_split(DEFAULT_VAL_PATH, args.task)
    test_df, test_input_source, test_label_source = load_split(DEFAULT_TEST_PATH, args.task)
    if len({input_source, val_input_source, test_input_source}) > 1:
        print(f"Warning: split input sources differ: {input_source}, {val_input_source}, {test_input_source}")
    if len({label_source, val_label_source, test_label_source}) > 1:
        print(f"Warning: split label sources differ: {label_source}, {val_label_source}, {test_label_source}")

    train_df = sample_frame(train_df, args.sample_size, args.seed)
    val_sample_size = max(100, math.ceil(args.sample_size * 0.25)) if args.sample_size else None
    test_sample_size = max(100, math.ceil(args.sample_size * 0.25)) if args.sample_size else None
    val_df = sample_frame(val_df, val_sample_size, args.seed)
    test_df = sample_frame(test_df, test_sample_size, args.seed)

    print_split_audit(train_df, val_df, test_df, input_source, label_source, args.task)

    label2id, id2label = build_label_maps(train_df["label_text"])
    train_df = encode_labels(train_df, label2id, "train")
    val_df = encode_labels(val_df, label2id, "validation")
    test_df = encode_labels(test_df, label2id, "test")

    run_prefix = f"modernbert_{args.task}"
    artifact_dir = Path(args.output_dir) if args.output_dir else ARTIFACT_DIR / run_prefix
    artifact_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label={idx: label for idx, label in id2label.items()},
        label2id=label2id,
        trust_remote_code=args.trust_remote_code,
    )
    accepts_token_type_ids = "token_type_ids" in inspect.signature(model.forward).parameters
    print(f"Model accepts token_type_ids: {accepts_token_type_ids}")

    def tokenize(batch):
        encoded = tokenizer(batch["input_text"], truncation=True, max_length=args.max_length)
        if not accepts_token_type_ids and "token_type_ids" in encoded:
            encoded.pop("token_type_ids")
        return encoded

    train_dataset = Dataset.from_pandas(train_df[["input_text", "label"]], preserve_index=False).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df[["input_text", "label"]], preserve_index=False).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df[["input_text", "label"]], preserve_index=False).map(tokenize, batched=True)

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_kwargs(transformers, args, artifact_dir / "trainer")),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model(str(artifact_dir))
    tokenizer.save_pretrained(str(artifact_dir))

    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    y_true = predictions.label_ids
    y_pred = logits.argmax(axis=-1)
    probabilities = softmax(logits)
    confidence = probabilities.max(axis=1)
    labels = [id2label[idx] for idx in sorted(id2label)]
    predicted_labels = [id2label[int(idx)] for idx in y_pred]
    true_labels = [id2label[int(idx)] for idx in y_true]
    report = classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist()
    training_config = vars(args).copy()
    if training_config.get("output_dir") is not None:
        training_config["output_dir"] = str(training_config["output_dir"])
    metrics = {
        "model_version": f"ticket-intelligence-v2-transformer-{args.task}",
        "task": args.task,
        "model_name": args.model_name,
        "model_family": "transformer_sequence_classifier",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "input_source": input_source,
        "label_source": label_source,
        "answer_leakage_prevention": "answer/reference_answer excluded from input_text",
        "row_counts": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "training_config": training_config,
        "labels": labels,
        "label2id": label2id,
        "id2label": {str(idx): label for idx, label in id2label.items()},
        "accepts_token_type_ids": accepts_token_type_ids,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "classification_report": report,
        "confusion_matrix": matrix,
    }

    label_mapping_path = artifact_dir / "label_mapping.json"
    label_mapping_path.write_text(
        json.dumps({"label2id": label2id, "id2label": {str(idx): label for idx, label in id2label.items()}}, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text(json.dumps(metrics["training_config"], indent=2), encoding="utf-8")

    metrics_path = OUTPUT_DIR / f"{run_prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path = OUTPUT_DIR / f"{run_prefix}_classification_report.csv"
    pd.DataFrame(report).T.to_csv(report_path)
    confusion_path = OUTPUT_DIR / f"{run_prefix}_confusion_matrix.png"
    write_confusion_matrix_png(labels, matrix, confusion_path, f"{args.model_name} {args.task} Confusion Matrix")

    prediction_rows = test_df.copy()
    prediction_rows["true_label"] = true_labels
    prediction_rows["predicted_label"] = predicted_labels
    prediction_rows["confidence"] = [round(float(value), 4) for value in confidence]
    for rank in range(min(3, len(labels))):
        top_indices = probabilities.argsort(axis=1)[:, ::-1][:, rank]
        prediction_rows[f"top_{rank + 1}_label"] = [id2label[int(idx)] for idx in top_indices]
        prediction_rows[f"top_{rank + 1}_probability"] = [round(float(probabilities[row_idx, label_idx]), 4) for row_idx, label_idx in enumerate(top_indices)]

    prediction_columns = [
        column
        for column in [
            "ticket_id",
            "external_id",
            "input_text",
            "true_label",
            "predicted_label",
            "confidence",
            "top_1_label",
            "top_1_probability",
            "top_2_label",
            "top_2_probability",
            "top_3_label",
            "top_3_probability",
        ]
        if column in prediction_rows.columns
    ]
    predictions_path = OUTPUT_DIR / f"{run_prefix}_predictions.csv"
    prediction_rows[prediction_columns].to_csv(predictions_path, index=False)

    error_df = prediction_rows[prediction_rows["true_label"] != prediction_rows["predicted_label"]].copy()
    error_df["error_confidence_type"] = error_df["confidence"].map(lambda value: "low_confidence" if value < 0.60 else "high_confidence")
    error_columns = prediction_columns + ["error_confidence_type"]
    error_path = OUTPUT_DIR / f"{run_prefix}_error_analysis.csv"
    error_df[error_columns].to_csv(error_path, index=False)

    run_summary_path = OUTPUT_DIR / f"{run_prefix}_run_summary.md"
    run_summary_path.write_text(
        "\n".join(
            [
                f"# ModernBERT v2 Run Summary: {args.task}",
                "",
                f"- Model: `{args.model_name}`",
                f"- Input: `{input_source}`",
                f"- Label: `{label_source}`",
                "- Leakage prevention: `answer` / `reference_answer` excluded from input.",
                f"- Train/val/test rows: `{len(train_df)}` / `{len(val_df)}` / `{len(test_df)}`",
                f"- Accuracy: `{metrics['accuracy']}`",
                f"- Macro-F1: `{metrics['macro_f1']}`",
                f"- Weighted-F1: `{metrics['weighted_f1']}`",
                f"- Artifact path: `{artifact_dir}`",
                f"- Metrics path: `{metrics_path}`",
                "",
                "This is a transformer-based ticket-routing benchmark, not a production-ready decision system.",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({"metrics_path": str(metrics_path), "artifact_dir": str(artifact_dir), "macro_f1": metrics["macro_f1"]}, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer classifier for OpsPilot Ticket Intelligence v2.")
    parser.add_argument("--task", choices=sorted(TASKS), default="parent_queue")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training when CUDA supports it.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--print-fallbacks",
        action="store_true",
        help="Print recommended fallback model names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_fallbacks:
        print(json.dumps({"default": DEFAULT_MODEL_NAME, "fallbacks": FALLBACK_MODELS}, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
