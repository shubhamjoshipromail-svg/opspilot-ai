"""Transformer-ready training pipeline for the second model layer.

This script is intentionally dependency-gated. The local workspace can train the
baseline offline; when torch/transformers/datasets are installed, this file
fine-tunes a compact sequence classifier such as MiniLM or DistilBERT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


MODULE_DIR = Path(__file__).resolve().parent
DATA_PATH = MODULE_DIR / "data" / "synthetic_tickets.csv"
OUTPUT_DIR = MODULE_DIR / "outputs" / "transformer"


def _dependency_status() -> dict[str, str]:
    status: dict[str, str] = {}
    for package in ["torch", "transformers", "datasets", "evaluate"]:
        try:
            module = __import__(package)
            status[package] = getattr(module, "__version__", "installed")
        except Exception as exc:
            status[package] = f"missing: {exc.__class__.__name__}"
    return status


def train(model_name: str, data_path: Path = DATA_PATH, epochs: int = 3) -> dict:
    deps = _dependency_status()
    missing = [name for name, value in deps.items() if value.startswith("missing")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if missing:
        status = {
            "status": "skipped",
            "reason": "Transformer dependencies are not installed in this workspace.",
            "missing": missing,
            "dependency_status": deps,
            "next_command": "pip install -r verticals/ticket-intelligence/requirements.txt",
        }
        (OUTPUT_DIR / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return status

    import evaluate
    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    df = pd.read_csv(data_path)
    labels = sorted(df["category"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    df["label"] = df["category"].map(label_to_id)
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["category"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["customer_message"], truncation=True, max_length=192)

    train_dataset = Dataset.from_pandas(train_df[["customer_message", "label"]]).map(tokenize, batched=True)
    eval_dataset = Dataset.from_pandas(test_df[["customer_message", "label"]]).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=y_pred, references=y_true)["accuracy"],
            "macro_f1": f1_metric.compute(predictions=y_pred, references=y_true, average="macro")["f1"],
        }

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(OUTPUT_DIR / "best_model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "best_model"))
    status = {
        "status": "trained",
        "base_model": model_name,
        "labels": labels,
        "metrics": metrics,
    }
    (OUTPUT_DIR / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(train(args.model_name, args.data, args.epochs), indent=2))


if __name__ == "__main__":
    main()
