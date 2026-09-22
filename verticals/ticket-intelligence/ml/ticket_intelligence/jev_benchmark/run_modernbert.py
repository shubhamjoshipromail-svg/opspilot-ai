"""Run the fine-tuned ModernBERT from the Hub on the same fixed test split.

Produces predictions in the identical schema to run_jev.py, so the two can be
compared row by row (McNemar) and on the same coverage curve.

The Hub model has no Inference Provider deployed, so this downloads the weights
(~0.1B params) and runs locally. On Apple Silicon it uses MPS.

Requires:  pip install torch transformers

Usage:
    python run_modernbert.py --limit 50
    python run_modernbert.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from taxonomy import to_clean_v1

DEFAULT_MODEL = "shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1"

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parents[2] / "data" / "processed" / "test.csv"
DEFAULT_OUT = HERE / "outputs" / "modernbert_parent_queue_predictions.csv"


def build_text(row: pd.Series) -> str:
    """Match the v2 training input exactly: subject + blank line + body."""
    subject = str(row.get("subject") or "").strip()
    body = str(row.get("body") or "").strip()
    if not subject and not body:
        return str(row.get("customer_message") or "").strip()
    return f"{subject}\n\n{body}".strip()


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fine-tuned ModernBERT on the test split.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise SystemExit("Missing deps. Run: pip install torch transformers")

    device = pick_device(args.device)
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    id2label = model.config.id2label
    labels = [id2label[i] for i in sorted(id2label)]
    print(f"{len(labels)} labels: {labels}")

    df = pd.read_csv(args.input)
    label_col = "category" if "category" in df.columns else "true_category"
    id_col = "ticket_id" if "ticket_id" in df.columns else "external_id"
    if args.limit:
        df = df.head(args.limit)

    # test.csv carries 10-class labels; this model predicts the merged space.
    true_labels = df[label_col].astype(str).map(to_clean_v1)
    texts = [build_text(row) for _, row in df.iterrows()]

    records = []
    started = time.perf_counter()
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        encoded = tokenizer(batch, truncation=True, max_length=args.max_length,
                            padding=True, return_tensors="pt").to(device)
        batch_started = time.perf_counter()
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu()
        per_item_ms = (time.perf_counter() - batch_started) * 1000.0 / len(batch)

        for offset, row_probs in enumerate(probs):
            idx = start + offset
            ordered, _ = torch.sort(row_probs, descending=True)
            top1 = float(ordered[0])
            top2 = float(ordered[1]) if len(ordered) > 1 else 0.0
            predicted = labels[int(torch.argmax(row_probs))]
            true_label = true_labels.iloc[idx]

            record = {
                "ticket_id": str(df.iloc[idx][id_col]),
                "true_label": true_label,
                "predicted_label": predicted,
                # Softmax max, matching how routing.py reads model confidence.
                "confidence": round(top1, 6),
                "top_1_probability": round(top1, 6),
                "top_2_probability": round(top2, 6),
                "top1_top2_margin": round(top1 - top2, 6),
                "correct": int(predicted == true_label),
                "latency_ms": round(per_item_ms, 2),
                "input_tokens": "",
                "model": args.model,
            }
            for label_idx, label in enumerate(labels):
                record[f"prob_{label}"] = round(float(row_probs[label_idx]), 6)
            records.append(record)

        done = min(start + args.batch_size, len(texts))
        if done % (args.batch_size * 8) == 0 or done == len(texts):
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"  {done}/{len(texts)}  acc={acc:.4f}")

    out = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    elapsed = time.perf_counter() - started
    print(f"\nWrote {len(out)} predictions to {args.out}")
    print(f"Accuracy: {out['correct'].mean():.4f}")
    print(f"Wall clock: {elapsed:.1f}s  ({len(out) / elapsed:.1f} tickets/s on {device})")


if __name__ == "__main__":
    main()
