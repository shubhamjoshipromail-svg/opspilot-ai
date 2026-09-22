"""Run Jev (TypeSafe System One) on the fixed parent_queue test split.

Zero-shot: Jev sees no training data, unlike the fine-tuned ModernBERT baseline.
Outputs a predictions CSV in the schema threshold_analysis.py already reads
(true_label / predicted_label / confidence), so the existing tooling works
unchanged.

Stdlib-only HTTP so it runs in the current env without installing anything.
Resumable: re-running appends only the tickets not already present in --out.

Usage:
    export TYPESAFE_API_KEY=...
    python run_jev.py --limit 50            # smoke test first
    python run_jev.py                       # full 3,563-row split
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from taxonomy import INSTRUCTIONS, TAXONOMIES, to_clean_v1

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
QUESTION_ID = "queue"

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parents[2] / "data" / "processed" / "test.csv"
DEFAULT_OUT = HERE / "outputs" / "jev_parent_queue_predictions.csv"

_print_lock = threading.Lock()


def load_api_key() -> str | None:
    """Read the key from the environment, else from a gitignored .env file.

    A shell `export` does not survive between separate shells, so the .env file
    is the reliable path. .env is already covered by the repo's .gitignore.
    """
    key = os.environ.get("TYPESAFE_API_KEY")
    if key and key.strip() and key.strip() != "your-key-here":
        return key.strip()

    candidates = [HERE / ".env", HERE.parents[4] / ".env"]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() != "TYPESAFE_API_KEY":
                continue
            value = value.strip().strip("'\"")
            if value and value != "your-key-here":
                return value
    return None


def build_state(row: pd.Series) -> str:
    """Same input the ModernBERT run used: subject + blank line + body.

    reference_answer / answer are deliberately excluded (post-resolution leakage).
    """
    subject = str(row.get("subject") or "").strip()
    body = str(row.get("body") or "").strip()
    if not subject and not body:
        return str(row.get("customer_message") or "").strip()
    return f"{subject}\n\n{body}".strip()


def ask_jev(state: str, criteria: dict[str, str], api_key: str,
            max_retries: int = 6, timeout: float = 60.0) -> tuple[dict, float]:
    """POST one ticket. Returns (choice answer, latency_ms).

    Retries 429/529/5xx with exponential backoff and jitter, per the API docs.
    """
    payload = {
        "state": state,
        "model": MODEL,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": criteria,
            }
        },
    }
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                latency_ms = (time.perf_counter() - started) * 1000.0
                data = json.loads(resp.read().decode("utf-8"))
            answer = data["answers"][QUESTION_ID]
            answer["_usage"] = data.get("usage", {})
            answer["_model"] = data.get("model", "")
            return answer, latency_ms
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 529) or exc.code >= 500:
                if attempt == max_retries - 1:
                    raise
                sleep_for = min(2.0 ** attempt, 30.0) + random.uniform(0, 1.0)
                time.sleep(sleep_for)
                continue
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError):
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2.0 ** attempt, 30.0) + random.uniform(0, 1.0))

    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Jev on the parent_queue test split.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--taxonomy", choices=sorted(TAXONOMIES), default="original_10",
                        help="Label space to ask over. Score clean_v1 separately to "
                             "compare against the ~0.73 ModernBERT run.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N rows (smoke test).")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        sys.exit(
            "No TypeSafe API key found.\n"
            f"Write it into {HERE / '.env'} as:\n"
            "    TYPESAFE_API_KEY=sk-...\n"
            "(that path is gitignored; a shell export will not reach this script)"
        )

    labels, criteria = TAXONOMIES[args.taxonomy]
    # test.csv stores 10-class labels; fold them into the asked label space so
    # true and predicted are comparable.
    map_true = to_clean_v1 if args.taxonomy == "clean_v1" else (lambda label: label)

    df = pd.read_csv(args.input)
    label_col = "category" if "category" in df.columns else "true_category"
    id_col = "ticket_id" if "ticket_id" in df.columns else "external_id"
    if args.limit:
        df = df.head(args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        prior = pd.read_csv(args.out)
        done = set(prior["ticket_id"].astype(str))
        with _print_lock:
            print(f"Resuming: {len(done)} rows already present in {args.out.name}")

    todo = [row for _, row in df.iterrows() if str(row[id_col]) not in done]
    if not todo:
        print("Nothing to do; all rows already predicted.")
        return

    prob_cols = [f"prob_{label}" for label in labels]
    fieldnames = [
        "ticket_id", "true_label", "predicted_label", "confidence",
        "top_1_probability", "top_2_probability", "top1_top2_margin",
        "correct", "latency_ms", "input_tokens", "model",
    ] + prob_cols

    write_header = not args.out.exists()
    fh = args.out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    counters = {"done": 0, "correct": 0, "failed": 0}
    started_at = time.perf_counter()

    def work(row: pd.Series) -> None:
        ticket_id = str(row[id_col])
        true_label = map_true(str(row[label_col]))
        try:
            answer, latency_ms = ask_jev(build_state(row), criteria, api_key)
        except Exception as exc:  # noqa: BLE001 - record and continue the sweep
            with _print_lock:
                counters["failed"] += 1
                print(f"  ! {ticket_id}: {exc}")
            return

        probs = answer.get("probabilities", {}) or {}
        ordered = sorted(probs.values(), reverse=True)
        top1 = float(ordered[0]) if ordered else 0.0
        top2 = float(ordered[1]) if len(ordered) > 1 else 0.0
        predicted = answer.get("choice", "")
        is_correct = int(predicted == true_label)

        record = {
            "ticket_id": ticket_id,
            "true_label": true_label,
            "predicted_label": predicted,
            "confidence": answer.get("confidence", ""),
            "top_1_probability": round(top1, 6),
            "top_2_probability": round(top2, 6),
            "top1_top2_margin": round(top1 - top2, 6),
            "correct": is_correct,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": answer.get("_usage", {}).get("input_tokens", ""),
            "model": answer.get("_model", ""),
        }
        for label in labels:
            record[f"prob_{label}"] = round(float(probs.get(label, 0.0)), 6)

        with _print_lock:
            writer.writerow(record)
            fh.flush()
            counters["done"] += 1
            counters["correct"] += is_correct
            n = counters["done"]
            if n % 25 == 0 or n == len(todo):
                elapsed = time.perf_counter() - started_at
                running_acc = counters["correct"] / n
                rate = n / elapsed if elapsed else 0.0
                print(f"  {n}/{len(todo)}  acc={running_acc:.4f}  {rate:.1f} tickets/s")

    print(f"Taxonomy: {args.taxonomy} ({len(labels)} classes)")
    print(f"Running {len(todo)} tickets at concurrency {args.concurrency}...")
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, todo))
    finally:
        fh.close()

    elapsed = time.perf_counter() - started_at
    n = counters["done"]
    print(f"\nWrote {n} predictions to {args.out}")
    if n:
        print(f"Raw accuracy: {counters['correct'] / n:.4f}")
    print(f"Failed: {counters['failed']}  |  Wall clock: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
