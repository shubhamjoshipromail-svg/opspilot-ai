"""Create fixed train/validation/test splits from the largest real ticket dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_utils import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_PATH,
    VERTICAL_ROOT,
    audit_dataset,
    create_fixed_splits,
    iter_dataset_files,
    read_dataset,
    select_largest_real_training_dataset,
)


def resolve_input_path(input_path: Path | None, project_root: Path, allow_synthetic_smoke: bool) -> Path:
    if input_path:
        return input_path
    audit_rows = [audit_dataset(path, project_root) for path in iter_dataset_files(project_root)]
    selected = select_largest_real_training_dataset(audit_rows, project_root)
    if selected:
        return selected
    if allow_synthetic_smoke:
        return SYNTHETIC_DATA_PATH
    raise FileNotFoundError(
        "No real normalized ticket dataset found. Add a CSV such as "
        "data/processed/tickets_en_normalized.csv with customer_message, true_category, "
        "true_priority, ticket_type, language, reference_answer, and tags; or pass "
        "--allow-synthetic-smoke for smoke-test splits only."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed train/val/test CSV splits for Ticket Intelligence.")
    parser.add_argument("--input", type=Path, default=None, help="Optional explicit dataset path.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--allow-synthetic-smoke", action="store_true")
    args = parser.parse_args()

    input_path = resolve_input_path(args.input, args.project_root, args.allow_synthetic_smoke)
    df = read_dataset(input_path)
    train_df, val_df, test_df, strategy = create_fixed_splits(
        df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    synthetic_smoke_split = input_path.resolve() == SYNTHETIC_DATA_PATH.resolve()
    output_dir = args.output_dir
    if synthetic_smoke_split and output_dir.resolve() == PROCESSED_DATA_DIR.resolve():
        output_dir = VERTICAL_ROOT / "data" / "smoke_processed"

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / DEFAULT_TRAIN_PATH.name
    val_path = output_dir / DEFAULT_VAL_PATH.name
    test_path = output_dir / DEFAULT_TEST_PATH.name
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "source_dataset": str(input_path),
        "random_state": args.random_state,
        "split_strategy": strategy,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "synthetic_smoke_split": synthetic_smoke_split,
        "output_dir": str(output_dir),
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
