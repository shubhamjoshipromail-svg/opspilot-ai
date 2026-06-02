"""Dataset discovery, auditing, and split helpers for Ticket Intelligence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


MODULE_DIR = Path(__file__).resolve().parent
VERTICAL_ROOT = MODULE_DIR.parents[1]
PROJECT_ROOT = VERTICAL_ROOT.parents[1]
SYNTHETIC_DATA_PATH = MODULE_DIR / "data" / "synthetic_tickets.csv"
PROCESSED_DATA_DIR = VERTICAL_ROOT / "data" / "processed"
DEFAULT_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
DEFAULT_VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
DEFAULT_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

KEY_FIELDS = [
    "customer_message",
    "true_category",
    "true_priority",
    "ticket_type",
    "language",
    "reference_answer",
    "tags",
]

TEXT_COLUMNS = ["customer_message", "message", "ticket_text", "text", "body", "description"]
CATEGORY_COLUMNS = ["true_category", "category", "label", "intent"]
PRIORITY_COLUMNS = ["true_priority", "priority", "urgency", "severity"]

SMOKE_DATA_FILENAMES = {"synthetic_tickets.csv"}
GENERATED_OUTPUT_DIR_NAMES = {"outputs", "__pycache__", "smoke_processed"}
GENERATED_REPORT_FILENAMES = {
    "data_audit.json",
    "data_audit_summary.csv",
    "seed_load_scripts.csv",
    "split_metadata.json",
    "train.csv",
    "val.csv",
    "test.csv",
}


@dataclass(frozen=True)
class ColumnMapping:
    text: str
    category: str
    priority: str | None


def normalize_label(value: Any) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def infer_column_mapping(df: pd.DataFrame) -> ColumnMapping | None:
    columns = list(df.columns)
    text = find_column(columns, TEXT_COLUMNS)
    category = find_column(columns, CATEGORY_COLUMNS)
    priority = find_column(columns, PRIORITY_COLUMNS)
    if not text or not category:
        return None
    return ColumnMapping(text=text, category=category, priority=priority)


def is_generated_or_smoke_path(path: Path) -> bool:
    parts = set(path.parts)
    return path.name in SMOKE_DATA_FILENAMES or path.name in GENERATED_REPORT_FILENAMES or bool(parts & GENERATED_OUTPUT_DIR_NAMES)


def iter_dataset_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    suffixes = {".csv", ".tsv", ".json", ".jsonl"}
    paths: list[Path] = []
    for path in project_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if ".git" in path.parts:
            continue
        if path.name in GENERATED_REPORT_FILENAMES:
            continue
        paths.append(path)
    return sorted(paths)


def read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
        return pd.DataFrame({"value": [payload]})
    raise ValueError(f"Unsupported dataset format: {path}")


def canonicalize_ticket_frame(df: pd.DataFrame) -> pd.DataFrame:
    mapping = infer_column_mapping(df)
    if mapping is None:
        raise ValueError("Dataset must include a customer message column and category label column.")

    output = df.copy()
    if mapping.text != "customer_message":
        output["customer_message"] = output[mapping.text]
    if mapping.category != "category":
        output["category"] = output[mapping.category]
    if mapping.priority and mapping.priority != "priority":
        output["priority"] = output[mapping.priority]
    elif "priority" not in output.columns:
        output["priority"] = None

    output["customer_message"] = output["customer_message"].astype(str).str.strip()
    output["category"] = output["category"].map(normalize_label)
    output["priority"] = output["priority"].map(normalize_label)
    output = output.dropna(subset=["customer_message", "category"])
    output["priority"] = output["priority"].fillna("unknown")
    if "ticket_id" not in output.columns:
        if "external_id" in output.columns:
            output["ticket_id"] = output["external_id"]
        else:
            output["ticket_id"] = [f"ticket-{idx:06d}" for idx in range(len(output))]
    output = output[output["customer_message"].str.len() > 0].copy()
    return output


def audit_dataset(path: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    try:
        df = read_dataset(path)
    except Exception as exc:
        return {
            "file_path": str(path.relative_to(project_root)),
            "read_error": f"{exc.__class__.__name__}: {exc}",
        }

    mapping = infer_column_mapping(df)
    columns = list(df.columns)
    key_missing = {
        field: int(df[field].isna().sum()) if field in df.columns else None
        for field in KEY_FIELDS
    }
    category_col = find_column(columns, CATEGORY_COLUMNS)
    priority_col = find_column(columns, PRIORITY_COLUMNS)
    language_col = find_column(columns, ["language", "lang", "locale"])
    text_col = find_column(columns, TEXT_COLUMNS)

    result: dict[str, Any] = {
        "file_path": str(path.relative_to(project_root)),
        "row_count": int(len(df)),
        "columns": columns,
        "missing_values_for_key_fields": key_missing,
        "category_distribution": {},
        "priority_distribution": {},
        "duplicate_customer_message_count": None,
        "language_distribution": {},
        "has_required_training_columns": mapping is not None,
        "is_smoke_or_generated_output": is_generated_or_smoke_path(path),
    }
    if category_col:
        result["category_distribution"] = df[category_col].fillna("<missing>").astype(str).value_counts().to_dict()
    if priority_col:
        result["priority_distribution"] = df[priority_col].fillna("<missing>").astype(str).value_counts().to_dict()
    if text_col:
        result["duplicate_customer_message_count"] = int(df[text_col].fillna("").astype(str).duplicated().sum())
    if language_col:
        result["language_distribution"] = df[language_col].fillna("<missing>").astype(str).value_counts().to_dict()
    return result


def select_largest_real_training_dataset(audit_rows: list[dict[str, Any]], project_root: Path = PROJECT_ROOT) -> Path | None:
    candidates = [
        row
        for row in audit_rows
        if row.get("has_required_training_columns")
        and not row.get("is_smoke_or_generated_output")
        and not row.get("read_error")
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda row: int(row.get("row_count", 0)))
    return project_root / best["file_path"]


def can_stratify(labels: pd.Series, test_size: float) -> bool:
    counts = labels.value_counts()
    if counts.empty or counts.min() < 2:
        return False
    expected_test = counts * test_size
    return bool((expected_test >= 1).all())


def create_fixed_splits(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if round(train_size + val_size + test_size, 6) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    df = canonicalize_ticket_frame(df)
    stratify_first = df["category"] if can_stratify(df["category"], val_size + test_size) else None
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=stratify_first,
    )

    relative_test_size = test_size / (val_size + test_size)
    stratify_second = temp_df["category"] if can_stratify(temp_df["category"], relative_test_size) else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=stratify_second,
    )

    strategy = "category_stratified" if stratify_first is not None else "random_no_stratification"
    if stratify_first is not None and stratify_second is None:
        strategy = "category_stratified_train_temp_random_val_test"
    return train_df.copy(), val_df.copy(), test_df.copy(), strategy


def load_default_splits() -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame] | None:
    if DEFAULT_TRAIN_PATH.exists() and DEFAULT_TEST_PATH.exists():
        train_df = canonicalize_ticket_frame(pd.read_csv(DEFAULT_TRAIN_PATH))
        test_df = canonicalize_ticket_frame(pd.read_csv(DEFAULT_TEST_PATH))
        val_df = canonicalize_ticket_frame(pd.read_csv(DEFAULT_VAL_PATH)) if DEFAULT_VAL_PATH.exists() else None
        return train_df, val_df, test_df
    return None


def load_smoke_split(random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    df = canonicalize_ticket_frame(pd.read_csv(SYNTHETIC_DATA_PATH))
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=random_state,
        stratify=df["category"],
    )
    return train_df.copy(), None, test_df.copy()
