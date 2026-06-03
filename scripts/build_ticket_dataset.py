"""Regenerate the OpsPilot English ticket dataset from a public HF source.

Source dataset:
    Tobi-Bueck/customer-support-tickets

Outputs:
    data/raw/customer_support_tickets_en.csv
    data/processed/tickets_en_normalized.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "customer_support_tickets_en.csv"
NORMALIZED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "tickets_en_normalized.csv"
DATASET_NAME = "Tobi-Bueck/customer-support-tickets"
DATASET_SPLIT = "train"


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _combine_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = []
    for column in columns:
        if column in df.columns:
            parts.append(df[column].fillna("").astype(str).str.strip())
    if not parts:
        raise ValueError("Could not create customer_message: no usable text columns found.")
    combined = parts[0]
    for part in parts[1:]:
        combined = (combined + "\n\n" + part).str.strip()
    return combined.str.strip()


def _build_tags(df: pd.DataFrame) -> pd.Series:
    tag_columns = [column for column in df.columns if column.lower().startswith("tag")]
    if not tag_columns:
        tag_column = _first_existing(list(df.columns), ["tags", "labels"])
        if tag_column:
            return df[tag_column].fillna("").astype(str).str.strip()
        return pd.Series([""] * len(df), index=df.index)

    def clean_tags(row: pd.Series) -> str:
        tags: list[str] = []
        for column in tag_columns:
            value = row.get(column)
            if pd.notna(value) and str(value).strip().lower() not in {"", "none", "nan"}:
                tags.append(str(value).strip())
        return "|".join(tags)

    return df.apply(clean_tags, axis=1)


def _normalize_label(value: Any) -> str | None:
    if pd.isna(value):
        return None
    label = str(value).strip()
    return label or None


def load_source_dataset(dataset_name: str = DATASET_NAME, split: str = DATASET_SPLIT) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'datasets' package is required. Install dependencies with "
            "`pip install -r requirements.txt` or `pip install datasets`."
        ) from exc

    dataset = load_dataset(dataset_name)
    if split not in dataset:
        raise ValueError(f"Dataset {dataset_name!r} does not contain split {split!r}. Available: {list(dataset)}")
    return dataset[split].to_pandas()


def normalize_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    language_col = _first_existing(list(df.columns), ["language", "lang", "locale"])
    if language_col:
        df_en = df[df[language_col].astype(str).str.lower().eq("en")].copy()
        non_en_count = len(df) - len(df_en)
        print(f"English filter: kept {len(df_en)} rows, excluded {non_en_count} non-English rows.")
    else:
        df_en = df.copy()
        print("No language column found; retaining all rows and documenting language as missing.")

    if df_en.empty:
        raise ValueError("English-only dataset is empty after filtering.")

    columns = list(df_en.columns)
    id_col = _first_existing(columns, ["id", "ticket_id", "external_id"])
    category_col = _first_existing(columns, ["queue", "category", "group", "department", "true_category"])
    priority_col = _first_existing(columns, ["priority", "true_priority", "urgency", "severity"])
    type_col = _first_existing(columns, ["type", "ticket_type", "request_type", "incident_type"])
    answer_col = _first_existing(columns, ["answer", "agent_answer", "resolution", "reference_answer", "response"])

    text_candidates = [
        column
        for column in [
            _first_existing(columns, ["customer_message", "message", "ticket_text", "text"]),
            _first_existing(columns, ["subject", "title"]),
            _first_existing(columns, ["body", "description", "content"]),
        ]
        if column
    ]
    if not text_candidates:
        raise ValueError(
            "Could not create customer_message. Expected one of customer_message/message/text, "
            "or subject/title plus body/description."
        )
    if not category_col and not priority_col:
        raise ValueError("Neither true_category nor true_priority can be created from the source dataset.")

    normalized = pd.DataFrame(index=df_en.index)
    if id_col:
        normalized["external_id"] = df_en[id_col].fillna("").astype(str).str.strip()
    else:
        normalized["external_id"] = [f"hf-tobi-{idx + 1:06d}" for idx in range(len(df_en))]

    normalized["customer_message"] = _combine_text(df_en, text_candidates)
    normalized["true_category"] = df_en[category_col].map(_normalize_label) if category_col else None
    normalized["true_priority"] = df_en[priority_col].map(_normalize_label) if priority_col else None
    normalized["status"] = "new"
    normalized["channel"] = "dataset"
    normalized["source"] = DATASET_NAME
    normalized["ticket_type"] = df_en[type_col].map(_normalize_label) if type_col else None
    normalized["language"] = df_en[language_col].map(_normalize_label) if language_col else "en"
    normalized["reference_answer"] = df_en[answer_col].map(_clean_text) if answer_col else None
    normalized["tags"] = _build_tags(df_en)

    before_missing = len(normalized)
    normalized = normalized[normalized["customer_message"].fillna("").astype(str).str.strip().str.len() > 0].copy()
    dropped_missing_message = before_missing - len(normalized)

    duplicate_count = int(normalized["customer_message"].duplicated().sum())
    if duplicate_count:
        normalized = normalized.drop_duplicates(subset=["customer_message"]).copy()

    if language_col:
        bad_languages = sorted(set(normalized["language"].dropna().astype(str).str.lower()) - {"en"})
        if bad_languages:
            raise ValueError(f"Non-English rows remain after filtering: {bad_languages}")

    print(f"Rows dropped for missing customer_message: {dropped_missing_message}")
    print(f"Exact duplicate customer_message rows removed: {duplicate_count}")
    return df_en, normalized.reset_index(drop=True)


def print_dataset_report(normalized: pd.DataFrame) -> None:
    print(f"Normalized shape: {normalized.shape}")
    print(f"Normalized columns: {list(normalized.columns)}")
    print("\nMissing values:")
    print(normalized.isna().sum().to_string())
    if "true_category" in normalized.columns:
        print("\nCategory distribution:")
        print(normalized["true_category"].fillna("<missing>").value_counts().to_string())
    if "true_priority" in normalized.columns:
        print("\nPriority distribution:")
        print(normalized["true_priority"].fillna("<missing>").value_counts().to_string())
    if "language" in normalized.columns:
        print("\nLanguage distribution:")
        print(normalized["language"].fillna("<missing>").value_counts().to_string())


def build_dataset(dataset_name: str = DATASET_NAME, split: str = DATASET_SPLIT) -> tuple[Path, Path]:
    raw_df = load_source_dataset(dataset_name, split)
    raw_en, normalized = normalize_dataset(raw_df)

    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NORMALIZED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_en.to_csv(RAW_OUTPUT_PATH, index=False)
    normalized.to_csv(NORMALIZED_OUTPUT_PATH, index=False)

    print(f"\nSaved raw English dataset: {RAW_OUTPUT_PATH}")
    print(f"Saved normalized dataset: {NORMALIZED_OUTPUT_PATH}")
    print_dataset_report(normalized)
    return RAW_OUTPUT_PATH, NORMALIZED_OUTPUT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the English OpsPilot ticket dataset from Hugging Face.")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", default=DATASET_SPLIT)
    args = parser.parse_args()
    build_dataset(args.dataset_name, args.split)


if __name__ == "__main__":
    main()
