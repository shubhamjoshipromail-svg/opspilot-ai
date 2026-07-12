"""Audit ticket datasets available in the OpsPilot repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_utils import PROJECT_ROOT, PROCESSED_DATA_DIR, audit_dataset, iter_dataset_files, select_largest_real_training_dataset


def audit_repository(project_root: Path = PROJECT_ROOT) -> list[dict]:
    return [audit_dataset(path, project_root) for path in iter_dataset_files(project_root)]


def find_seed_load_scripts(project_root: Path = PROJECT_ROOT) -> list[dict]:
    keywords = ("seed", "load", "import", "init")
    scripts: list[dict] = []
    for path in sorted(project_root.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name == "__init__.py":
            continue
        lowered = path.name.lower()
        if any(keyword in lowered for keyword in keywords):
            scripts.append({"file_path": str(path.relative_to(project_root))})
    return scripts


def write_reports(rows: list[dict], load_scripts: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_files": rows,
        "database_seed_load_scripts": load_scripts,
    }
    (output_dir / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_rows = []
    for row in rows:
        summary_rows.append(
            {
                "file_path": row.get("file_path"),
                "row_count": row.get("row_count"),
                "columns": ", ".join(row.get("columns", [])),
                "has_required_training_columns": row.get("has_required_training_columns"),
                "is_smoke_or_generated_output": row.get("is_smoke_or_generated_output"),
                "duplicate_customer_message_count": row.get("duplicate_customer_message_count"),
                "read_error": row.get("read_error"),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "data_audit_summary.csv", index=False)
    pd.DataFrame(load_scripts).to_csv(output_dir / "seed_load_scripts.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ticket datasets and identify the largest real training candidate.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    rows = audit_repository(args.project_root)
    load_scripts = find_seed_load_scripts(args.project_root)
    write_reports(rows, load_scripts, args.output_dir)
    best = select_largest_real_training_dataset(rows, args.project_root)

    print(
        json.dumps(
            {
                "dataset_count": len(rows),
                "database_seed_load_script_count": len(load_scripts),
                "largest_real_training_dataset": str(best) if best else None,
            },
            indent=2,
        )
    )
    if not best:
        print(
            "No real normalized training dataset was found. "
            "Synthetic smoke data remains available, but it will not be used for main training."
        )


if __name__ == "__main__":
    main()
