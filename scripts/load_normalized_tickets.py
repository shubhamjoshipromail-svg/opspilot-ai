import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import SessionLocal  # noqa: E402
from backend.app.models import Ticket  # noqa: E402


TICKET_COLUMNS = [
    "external_id",
    "customer_message",
    "true_category",
    "true_priority",
    "status",
    "channel",
    "source",
    "ticket_type",
    "language",
    "reference_answer",
    "tags",
]


def clean_value(value: Any) -> Any:
    """Convert pandas missing values to None before inserting into SQLAlchemy."""
    if pd.isna(value):
        return None
    return value


def row_to_ticket_payload(row: pd.Series) -> dict[str, Any]:
    payload = {column: clean_value(row.get(column)) for column in TICKET_COLUMNS}
    if not payload.get("status"):
        payload["status"] = "new"
    return payload


def load_tickets(input_path: Path, limit: int | None, clear_existing: bool) -> int:
    df = pd.read_csv(input_path)
    if limit is not None:
        df = df.head(limit)

    db = SessionLocal()
    loaded_count = 0

    try:
        if clear_existing:
            db.query(Ticket).delete()
            db.commit()

        existing_external_ids = {
            external_id
            for (external_id,) in db.query(Ticket.external_id).filter(Ticket.external_id.isnot(None)).all()
        }

        for _, row in df.iterrows():
            payload = row_to_ticket_payload(row)
            external_id = payload.get("external_id")

            if external_id and external_id in existing_external_ids:
                continue

            db.add(Ticket(**payload))
            loaded_count += 1

            if external_id:
                existing_external_ids.add(external_id)

        db.commit()
        return loaded_count
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load normalized English tickets into the tickets table.")
    parser.add_argument("--input", required=True, type=Path, help="Path to tickets_en_normalized.csv")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for test loads.")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing tickets before loading the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded_count = load_tickets(args.input, args.limit, args.clear_existing)
    print(f"Loaded {loaded_count} tickets.")


if __name__ == "__main__":
    main()
