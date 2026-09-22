# Development Guide

Last reviewed: 2026-07-12

## Environment

Create a local environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy environment settings:

```bash
cp .env.example .env
```

Important variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | SQLAlchemy database URL | Local SQLite fallback |
| `OPSPILOT_ROUTING_THRESHOLD` | Auto-route confidence threshold | `0.80` |
| `OPSPILOT_API_URL` | Streamlit frontend API base URL | `http://localhost:8000` |
| `OPSPILOT_API_TIMEOUT` | Streamlit API timeout in seconds | `120` |

Do not commit `.env`, secrets, local database files, caches, or generated model
artifacts.

## Database

Run migrations:

```bash
alembic upgrade head
```

Development fallback:

```bash
python -m backend.app.db.init_db
```

Load a small local ticket sample after regenerating the dataset:

```bash
python scripts/load_normalized_tickets.py \
  --input data/processed/tickets_en_normalized.csv \
  --limit 100
```

Use `--clear-existing` only when you intentionally want to remove existing local
ticket rows.

## Backend

Run the FastAPI backend:

```bash
uvicorn backend.app.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ticket routing smoke call:

```bash
curl -X POST http://127.0.0.1:8000/api/ticket-intelligence/route \
  -H 'Content-Type: application/json' \
  -d '{"subject":"Invoice payment issue","body":"I was charged twice for my subscription."}'
```

The first routing call may be slow if the hosted Hugging Face model is cold.

## Frontend

Run Streamlit from the repository root:

```bash
streamlit run frontend/streamlit_app.py
```

If the backend is not on `localhost:8000`:

```bash
export OPSPILOT_API_URL=http://127.0.0.1:8000
streamlit run frontend/streamlit_app.py
```

## Tests

Run all tests:

```bash
pytest
```

Current verified result on 2026-07-12:

```text
18 passed
```

The tests mock the hosted model where practical so the suite does not depend on
network model availability.

## Dataset Regeneration

Regenerate local source data:

```bash
python scripts/build_ticket_dataset.py
```

Create fixed splits:

```bash
cd verticals/ticket-intelligence
python ml/ticket_intelligence/create_splits.py --input ../../data/processed/tickets_en_normalized.csv
```

Generated CSVs and split archives are local artifacts and should remain ignored
unless the project deliberately changes artifact policy.

## Deployment Notes

Railway backend deployment uses:

- `Dockerfile`
- `requirements-railway.txt`
- `railway.json`
- app target: `backend.app.main:app`
- health path: `/health`

The Railway runtime installs CPU PyTorch and Transformers for hosted model
inference.
