# Backend

This folder will contain the FastAPI backend for OpsPilot AI.

Current status: **Phase 1 backend/database vertical slice**.

Planned responsibilities:

- Ticket intake and case management APIs.
- Postgres persistence.
- Model orchestration interfaces.
- Retrieval and recommendation service boundaries.
- Human review workflow endpoints.
- Evaluation trace storage.
- Analytics endpoints.

Not implemented yet:

- ML models.
- RAG or policy retrieval.
- LLM calls.
- AI evaluation.
- Analytics dashboards.
- Full Streamlit UI.
- Authentication or authorization.

## Phase 1: Ticket CRUD With Postgres

Phase 1 implements the first real backend/database slice:

- Create a ticket.
- Save it to the configured database.
- List tickets.
- View one ticket.
- Partially update a ticket.
- Delete a ticket.

The backend reads `DATABASE_URL` from `.env`. Railway Postgres URLs using the `postgresql://` scheme are supported. If `DATABASE_URL` is missing, the app falls back to local SQLite at `sqlite:///./opspilot_dev.db`.

## Phase 1.6: Normalized English Ticket Dataset

The current normalized English dataset is expected at:

```text
data/processed/tickets_en_normalized.csv
```

It includes:

- Routing/category label: `true_category`
- Priority label: `true_priority`
- Ticket type: `ticket_type`
- Reference answer: `reference_answer`
- Tags stored as pipe-separated text for the MVP: `tags`

The `tickets` table now includes these dataset-aligned fields in addition to the original Phase 1 CRUD fields.

Important database note: Alembic migrations are intentionally not implemented yet. If the `tickets` table already exists without the new Phase 1.6 columns, drop/recreate it manually in development or reset the database before running `python -m backend.app.db.init_db`.

### Configure Environment

From the repository root:

```bash
cp .env.example .env
```

Paste your Railway Postgres connection string into `.env`:

```text
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
APP_ENV=development
```

### Initialize Tables

```bash
python -m backend.app.db.init_db
```

This creates the current SQLAlchemy tables. Alembic migrations are intentionally not included yet.

### Load Normalized Tickets

For a small test load:

```bash
python scripts/load_normalized_tickets.py \
  --input data/processed/tickets_en_normalized.csv \
  --limit 100 \
  --clear-existing
```

Then start the backend and list tickets:

```bash
uvicorn backend.app.main:app --reload
curl http://127.0.0.1:8000/tickets
```

### Run The Backend

```bash
uvicorn backend.app.main:app --reload
```

### Example Requests

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Create a ticket:

```bash
curl -X POST http://127.0.0.1:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "external_id": "TICKET-001",
    "customer_message": "My order never arrived and I want a refund.",
    "channel": "email",
    "source": "sample",
    "true_category": "shipping_delay",
    "true_priority": "high"
  }'
```

List tickets:

```bash
curl http://127.0.0.1:8000/tickets
```

Get one ticket:

```bash
curl http://127.0.0.1:8000/tickets/1
```

Update a ticket:

```bash
curl -X PATCH http://127.0.0.1:8000/tickets/1 \
  -H "Content-Type: application/json" \
  -d '{
    "status": "in_review"
  }'
```

Delete a ticket:

```bash
curl -X DELETE http://127.0.0.1:8000/tickets/1
```

### Run Tests

```bash
pytest
```
