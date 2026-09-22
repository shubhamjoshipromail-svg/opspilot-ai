# Backend

This folder contains the canonical FastAPI backend for OpsPilot AI.

Current status: **MVP product backbone**.

Planned responsibilities:

- Ticket intake and case management APIs.
- Postgres persistence.
- Model orchestration interfaces.
- Retrieval and recommendation service boundaries.
- Human review workflow endpoints.
- Evaluation trace storage.
- Analytics endpoints.

Implemented:

- Ticket CRUD.
- Hugging Face Ticket Intelligence routing.
- Database-backed routing prediction logging.
- Dashboard summary and recent-prediction APIs.
- Alembic database migrations.

Not implemented yet:

- RAG or policy retrieval.
- LLM calls.
- AI evaluation.
- Streamlit dashboard pages.
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
alembic upgrade head
```

This creates or upgrades the `tickets` and `ticket_routing_predictions` tables.
`python -m backend.app.db.init_db` remains available as a local-development
fallback, but Alembic is the canonical schema-management path.

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

Route and log a ticket prediction:

```bash
curl -X POST http://127.0.0.1:8000/api/ticket-intelligence/route \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Invoice payment issue",
    "body": "I was charged twice for my subscription."
  }'
```

Dashboard summary:

```bash
curl http://127.0.0.1:8000/api/dashboard/summary
```

Recent predictions:

```bash
curl "http://127.0.0.1:8000/api/dashboard/recent-predictions?limit=20"
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
