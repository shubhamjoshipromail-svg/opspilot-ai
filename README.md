# OpsPilot

**A modular AI operations platform for controlled ticket routing, human review,
workflow auditability, and operational analytics.**

OpsPilot is designed to connect company databases, internal tools, specialized
machine-learning models, retrieval systems, and future agentic workflows into a
controlled decision-support system.

It is not intended to be a single chatbot or a standalone classifier. The
product goal is an AI operations backbone where independent AI modules can be
plugged into business workflows behind stable APIs, persistent records, human
review rules, and measurable outcomes.

## Current MVP

The current MVP focuses on service-ticket intelligence:

- Ticket CRUD and database persistence.
- Seven-queue ticket routing using a hosted ModernBERT model.
- Confidence scoring and complete probability distributions.
- `auto_route` versus `human_review` decisions.
- Prediction persistence and model metadata.
- Database-backed dashboard metrics.
- An operator-facing Streamlit console.

### Implementation Status

| Capability | Status |
| --- | --- |
| Canonical FastAPI backend | Implemented |
| Ticket CRUD | Implemented |
| Hugging Face routing service | Implemented |
| Prediction persistence | Implemented |
| Alembic migrations | Implemented |
| Dashboard APIs | Implemented |
| Canonical Streamlit MVP | Implemented |
| API-client error handling | Implemented |
| Curated demo-data seed | Planned |
| Human review records and overrides | Planned |
| Policy retrieval/RAG | Planned |
| Recommendations and response drafting | Planned |
| Authentication and permissions | Planned |
| Agent/tool orchestration | Planned |

## Product Loop

```text
Operator enters ticket subject and body
                    |
                    v
         Streamlit calls FastAPI
                    |
                    v
     Ticket Intelligence runs routing
                    |
                    v
  Queue + confidence + review decision
                    |
                    v
       Prediction stored in database
                    |
                    v
 Dashboard and model analytics update
```

The routing endpoint is intentionally only one AI module. Future modules should
follow the same pattern: stable contract, service boundary, persistent output,
version metadata, human-control rules, and measurable behavior.

## Canonical Architecture

```text
frontend/streamlit_app.py
        |
        | HTTP
        v
backend/app (FastAPI)
        |
        +-- Ticket CRUD
        +-- Ticket Intelligence
        +-- Prediction logging
        +-- Dashboard services
        |
        v
SQLAlchemy + Alembic
        |
        +-- PostgreSQL in hosted environments
        +-- SQLite fallback for development

verticals/ticket-intelligence/ml
        |
        +-- Research, training, and evaluation only
```

Canonical boundaries:

- `backend/app` is the deployable product backend.
- `frontend/streamlit_app.py` is the canonical MVP frontend.
- `verticals/ticket-intelligence/ml` is the research and training lab.
- `verticals/ticket-intelligence/app` is a legacy standalone prototype unless
  explicitly revived.
- `data/policies` contains mock policy material for future retrieval work.
- Model binaries and large generated datasets are not committed to Git.

See [Product Architecture](docs/PRODUCT_ARCHITECTURE.md) for the complete
architecture, module boundaries, build order, and glossary.

## Repository Map

```text
Ops-copilot/
├── backend/
│   ├── README.md
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── database.py
│       ├── db/
│       ├── models/
│       ├── routes/
│       ├── schemas/
│       └── services/
├── frontend/
│   ├── api_client.py
│   ├── streamlit_app.py
│   └── README.md
├── migrations/
│   └── versions/
├── verticals/
│   └── ticket-intelligence/
│       ├── ml/
│       ├── notebooks/
│       ├── docs/
│       └── app/                  # legacy prototype
├── data/
│   ├── policies/
│   ├── raw/
│   └── processed/
├── docs/
├── scripts/
├── tests/
├── Dockerfile
├── railway.json
└── requirements.txt
```

## Backend API

The canonical application is `backend.app.main:app`.

### Health

```text
GET /health
```

### Tickets

```text
POST   /tickets
GET    /tickets
GET    /tickets/{ticket_id}
PATCH  /tickets/{ticket_id}
DELETE /tickets/{ticket_id}
```

### Ticket Intelligence

```text
POST /api/ticket-intelligence/route
```

Request:

```json
{
  "subject": "Invoice payment issue",
  "body": "I was charged twice for my subscription."
}
```

Response:

```json
{
  "predicted_queue": "billing_and_payments",
  "confidence": 0.91,
  "decision": "auto_route",
  "threshold": 0.8,
  "probabilities": {
    "billing_and_payments": 0.91,
    "customer_general": 0.01,
    "human_resources": 0.01,
    "returns_and_exchanges": 0.01,
    "sales_and_pre_sales": 0.01,
    "service_outages_and_maintenance": 0.01,
    "technical_product_support": 0.04
  },
  "model_id": "shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1",
  "latency_ms": 42,
  "timestamp": "2026-06-13T12:00:00Z"
}
```

The default routing threshold is `0.80`:

- Confidence at or above the threshold: `auto_route`
- Confidence below the threshold: `human_review`

Successful responses are persisted automatically. Model load or inference
failures return a controlled HTTP `503` response and are not recorded as
successful predictions.

### Dashboard

```text
GET /api/dashboard/summary
GET /api/dashboard/recent-predictions?limit=20
```

Dashboard data includes:

- Total tickets.
- Total routing predictions.
- Queue distribution.
- Auto-route and human-review counts.
- Average confidence.
- Recent persisted predictions.

## Ticket Intelligence Model

The canonical runtime model is hosted on Hugging Face:

```text
shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1
```

Supported queues:

- `billing_and_payments`
- `customer_general`
- `human_resources`
- `returns_and_exchanges`
- `sales_and_pre_sales`
- `service_outages_and_maintenance`
- `technical_product_support`

The backend:

- Loads the tokenizer and classifier from Hugging Face.
- Caches the loaded model and tokenizer.
- Uses CUDA when available and CPU otherwise.
- Applies softmax and validates the seven-label contract.
- Records model ID, latency, threshold, probabilities, and timestamp.
- Converts model-unavailable failures into HTTP `503`.

The model runs in the backend, not in Streamlit. The repository does not commit
model weights or require the frontend to load ML dependencies.

## Database

SQLAlchemy is the canonical ORM. PostgreSQL is supported through
`DATABASE_URL`; SQLite is used when that variable is absent.

### Current Tables

#### `tickets`

Stores canonical case records, status, source/channel metadata, optional
ground-truth category and priority, tags, language, and timestamps.

#### `ticket_routing_predictions`

Stores:

- Optional linked ticket ID.
- Subject and body.
- Predicted queue.
- Confidence and threshold.
- Routing decision.
- Full probabilities JSON.
- Model ID.
- Latency.
- Creation timestamp.

### Migrations

Alembic owns schema changes:

```bash
alembic upgrade head
```

`python -m backend.app.db.init_db` remains a development fallback, but Alembic
is the canonical migration path.

## Streamlit Frontend

The canonical frontend is:

```text
frontend/streamlit_app.py
```

It communicates with FastAPI only over HTTP and never imports the research
vertical or model code.

Current tabs:

1. **Dashboard**
   - Total tickets and predictions.
   - Auto-route and human-review counts.
   - Average confidence and review rate.
   - Queue and decision charts.
   - Recent predictions preview.

2. **Ticket Routing Demo**
   - Subject and body form.
   - Queue, confidence, decision, threshold, model ID, latency, and timestamp.
   - Sorted probability chart.
   - Automatic refresh after successful persistence.

3. **Recent Predictions**
   - Newest 20 stored routing records.
   - Queue, confidence, decision, threshold, latency, timestamp, and subject.

4. **Ticket Inbox**
   - Database-backed ticket records.
   - Status, category, priority, source/channel, message, and timestamps.

5. **Model Analytics**
   - Queue and decision mix.
   - Auto-route and review rates.
   - Confidence chart and latency summary from recent predictions.
   - Clearly labeled as bounded demo analytics.

### Frontend Error Handling

`frontend/api_client.py` provides structured handling for:

- Backend connection failures and timeouts.
- HTTP `503` model unavailability or cold starts.
- Request validation errors.
- Unexpected HTTP responses.
- Non-JSON and malformed successful responses.

The UI also handles empty ticket/prediction states and validates routing input
before making an API call.

## Local Setup

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Example:

```text
DATABASE_URL=sqlite:///./opspilot_dev.db
APP_ENV=development
OPSPILOT_ROUTING_THRESHOLD=0.80
```

For PostgreSQL:

```text
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
```

### 3. Apply migrations

```bash
alembic upgrade head
```

### 4. Start FastAPI

```bash
uvicorn backend.app.main:app --reload
```

Backend:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

### 5. Start Streamlit

In another terminal:

```bash
export OPSPILOT_API_URL=http://localhost:8000
streamlit run frontend/streamlit_app.py
```

Frontend:

```text
http://localhost:8501
```

### 6. Test the product loop

1. Open the Ticket Routing Demo tab.
2. Enter a subject and body.
3. Submit the ticket.
4. Review the routing decision and probabilities.
5. Confirm Dashboard and Recent Predictions update.

If local Torch or Transformers dependencies are unavailable, the backend
returns HTTP `503` and the frontend displays a controlled cold-start/model
unavailable message.

## Testing

Run the complete suite:

```bash
pytest -q
```

Current coverage includes:

- Health endpoint.
- Ticket CRUD.
- Routing API contract and threshold behavior.
- Prediction persistence.
- Dashboard aggregation and recent records.
- Controlled HTTP `503` behavior.
- Frontend API-client paths and error handling.

Compile frontend files:

```bash
python -m compileall -q frontend
```

## Deployment

The repository includes:

- A backend `Dockerfile`.
- CPU PyTorch and Transformers dependencies in `requirements-railway.txt`.
- Railway start and health-check configuration.
- PostgreSQL URL normalization.
- Hugging Face cache configuration under `/tmp`.

Deployment requirements:

1. Configure `DATABASE_URL`.
2. Run `alembic upgrade head`.
3. Ensure the backend can download the Hugging Face model.
4. Confirm CPU memory, cold-start time, and request timeout behavior.
5. Set `OPSPILOT_API_URL` in the Streamlit environment to the deployed backend.

The current Docker/Railway configuration deploys the FastAPI backend. A
separate frontend deployment configuration remains future work.

## Research and Training Lab

The Ticket Intelligence research code remains under:

```text
verticals/ticket-intelligence/ml
```

It includes:

- Dataset audit and fixed split generation.
- TF-IDF and logistic-regression baselines.
- Feature inspection and error analysis.
- Confidence-threshold analysis.
- TensorFlow experiments.
- Transformer and ModernBERT training scripts.
- Model comparison and responsible-AI documentation.

The public source dataset is:

```text
Tobi-Bueck/customer-support-tickets
```

The research lab is intentionally separate from the product runtime. Generated
datasets, local model artifacts, outputs, and model weights are gitignored.

`verticals/ticket-intelligence/app` is a legacy standalone prototype and is not
the canonical product frontend or backend.

## Current Limitations

- Routing submissions are logged as predictions but do not automatically create
  canonical ticket records.
- Recent-prediction and confidence analytics currently use a bounded result set.
- There is no curated 500–2,000 record demo seed yet.
- Human-review decisions, assignments, overrides, and outcomes are not modeled.
- Authentication and role-based permissions are not implemented.
- Policy retrieval, recommendations, drafting, and AI evaluation are planned.
- Cloud model performance and concurrency require deployment validation.
- The frontend deployment is not yet included in the Docker/Railway setup.

## Roadmap

### Stage 1: Product Backbone

Implemented:

- Canonical FastAPI service.
- Ticket CRUD.
- Final routing contract.
- Prediction logging.
- Dashboard APIs.
- Alembic migrations.

### Stage 2: Canonical Streamlit MVP

Implemented:

- HTTP-only API client.
- Dashboard.
- Ticket Routing Demo.
- Recent Predictions.
- Ticket Inbox.
- Model Analytics.
- Offline, validation, and model-unavailable states.

### Stage 3: Curated Demo Data

Planned:

- Repeatable seed process.
- Approximately 500–2,000 representative records.
- No full training-dataset import.

### Stage 4: Human Review Workflow

Planned:

- Review assignments.
- Approvals, rejections, and escalations.
- Queue and priority overrides.
- Reviewer notes and final outcomes.

### Stage 5: Cloud Hardening

Planned:

- Migration automation.
- Readiness and model-warmup checks.
- Structured logs and monitoring.
- Cold-start, memory, latency, and concurrency validation.
- Frontend deployment configuration.

### Stage 6: Additional AI Modules

Planned:

- Policy retrieval.
- Grounded next-best-action recommendations.
- Response drafting.
- Evaluation traces.
- Controlled tool and agent orchestration.

## Responsible AI Principles

- AI output is decision support, not unquestioned truth.
- Low-confidence outputs require human review.
- Sensitive actions should require explicit approval.
- Model IDs, thresholds, probabilities, and timestamps should remain auditable.
- Training data and production customer data must remain clearly separated.
- Production use requires retention, redaction, privacy, bias, and monitoring
  policies.
- Future generated recommendations should preserve supporting evidence.

## Documentation

- [Product Architecture](docs/PRODUCT_ARCHITECTURE.md)
- [Backend Guide](backend/README.md)
- [Frontend Guide](frontend/README.md)
- [Architecture Notes](docs/architecture.md)
- [Data Model](docs/data_model.md)
- [Evaluation Plan](docs/evaluation_plan.md)
- [Analytics Plan](docs/analytics_plan.md)
- [Ticket Intelligence Vertical](verticals/ticket-intelligence/README.md)

## Positioning

OpsPilot demonstrates the design of an applied AI product that combines:

- Product and workflow architecture.
- Backend and database engineering.
- Specialized model integration.
- Human-in-the-loop controls.
- Auditability and model observability.
- Operator-facing analytics.
- A clean boundary between research and deployed product code.

The project should be presented as an evolving AI operations platform, not as a
toy chatbot, a single notebook, or an autonomous decision-maker.
