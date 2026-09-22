# Repository Audit

Last reviewed: 2026-07-12

## Git And GitHub State

- Local repository: `/Users/shubhamjoshi/Documents/Ops-copilot`
- GitHub remote: `git@github.com:shubhamjoshipromail-svg/opspilot-ai.git`
- GitHub repository: `shubhamjoshipromail-svg/opspilot-ai`
- GitHub visibility: public
- GitHub default branch: `main`
- Remote branches observed: `main`, `codex/ticket-intelligence`
- Current local branch during audit: `codex/ticket-intelligence`
- Current local branch status at audit start: ahead of `origin/codex/ticket-intelligence` by 2 commits
- Recent GitHub PRs returned by the connector: none
- `gh` CLI availability in this environment: unavailable

The remote default branch is `main`, but the most complete product work is on
`codex/ticket-intelligence`. Local `main` is not aligned with `origin/main`, so
use `origin/main` rather than local `main` when comparing against the default
branch.

## Uncommitted Work Classification

At audit start, the working tree contained:

| Path | Classification | Action |
| --- | --- | --- |
| `README.md` | Intentional documentation/product positioning work | Preserve and test with repo changes |
| `frontend/README.md` | Intentional frontend documentation work | Preserve |
| `frontend/streamlit_app.py` | Important unfinished product work; canonical Streamlit console | Preserve and test |
| `frontend/api_client.py` | Important product work; frontend API boundary and error handling | Preserve and test |
| `tests/test_frontend_api_client.py` | Important regression tests | Preserve and test |
| `verticals/ticket-intelligence/ticket_splits.zip` | Generated dataset split archive | Preserve locally, ignore in Git |
| `data/raw/customer_support_tickets_en.csv` | Regenerated local dataset | Keep ignored |
| `data/processed/tickets_en_normalized.csv` | Regenerated local dataset | Keep ignored |
| `verticals/ticket-intelligence/ml/ticket_intelligence/artifacts/*` | Local model artifacts | Keep ignored |
| `verticals/ticket-intelligence/ml/ticket_intelligence/outputs/*` | Local evaluation outputs | Keep ignored |
| `opspilot_dev.db` | Local SQLite database | Keep ignored |
| `__pycache__`, `.pytest_cache`, `.DS_Store` | Local/generated files | Keep ignored |

No staged files were present at audit start.

## Current Application Architecture

The repository currently has a conservative but workable structure:

```text
backend/app/                         Canonical FastAPI product backend
frontend/streamlit_app.py            Canonical Streamlit product console
frontend/api_client.py               HTTP client boundary used by Streamlit
migrations/                          Alembic migrations
verticals/ticket-intelligence/ml/    Ticket Intelligence research/training lab
verticals/ticket-intelligence/app/   Legacy standalone prototype
data/policies/                       Mock policy text for future retrieval
scripts/                             Dataset and database utilities
tests/                               Backend and frontend API-client tests
docs/                                Product, architecture, analytics, and ML docs
```

The safest near-term organization decision is to keep these paths stable. Moving
the project into an `apps/` or top-level `ml/` layout would require import and
deployment updates with limited immediate product value.

## Fully Working Or Verified

- FastAPI app registration and `/health`.
- Ticket CRUD routes.
- SQLAlchemy database setup with SQLite fallback and Postgres via `DATABASE_URL`.
- Alembic baseline migration for `tickets` and `ticket_routing_predictions`.
- Ticket Intelligence routing endpoint contract.
- Hosted Hugging Face ModernBERT runtime service boundary.
- Confidence threshold behavior: `auto_route` at or above `0.80`, otherwise `human_review`.
- Successful prediction persistence.
- Dashboard summary and recent-predictions APIs.
- Streamlit API client error handling.
- Streamlit dashboard/routing/inbox/model-analytics console.
- Test suite: `18 passed` on 2026-07-12.

## Partial

- Streamlit frontend is a useful MVP console but does not yet provide full case
  review, override capture, policy evidence, or approval workflows.
- Dashboard analytics cover routing volume, decision mix, queue distribution,
  confidence, and recent latency, but not full operational ROI.
- Ticket records and routing predictions can be linked by schema, but the current
  routing demo logs predictions separately and does not create or update ticket
  records.
- Local ML lab has train/evaluate scripts and local artifacts, but the product
  runtime currently uses the hosted seven-label model.

## Experimental Or Local-Only

- TF-IDF and Logistic Regression baseline artifacts in
  `verticals/ticket-intelligence/ml/ticket_intelligence/artifacts/`.
- Local evaluation outputs in
  `verticals/ticket-intelligence/ml/ticket_intelligence/outputs/`.
- TensorFlow and transformer training scripts.
- Notebooks under `verticals/ticket-intelligence/notebooks/`.
- Legacy standalone API/Streamlit prototype under
  `verticals/ticket-intelligence/app/`.

## Planned But Not Implemented

- Human review records and overrides.
- Policy retrieval/RAG.
- Policy-grounded recommendations.
- Response drafting.
- Evaluation traces.
- Authentication and authorization.
- ROI dashboard with persisted business assumptions.
- Drift monitoring and retraining alerts.

## Duplicate Or Conflicting Implementations

- The canonical runtime backend is `backend/app`.
- `verticals/ticket-intelligence/app` is retained as a legacy prototype and
  should not receive routine product work unless intentionally revived.
- The local ML lab documents a 10-label baseline taxonomy, while the product
  routing API serves a seven-label hosted model. This is not necessarily wrong,
  but it should be documented clearly whenever metrics or labels are discussed.

## Safety Notes

- Do not delete local ignored datasets, model artifacts, or evaluation outputs.
  They are intentionally not tracked because of size, regeneration, or artifact
  churn.
- Do not move model-critical files unless every import/path and command is
  updated and verified.
- Do not push substantial changes directly to `main`; use the
  `codex/ticket-intelligence` branch and a PR flow.
