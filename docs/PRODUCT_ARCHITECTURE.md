# OpsPilot Product Architecture

## 1. Product Definition

OpsPilot is a modular AI operations platform designed to connect company
databases, internal tools, specialized machine learning models, retrieval
systems, and agentic workflows into a controlled decision-support system.

The goal is not to build a single chatbot. The goal is to build an AI
operations backbone where different AI modules can be plugged in depending on
the business workflow.

OpsPilot is intended for operational processes that combine unstructured input,
business rules, internal data, policy constraints, model predictions, and human
judgment. Examples include service-ticket routing, claims review, billing
operations, internal service desks, fulfillment exceptions, compliance review,
and other queue-based workflows.

The product architecture emphasizes:

- Stable API and database boundaries.
- Replaceable AI modules rather than one hard-coded model.
- Explicit confidence and routing decisions.
- Human review for uncertain or sensitive actions.
- Persistent model outputs and audit history.
- Database-backed operational and model analytics.
- A staged path from deterministic workflows to more capable AI orchestration.

### Status Vocabulary

This document uses three status labels:

- **Implemented:** Present in the canonical product code and covered by the
  current application or tests.
- **Partial:** A meaningful foundation exists, but the product workflow is not
  complete.
- **Planned:** Documented as a future product capability but not yet implemented.

## 2. MVP Scope

The current MVP focuses on service-ticket intelligence. It establishes the
smallest useful product backbone for receiving tickets, running a specialized
routing model, recording its output, and exposing database-backed analytics.

The MVP workflow is:

1. A ticket is created or ticket text is submitted for routing.
2. The Ticket Intelligence service combines the subject and body.
3. A hosted ModernBERT classifier predicts the operational queue.
4. The service returns a confidence score and all queue probabilities.
5. A configured confidence threshold selects either `auto_route` or
   `human_review`.
6. Successful predictions are persisted for audit and analytics.
7. Dashboard APIs summarize ticket volume, queue distribution, routing
   decisions, confidence, and recent predictions.

MVP capabilities:

- Ticket CRUD: **Implemented**
- Ticket routing API contract: **Implemented**
- Confidence-based routing decision: **Implemented**
- Hugging Face model integration: **Implemented**, with cloud runtime and model
  availability still dependent on deployment configuration
- Prediction persistence: **Implemented**
- Database migrations: **Implemented**
- Dashboard read APIs: **Implemented**
- Main Streamlit product interface: **Partial**, currently a placeholder
- Curated demo database seed: **Planned**

## 3. Core Architecture

OpsPilot follows a layered architecture:

```text
Operator or internal system
            |
            v
Canonical Streamlit frontend / API client
            |
            v
FastAPI routes and request validation
            |
            v
Workflow and domain services
      |           |           |
      v           v           v
Ticket store   AI modules   Future tool connectors
      |           |
      v           v
SQLAlchemy     Hosted models / retrieval / agents
      |
      v
PostgreSQL or local SQLite
```

### API Layer

FastAPI defines stable product contracts for ticket management, model
invocation, dashboard reads, and future review workflows. Routes should remain
thin: they validate requests, invoke services, map domain errors to HTTP
responses, and serialize results.

### Service Layer

Services contain model inference, prediction persistence, aggregation, and
future workflow orchestration. AI-provider or model-specific details should not
be embedded directly in frontend code or database models.

### Persistence Layer

SQLAlchemy models provide the canonical application state. Alembic owns schema
evolution. PostgreSQL is the intended hosted database, while SQLite remains a
development fallback.

### AI Module Layer

Each AI capability should expose a narrow input/output contract. A workflow can
then replace a model, retrieval method, or provider without rewriting the whole
application.

### Human Control Layer

Confidence thresholds, policy rules, review queues, overrides, and audit logs
form the control system around AI output. AI results are decision support, not
unrecorded autonomous actions.

## 4. Canonical Repo Map

```text
Ops-copilot/
├── backend/
│   └── app/                         Canonical FastAPI product backend
│       ├── main.py                  Application and router registration
│       ├── config.py                Environment configuration
│       ├── database.py              SQLAlchemy engine and sessions
│       ├── db/                      Local initialization utilities
│       ├── models/                  Canonical ORM entities
│       ├── routes/                  HTTP API routes
│       ├── schemas/                 Pydantic request/response contracts
│       └── services/                Domain, model, persistence, analytics logic
├── frontend/
│   └── streamlit_app.py             Canonical MVP frontend
├── migrations/                      Canonical Alembic migrations
├── verticals/
│   └── ticket-intelligence/
│       ├── ml/                      Research and training lab
│       ├── notebooks/               Experiments and walkthroughs
│       ├── docs/                    Model research documentation
│       └── app/                     Legacy standalone prototype
├── data/
│   ├── policies/                    Mock policy and future RAG seed material
│   ├── raw/                         Local, gitignored source datasets
│   └── processed/                   Local, gitignored processed datasets
├── scripts/                         Dataset and development utilities
├── tests/                           Canonical backend tests
└── docs/                            Product and architecture documentation
```

### Canonical Boundaries

- `backend/app` is the canonical backend. New product APIs and services belong
  there.
- `frontend/streamlit_app.py` is the canonical MVP frontend. It should consume
  backend APIs rather than import model training or inference modules directly.
- `verticals/ticket-intelligence/ml` is a research and training lab. It may
  produce models and evaluation evidence, but it is not the product runtime.
- `verticals/ticket-intelligence/app` is a legacy standalone prototype unless
  explicitly revived. It should not receive routine product development.
- `docs` contains canonical product, architecture, evaluation, and operational
  documentation.
- `data/policies` contains mock material for future retrieval experiments. It is
  not an authoritative policy source.

Legacy files are retained for research history and reference. Their presence
does not make them part of the canonical runtime.

## 5. Current Implemented Components

### Backend

- FastAPI application and health endpoint: **Implemented**
- Ticket create, list, read, update, and delete APIs: **Implemented**
- Ticket Intelligence routing endpoint: **Implemented**
- Dashboard summary endpoint: **Implemented**
- Recent routing predictions endpoint: **Implemented**
- Controlled model-unavailable response: **Implemented**
- Authentication and authorization: **Planned**

### Database

- SQLAlchemy engine and session management: **Implemented**
- PostgreSQL through `DATABASE_URL`: **Implemented**
- SQLite development fallback: **Implemented**
- `tickets` table: **Implemented**
- `ticket_routing_predictions` table: **Implemented**
- Alembic migration baseline: **Implemented**
- Human review, event, user, permission, and evaluation tables: **Planned**

### Frontend

- Canonical Streamlit entry point: **Implemented**
- Product dashboard, routing demo, inbox, and analytics views: **Planned**
- API client and user-facing error/loading states: **Planned**

### AI and Research

- ModernBERT ticket-routing runtime service: **Implemented**
- Model/tokenizer caching: **Implemented**
- TF-IDF, TensorFlow, and transformer research pipelines: **Implemented as
  research tooling**
- Model training inside the product backend: **Out of scope**
- Retrieval, recommendations, drafting, and AI evaluation: **Planned**

## 6. Ticket Intelligence Module

Ticket Intelligence is the first production-oriented AI module plugged into
OpsPilot.

### Runtime Contract

Canonical endpoint:

```text
POST /api/ticket-intelligence/route
```

Input:

```json
{
  "subject": "Invoice payment issue",
  "body": "I was charged twice for my subscription."
}
```

The response includes:

- Predicted queue.
- Top confidence.
- `auto_route` or `human_review` decision.
- Decision threshold.
- Probability for each supported queue.
- Model identifier.
- Inference latency.
- UTC timestamp.

### Supported Queues

- `billing_and_payments`
- `customer_general`
- `human_resources`
- `returns_and_exchanges`
- `sales_and_pre_sales`
- `service_outages_and_maintenance`
- `technical_product_support`

### Model

The canonical runtime points to:

```text
shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1
```

The backend is responsible for loading and caching the Hugging Face tokenizer
and classifier. Model artifacts are not committed to this repository. Hosted
environments may install CPU PyTorch and Transformers and download the model at
runtime.

### Decision Policy

The default auto-route threshold is `0.80`:

- Confidence greater than or equal to `0.80`: `auto_route`
- Confidence below `0.80`: `human_review`

The threshold is configurable and should eventually be governed by evaluated
coverage, accuracy, business risk, and queue-specific policies.

### Persistence

Successful routing calls are stored in `ticket_routing_predictions`. Model
failures return a controlled service-unavailable response and are not recorded
as successful predictions.

The current API supports storing subject and body. The database model also
allows an optional link to a canonical ticket record, providing a path toward a
combined ticket-create-and-route workflow.

## 7. Future Plug-and-Play AI Modules

Future modules should follow the same pattern as Ticket Intelligence: a stable
contract, a service implementation, persistent outputs, explicit model or
provider versions, and measurable behavior.

### Priority and Escalation Risk

**Planned**

Estimate operational urgency and escalation risk using ticket text plus
business metadata such as account tier, SLA, impact, recurrence, and channel.

### Policy Retrieval

**Planned**

Retrieve applicable policy or SOP evidence. The first version should use a
simple, inspectable retrieval method before introducing embeddings or a vector
database.

### Next-Best-Action Recommendation

**Planned**

Combine ticket facts, model outputs, workflow state, and retrieved evidence to
suggest an operational action. Recommendations should include rationale,
confidence, and review requirements.

### Response Drafting

**Planned**

Draft customer-facing or internal responses grounded in approved evidence.
Drafts should remain editable and require review for sensitive cases.

### AI Evaluation

**Planned**

Score model or generated outputs for correctness, groundedness, policy
compliance, hallucination risk, escalation handling, and usefulness.

### Forecasting and Anomaly Detection

**Planned**

Analyze queue volume, backlog, SLA risk, recurring incidents, and unusual
operational patterns.

### Internal Tool Connectors

**Planned**

Connect to CRM, help desk, billing, order, identity, and data warehouse systems
through controlled adapters. Connectors should expose only the data and actions
required for a workflow.

## 8. Agent and Orchestration Vision

Agentic workflows are a future coordination layer, not the MVP foundation.

An OpsPilot orchestrator may eventually:

1. Receive a case and inspect its workflow state.
2. Invoke classification and risk modules.
3. Query approved internal systems for context.
4. Retrieve relevant policies.
5. Propose an action and supporting rationale.
6. Request human review when required.
7. Execute an approved tool action.
8. Record every model call, tool call, decision, and outcome.

The orchestrator should operate as a constrained state machine or workflow
service rather than an unrestricted autonomous agent. Each tool must have:

- A defined input and output contract.
- Explicit permissions.
- Read-only or write capability classification.
- Timeouts and failure handling.
- Idempotency where applicable.
- Audit logging.
- Human approval rules for consequential actions.

Possible future technologies include background workers, durable workflow
engines, tool-calling LLMs, MCP-compatible connectors, and provider-specific
agent SDKs. None of these are currently part of the canonical MVP.

## 9. Human Review and Audit Layer

Human review is a core product boundary, not a fallback UI detail.

The current Ticket Intelligence module produces a binary review decision:

- `auto_route`
- `human_review`

The future review layer should support:

- Review queue assignment.
- Approval, rejection, and escalation.
- Queue, priority, and risk overrides.
- Edited recommendations or response drafts.
- Reviewer notes and override reasons.
- Reviewer identity and timestamps.
- Final disposition and business outcome.

Every meaningful AI-assisted decision should preserve:

- Original input.
- Model or provider identifier.
- Model output and confidence.
- Thresholds and business rules.
- Supporting evidence.
- Human action and override.
- Final workflow outcome.

The review layer is **planned**. Prediction persistence is the first implemented
audit foundation.

## 10. Database and Logging Layer

The database is the source of truth for product workflow state. Dashboard
metrics should be computed from stored records rather than hardcoded examples.

### Current Tables

#### `tickets`

**Implemented**

Stores the canonical case record, source metadata, optional ground-truth
labels, status, and timestamps.

#### `ticket_routing_predictions`

**Implemented**

Stores:

- Optional `ticket_id`.
- Subject and body.
- Predicted queue.
- Confidence.
- Threshold.
- Routing decision.
- Full probability distribution.
- Model identifier.
- Latency.
- Creation timestamp.

### Planned Tables

- `ticket_events`
- `human_reviews`
- `risk_scores`
- `retrieval_results`
- `recommendations`
- `draft_responses`
- `evaluation_traces`
- `model_versions`
- `users`, `roles`, and permissions

### Logging Principles

- Store structured workflow records in the database.
- Use application logs for operational diagnostics, not as the sole audit
  trail.
- Avoid storing secrets in inputs, outputs, or logs.
- Define retention and redaction policies before using production customer
  data.
- Record model and prompt versions for reproducibility.
- Separate successful predictions from model-service failures.

## 11. Dashboard Layer

The dashboard layer turns persisted workflow records into operational and model
visibility.

### Current Dashboard APIs

```text
GET /api/dashboard/summary
GET /api/dashboard/recent-predictions
```

The summary endpoint provides:

- Total tickets.
- Total routing predictions.
- Predicted queue distribution.
- Auto-route versus human-review counts.
- Average confidence.

The recent-predictions endpoint provides a bounded, newest-first list of routing
records.

### Planned Streamlit Views

#### Dashboard

Overall ticket and routing metrics, recent activity, and review workload.

#### Ticket Routing Demo

Subject/body input, routing result, confidence, decision, and probability
breakdown.

#### Ticket Inbox

Database-backed ticket list, status, assignment, and ticket detail navigation.

#### Model Analytics

Confidence distribution, queue mix, auto-route coverage, human-review rate,
latency, errors, and future quality metrics.

The Streamlit application should call FastAPI. It should not directly load the
Hugging Face model or import research code.

## 12. MVP Build Order

### Stage 1: Product Backbone

**Implemented**

- Establish `backend/app` as canonical.
- Finalize Ticket Intelligence API contract.
- Add prediction persistence.
- Add Alembic migrations.
- Add dashboard read APIs.
- Add database-backed tests.

### Stage 2: Canonical Streamlit MVP

**Planned**

- Build Dashboard, Ticket Routing Demo, Ticket Inbox, and Model Analytics tabs.
- Add a small API client layer.
- Handle validation, loading, empty, and service-unavailable states.

### Stage 3: Curated Demo Data

**Planned**

- Build a repeatable seed process.
- Load approximately 500–2,000 representative tickets.
- Avoid loading the full training dataset.
- Preserve clear separation between demo labels and production outcomes.

### Stage 4: Review Workflow

**Planned**

- Add ticket events and human review records.
- Add assignments, overrides, and final decisions.
- Connect human-review counts to an actionable queue.

### Stage 5: Cloud Hardening

**Planned**

- Run migrations during deployment.
- Configure PostgreSQL and Hugging Face access.
- Confirm CPU memory, model download, cold-start, and concurrency behavior.
- Add structured logging, readiness checks, and deployment monitoring.

### Stage 6: Additional AI Modules

**Planned**

- Add policy retrieval.
- Add grounded recommendations.
- Add response drafting.
- Add evaluation traces.
- Introduce orchestration only after these individual modules have stable
  contracts.

## 13. Explicitly Out of Scope for the MVP

The MVP does not attempt to provide:

- A general-purpose chatbot.
- Fully autonomous case resolution.
- Unrestricted multi-agent collaboration.
- Automatic refunds, payments, account changes, or other consequential actions.
- Production enterprise authentication and role-based access control.
- A large vector database or complex RAG stack.
- LLM-generated recommendations or responses.
- A complete AI evaluation platform.
- Multilingual ticket routing.
- Online model training or fine-tuning.
- Committed model binaries or large datasets.
- Full help-desk replacement functionality.
- Production claims about model accuracy, fairness, or business ROI without
  measured evidence.

These boundaries keep the MVP focused on a reliable, inspectable workflow
backbone.

## 14. Glossary

### AI Module

A bounded capability, such as routing, retrieval, risk scoring, recommendation,
or evaluation, exposed through a stable service contract.

### Canonical Runtime

The product code path intended for deployment and continued development. For
OpsPilot, this is `backend/app` plus `frontend/streamlit_app.py`.

### Decision Support

AI output that informs an operator or controlled workflow without silently
replacing accountable human decisions.

### Human Review

A workflow state requiring a person to inspect, approve, reject, edit, or
override an AI-assisted decision.

### Model Confidence

The classifier's probability for its selected queue. It is a model signal, not
an unconditional measure of correctness.

### Routing Threshold

The minimum confidence required for an `auto_route` decision. Predictions below
the threshold are sent to `human_review`.

### Prediction Persistence

Recording model input, output, metadata, timing, and decision in the database
for analytics and auditability.

### Retrieval-Augmented Generation

A method that retrieves approved evidence before generating an answer or
recommendation. RAG is planned but not implemented in the current MVP.

### Agentic Workflow

A coordinated sequence in which an AI system selects and invokes approved
modules or tools. In OpsPilot, future agentic workflows must remain constrained,
permissioned, observable, and reviewable.

### Tool Connector

An adapter that allows a workflow to read from or write to an internal system
through a narrow, controlled interface.

### Research Vertical

Code used for training, experiments, evaluation, and model comparison. Research
verticals may inform the product but are not automatically part of the deployed
runtime.

### Audit Trail

The stored history of inputs, model outputs, rules, evidence, human actions, and
workflow outcomes needed to reconstruct a decision.
