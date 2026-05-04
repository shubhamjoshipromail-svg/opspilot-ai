# Architecture

## Planned System Components

OpsPilot AI is planned as a modular application with clear service boundaries:

- **Frontend:** Streamlit application for reviewing cases, AI outputs, policy evidence, human decisions, and analytics.
- **Backend API:** FastAPI application for case intake, persistence, workflow orchestration, and analytics access.
- **Database:** Postgres for cases, model outputs, review decisions, evaluation traces, and metrics.
- **Model layer:** Swappable services for triage, risk scoring, retrieval, recommendation, drafting, and evaluation.
- **Policy corpus:** Markdown policy and SOP documents used for retrieval and grounding.
- **Evaluation layer:** Offline and online checks for quality, groundedness, compliance, and business relevance.
- **Analytics layer:** Operational, model, and ROI reporting.

## Planned Data Flow

1. Ticket is submitted through API or imported from a sample dataset.
2. Backend validates and stores the ticket.
3. Triage model predicts category.
4. Risk model predicts escalation risk and priority.
5. Retriever searches policies and returns evidence.
6. Recommender proposes a next best action.
7. Drafting component proposes a response.
8. Evaluator scores AI outputs.
9. Human reviewer approves, edits, rejects, or escalates.
10. System stores all outputs, versions, evidence, and feedback.
11. Analytics layer aggregates operational, model, and ROI metrics.

## Service Boundaries

### API Boundary

FastAPI should expose workflow endpoints without embedding model-specific logic directly inside routes.

### Persistence Boundary

Database code should manage storage and retrieval without deciding business logic.

### Model Boundary

Each model family should be accessed through a clear interface so future algorithms can be swapped without rewriting the workflow.

### Evaluation Boundary

Evaluators should produce structured scores, reasons, and trace metadata. They should not silently block or approve actions without review rules.

### Human Review Boundary

Human decisions should be explicit records with reviewer identity, decision type, edited output, notes, and timestamps.

## Production-Grade Design Principles

- Modular model interfaces.
- Version every model, prompt, retrieval method, and evaluator.
- Store evidence used for recommendations.
- Preserve audit trails.
- Keep humans in the loop for risky actions.
- Prefer baselines before complex models.
- Separate workflow orchestration from model implementations.
- Measure quality and business impact.
- Avoid provider lock-in.
- Treat AI output as a recommendation, not a source of truth.

## Future Deployment Architecture

A future deployment may include:

- Streamlit frontend service.
- FastAPI backend service.
- Railway Postgres database.
- Environment-managed API keys.
- Scheduled evaluation jobs.
- CI checks with pytest.
- Optional background worker for long-running model calls.
- Optional vector database or Postgres vector extension for retrieval.

This scaffold does not implement deployment automation yet.
