# Product Spec

## Product Vision

OpsPilot AI is planned as an AI operations copilot for teams that process messy, policy-sensitive cases. The product should help operators understand incoming requests, prioritize the riskiest cases, ground recommendations in policy evidence, evaluate AI outputs, and keep humans in control of important decisions.

The long-term goal is not to replace operations teams. The goal is to reduce repetitive analysis, improve decision consistency, preserve auditability, and make AI-assisted workflows measurable.

## User Personas

### Operations Agent

Needs to quickly understand a case, find the right policy, decide what to do next, and send a clear response.

### Operations Manager

Needs visibility into backlog, escalations, quality, agent workload, resolution time, and business impact.

### AI Review Specialist

Needs to inspect recommendations, evaluate model behavior, approve or reject AI outputs, and capture feedback.

### Product/Data Analyst

Needs to measure workflow quality, model performance, customer impact, and ROI.

### AI Engineer

Needs modular interfaces, evaluation datasets, model versioning, feedback loops, and reliable traces.

## Core User Workflow

1. A new ticket enters the system.
2. The system stores the ticket and normalizes key fields.
3. A triage model predicts category and confidence.
4. A risk model estimates priority and escalation risk.
5. A retriever finds relevant policy or SOP evidence.
6. A recommender suggests a next best action.
7. A drafting component creates a proposed response.
8. An evaluator checks quality, groundedness, policy compliance, and hallucination risk.
9. A human reviewer approves, edits, rejects, or escalates.
10. Final decisions and feedback are saved for analytics and future improvement.

## MVP Scope

The MVP should eventually include:

- Ticket CRUD in FastAPI.
- Postgres storage.
- Streamlit case inbox.
- Case detail workspace.
- Baseline triage model.
- Rule-based risk scoring.
- Simple policy retrieval.
- Rule-based recommendation prototype.
- Human review and override capture.
- Basic evaluation trace records.
- Operational analytics dashboard.

## Non-Goals

The first implementation phases should not attempt to build:

- Fully autonomous case resolution.
- Complex multi-agent workflows.
- Enterprise authentication.
- Production-grade queueing infrastructure.
- Deep learning model training at the start.
- A large-scale vector search system before baseline retrieval exists.
- A generic chatbot interface with no operations workflow.

## Future Product Pages

Planned Streamlit pages:

- Case Inbox.
- Case Workspace.
- Policy Evidence.
- Recommendation Review.
- Human Approval Queue.
- Evaluation Traces.
- Model Performance.
- Operations Analytics.
- ROI Dashboard.
- Experiment Comparison.
