# Roadmap

## Phase 0: Repo Scaffold And Documentation

Status: current phase.

Goals:

- Create repository structure.
- Add README and planning docs.
- Add placeholder policies.
- Add minimal backend and frontend placeholders.
- Avoid implementing business logic too early.

## Phase 1: Postgres + FastAPI Ticket CRUD

Goals:

- Define initial ticket schema.
- Add SQLAlchemy setup.
- Add migrations.
- Implement ticket create/read/update/list APIs.
- Add basic tests.

## Phase 2: Streamlit Case Inbox/Workspace

Goals:

- Display ticket inbox.
- Show ticket detail page.
- Add human review controls.
- Connect to backend APIs.

## Phase 3: Baseline ML Models

Goals:

- Build simple triage baseline.
- Build rule-based risk scoring.
- Store model outputs.
- Add model interface contracts.

## Phase 4: Policy Retrieval/RAG

Goals:

- Load policy documents.
- Implement keyword retrieval baseline.
- Store retrieval evidence.
- Add retrieval evaluation.
- Later add embeddings or RAG.

## Phase 5: LLM Recommendation And Response Drafting

Goals:

- Add provider-agnostic LLM interface.
- Generate recommended actions.
- Draft responses grounded in retrieved evidence.
- Store prompt and model versions.

## Phase 6: AI Judge And Evaluation Traces

Goals:

- Define evaluation rubrics.
- Score groundedness, quality, hallucination risk, policy compliance, and escalation correctness.
- Store evaluation traces.
- Compare AI judge output to human feedback.

## Phase 7: Analytics And ROI Dashboard

Goals:

- Build operations dashboard.
- Build model quality dashboard.
- Build ROI dashboard.
- Add assumptions and sensitivity analysis.

## Phase 8: Advanced Model Experiments

Goals:

- Test embeddings plus classifier.
- Test transformer models.
- Compare retrieval strategies.
- Explore contextual bandit recommendations.
- Explore pairwise preference or reward modeling.

## Phase 9: Deployment And Portfolio Case Study

Goals:

- Deploy backend and frontend.
- Connect hosted Postgres.
- Add CI checks.
- Write portfolio case study.
- Document architecture, tradeoffs, metrics, and responsible AI decisions.
