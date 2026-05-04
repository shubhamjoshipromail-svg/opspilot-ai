# OpsPilot AI

**Production-style AI Operations Copilot for ticket triage, escalation-risk scoring, policy-grounded recommendations, human-in-the-loop review, AI evaluation, and ROI analytics.**

## 1. Executive Summary

OpsPilot AI is planned as a production-style AI workflow product for operations teams that manage high-volume, messy, policy-sensitive cases such as support tickets, claims, disputes, billing issues, fulfillment exceptions, and internal service requests.

This repository currently includes the **Phase 1 backend/database vertical slice**. It contains the original planning scaffold plus a minimal FastAPI and SQLAlchemy ticket CRUD workflow that can persist tickets to Railway Postgres through `DATABASE_URL` or fall back to local SQLite for development.

The future product will combine structured case management, machine learning, retrieval over policies and SOPs, LLM-assisted recommendations, evaluation traces, human approval workflows, and business impact analytics.

## 2. Problem Statement

Operations teams often process large volumes of unstructured requests. These cases may arrive with incomplete context, inconsistent language, emotional customer tone, or unclear policy implications. Teams need to decide:

- What type of issue is this?
- How urgent is it?
- Is there escalation risk?
- Which policy or SOP applies?
- What is the best next action?
- Can an AI-generated recommendation be trusted?
- When should a human approve, reject, or override the AI?

Most simple chatbot demos do not solve this workflow. OpsPilot AI is designed as an operational decision-support system, not a generic chat interface.

## 3. Why This Matters For Business

Better operations workflows can reduce manual review time, improve consistency, prevent costly escalations, and create measurable business value. A mature version of OpsPilot AI should help teams:

- Shorten ticket resolution time.
- Prioritize risky cases earlier.
- Improve policy compliance.
- Reduce inconsistent responses across agents.
- Capture human feedback for model improvement.
- Track AI quality and operational ROI.
- Create audit trails for sensitive decisions.

## 4. Target Users

- Customer support operations managers.
- Claims operations teams.
- Trust and safety reviewers.
- Billing and dispute teams.
- Internal IT or HR service desk teams.
- AI operations reviewers.
- Business operations analysts.
- Product operations teams.

## 5. Target Industries

- SaaS and B2B software.
- E-commerce and marketplaces.
- Fintech and payments.
- Insurance claims.
- Healthcare operations.
- Logistics and shipping.
- Telecom and subscription businesses.
- Internal enterprise service operations.

## 6. Target Roles This Project Demonstrates Fit For

This project is intentionally designed to show skills relevant to:

- Applied AI Engineer.
- AI Solutions Engineer.
- Forward-Deployed AI Engineer.
- Operations Data Scientist.
- Product Data Scientist.
- AI Product Analyst.
- Decision Scientist.
- Strategy & Ops AI.
- AI Workflow Automation Engineer.

## 7. Future Product Workflow

The planned workflow for each incoming case:

1. Store the ticket in Postgres.
2. Classify the ticket category.
3. Predict priority and escalation risk.
4. Retrieve relevant policy or SOP evidence.
5. Recommend the next best action.
6. Draft a customer or internal response.
7. Evaluate AI output for quality, groundedness, hallucination risk, policy compliance, and escalation correctness.
8. Require human approval for risky actions.
9. Save human feedback, override reasons, and final decisions.
10. Produce analytics on business impact, model performance, workflow quality, and ROI.

## 8. Planned Architecture

The planned system will use modular service boundaries:

- **Backend API:** FastAPI service for case intake, review workflow, decisions, and analytics endpoints.
- **Database:** Postgres for tickets, predictions, recommendations, feedback, and audit trails.
- **Model services:** Swappable interfaces for classification, risk scoring, retrieval, recommendation, drafting, and evaluation.
- **Policy retrieval layer:** Initially keyword-based, later embeddings/RAG.
- **Human review layer:** Approval, rejection, override, and escalation workflows.
- **Frontend:** Streamlit MVP for case inbox, review workspace, model traces, and analytics.
- **Evaluation layer:** Offline and online evaluation of predictions, recommendations, policy grounding, and response quality.

## 9. Planned Tech Stack

- Python.
- FastAPI.
- Postgres.
- SQLAlchemy.
- Pydantic.
- Streamlit.
- pandas.
- scikit-learn.
- pytest.
- Future LLM providers and model APIs.
- Future vector search or retrieval backend.
- Future deployment with Railway or similar platform.

## 10. Core Modules

Planned modules include:

- Ticket ingestion and normalization.
- Triage classification.
- Escalation-risk scoring.
- Policy and SOP retrieval.
- Next-best-action recommendation.
- Response drafting.
- AI output evaluation.
- Human-in-the-loop review.
- Feedback capture.
- Operational analytics.
- Model performance analytics.
- ROI analytics.

## 11. Planned Database Design

Future tables may include:

- `tickets`: incoming cases and normalized fields. This is the only table implemented in Phase 1.
- `ticket_events`: timeline of case changes.
- `triage_predictions`: category, confidence, and model version.
- `risk_scores`: escalation risk, priority, drivers, and model version.
- `retrieval_results`: policies or SOPs used as evidence.
- `recommendations`: suggested actions and rationale.
- `draft_responses`: AI-generated response drafts.
- `evaluation_traces`: quality, groundedness, hallucination risk, and compliance scores.
- `human_reviews`: approvals, rejections, edits, overrides, and reviewer notes.
- `model_versions`: version metadata for algorithms and prompts.
- `business_metrics`: time saved, escalation reduction, throughput, and ROI calculations.

Only the `tickets` table is implemented so far. Phase 1.6 aligns it with the normalized English dataset at `data/processed/tickets_en_normalized.csv`, including `true_category`, `true_priority`, `ticket_type`, `reference_answer`, and pipe-separated `tags`. The remaining tables are still planned and intentionally not implemented yet.

## 12. Planned Model Strategy

The project will keep algorithms swappable:

- `triage_model_v1`: simple baseline classifier.
- `triage_model_v2`: embeddings plus classifier.
- `triage_model_v3`: transformer or deep learning model.
- `risk_model_v1`: rules.
- `risk_model_v2`: supervised ML.
- `retriever_v1`: keyword search.
- `retriever_v2`: embeddings/RAG.
- `recommender_v1`: rules.
- `recommender_v2`: contextual bandit.
- `judge_v1`: LLM rubric scoring.
- `judge_v2`: pairwise preference or reward model.

The scaffold does not implement any of these models yet.

## 13. Planned Analytics Strategy

Analytics will focus on three levels:

- **Operational analytics:** volume, backlog, resolution time, approval rate, override rate, escalation rate.
- **Model analytics:** accuracy, precision, recall, calibration, confidence distributions, error slices.
- **Business analytics:** estimated time saved, avoided escalations, cost per case, quality improvements, ROI.

## 14. Planned Evaluation Strategy

Evaluation will include:

- Classification metrics for triage labels.
- Risk scoring metrics and calibration checks.
- Retrieval relevance and policy coverage.
- Response quality and policy compliance rubrics.
- AI judge scoring with human validation.
- Human feedback loops for continuous improvement.
- Responsible AI checks for hallucination risk and unsafe automation.

## 15. Planned Human-In-The-Loop Workflow

The future system will require human review for high-risk actions, low-confidence predictions, policy-sensitive decisions, and customer-facing drafts. Reviewers should be able to:

- Approve recommendations.
- Edit response drafts.
- Reject AI suggestions.
- Override category, priority, or risk.
- Add notes explaining decisions.
- Escalate cases to a specialist.
- Feed final outcomes back into evaluation datasets.

## 16. Planned Business ROI Logic

Future ROI estimates may combine:

- Average handling time reduction.
- Cases deflected from escalation.
- Improved first-contact resolution.
- Reduced manual policy lookup time.
- Reduced rework due to incorrect responses.
- Labor cost assumptions.
- Quality and compliance improvements.

ROI calculations will be presented as decision-support analytics, not as exact financial claims.

## 17. MVP Roadmap

- **Phase 0:** Repo scaffold and documentation.
- **Phase 1:** Postgres plus FastAPI ticket CRUD. Current phase.
- **Phase 2:** Streamlit case inbox and workspace.
- **Phase 3:** Baseline ML models.
- **Phase 4:** Policy retrieval/RAG.
- **Phase 5:** LLM recommendation and response drafting.
- **Phase 6:** AI judge and evaluation traces.
- **Phase 7:** Analytics and ROI dashboard.
- **Phase 8:** Advanced model experiments.
- **Phase 9:** Deployment and portfolio case study.

## 18. Future Technical Upgrades

- Background workers for async model jobs.
- Vector database or Postgres vector extension.
- Model registry and experiment tracking.
- Prompt versioning.
- Evaluation datasets and golden test sets.
- Role-based access control.
- Observability, logging, and tracing.
- CI/CD pipeline.
- Dockerized local development.
- Production deployment.

## 19. Local Setup Instructions For The Scaffold

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy environment variables:

```bash
cp .env.example .env
```

Initialize database tables:

```bash
python -m backend.app.db.init_db
```

Run the backend:

```bash
uvicorn backend.app.main:app --reload
```

Check the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Run the frontend placeholder:

```bash
streamlit run frontend/streamlit_app.py
```

## 20. Railway Postgres Notes

Railway Postgres can be used later for the deployed database. The planned flow:

1. Create a Railway project.
2. Add a Postgres service.
3. Copy the database connection string into `DATABASE_URL`.
4. Configure backend deployment environment variables.
5. Add migrations only after the data model is implemented.

Phase 1 reads `DATABASE_URL` from `.env`. If `DATABASE_URL` is not set, the backend uses local SQLite at `sqlite:///./opspilot_dev.db`.

## 21. Folder Structure

```text
opspilot-ai/
  README.md
  backend/
  frontend/
  data/
  experiments/
  docs/
  tests/
```

Detailed structure is included in the repository files and docs.

## 22. Portfolio/Resume Positioning

OpsPilot AI should be positioned as a production-style applied AI project that connects product thinking, operations analytics, ML system design, human review, evaluation, and measurable business impact.

It is not intended to be framed as:

- A toy chatbot.
- A basic dashboard.
- A Kaggle-only notebook.
- A single-model demo.

It is intended to demonstrate the ability to design AI workflows that could be used inside real operations teams.

## 23. Example Future Resume Bullets

- Designed and built OpsPilot AI, a production-style AI operations copilot for case triage, escalation-risk scoring, policy-grounded recommendations, human review, evaluation, and ROI analytics.
- Implemented modular model interfaces for swappable triage, risk, retrieval, recommendation, and evaluation components.
- Built a human-in-the-loop review workflow that captured approvals, overrides, reviewer notes, model versions, and evaluation traces.
- Developed analytics dashboards tracking operational throughput, model performance, escalation risk, quality metrics, and estimated business ROI.
- Created a policy-grounded AI recommendation workflow with retrieval evidence, response drafting, hallucination-risk checks, and compliance evaluation.

## 24. Limitations And Responsible AI Considerations

This repository is only a scaffold. It does not yet include a working product, trained models, retrieval logic, LLM calls, or database implementation.

Future versions should account for:

- Human approval for high-impact decisions.
- Clear audit trails for AI-generated recommendations.
- Model confidence and uncertainty.
- False positives and false negatives in escalation detection.
- Policy grounding and citation quality.
- Hallucination risk.
- Bias and inconsistent treatment across customer groups.
- Data privacy and retention.
- Secure handling of API keys and customer data.

OpsPilot AI should assist human operators, not silently automate sensitive decisions without review.
