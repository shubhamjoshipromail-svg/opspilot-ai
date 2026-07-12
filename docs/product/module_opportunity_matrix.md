# Module Opportunity Matrix

Last reviewed: 2026-07-12

This matrix evaluates possible customer-support intelligence modules against
the currently observed repository and dataset. It does not claim that planned
features already exist.

## Recommendation

The next highest-ROI vertical is:

```text
Policy-grounded human review
```

This builds directly on the existing Ticket Intelligence system:

1. Route a ticket with the hosted model.
2. Retrieve relevant policy evidence from `data/policies`.
3. Generate a conservative next-action recommendation.
4. Capture human approval, edit, rejection, escalation, and override notes.
5. Track approval rate, override rate, review workload, and policy coverage.

This is more portfolio-relevant than adding another standalone classifier
because it shows AI workflow design, retrieval grounding, auditability, backend
product architecture, and operations analytics.

## Opportunity Matrix

| Module | Current data fit | Suggested method | MVP fit | Notes |
| --- | --- | --- | --- | --- |
| Ticket category classification | Strong | Supervised ML | Already core | Existing local 10-label baseline and hosted 7-label runtime |
| Department / queue routing | Strong | Supervised ML + thresholds | Already core | Runtime API predicts seven operational queues |
| Priority prediction | Medium | Supervised ML + metadata later | Later MVP | Labels exist, but text-only priority is limited |
| Confidence-based human review | Strong | Rules + persisted review records | Next | Already partially present via `auto_route` / `human_review` |
| Policy retrieval | Strong for prototype | Keyword/BM25 first, embeddings later | Next | `data/policies` already exists |
| Policy-grounded recommendation | Medium | Rules first, LLM later | Next | Needs retrieval evidence and human review controls |
| Human overrides and feedback | Strong product fit | CRUD + audit tables | Next | No special labels required |
| Suggested response retrieval | Medium | Retrieval over historical/reference answers | Later | `reference_answer` exists, but should not leak into predictive training |
| Ticket summarization | Medium | LLM-assisted | Later | Needs conversations or long bodies for strong value |
| Conversation summarization | Weak current fit | LLM-assisted | Later | Dataset has ticket text, not full conversation threads |
| Ticket type classification | Strong | Supervised ML | Later | `ticket_type` exists: Incident, Request, Problem, Change |
| Subcategory / intent classification | Medium | Supervised or weak labels | Later | Tags may help, but taxonomy needs careful design |
| Urgency detection | Medium | Rules + supervised labels | Later | Priority exists, but urgency is not a separate verified label |
| Escalation-risk prediction | Weak as supervised task | Rules now, supervised only with outcomes | Later | No true escalation outcome label observed |
| SLA-breach-risk prediction | Not supported | Needs SLA/timestamps | Not MVP | No SLA fields observed |
| Customer sentiment detection | Medium | LLM/rules/third-party model | Later | No sentiment label observed |
| Churn / dissatisfaction risk | Not supported | Needs account/customer outcomes | Not MVP | No churn or satisfaction labels observed |
| Duplicate-ticket detection | Medium | Embeddings/similarity | Later | Text supports similarity, but no duplicate labels |
| Similar-ticket retrieval | Medium | Embeddings/BM25 | Later | Useful with reference answers and historical cases |
| Root-cause clustering | Medium | Unsupervised clustering | Later | Needs human naming and validation |
| Emerging incident detection | Weak current fit | Time-series + clustering | Later | No production time stream observed |
| Resolution-time prediction | Not supported | Needs timestamps/outcomes | Not MVP | No resolution-time labels observed |
| First-contact resolution prediction | Not supported | Needs outcome labels | Not MVP | No FCR label observed |
| PII detection/redaction | Medium | Rules + NER model | Later | Useful platform module, but not dataset-labeled |
| Toxicity/abuse/policy risk | Medium | Rules or existing safety model | Later | No verified toxicity labels |
| Quality-assurance scoring | Medium | LLM rubric + human validation | Later | Needs reviewed examples for calibration |
| Agent-performance analytics | Not supported | Needs agent IDs/outcomes | Not MVP | No agent identifiers observed |
| Workload forecasting | Not supported | Needs timestamped volume stream | Not MVP | No reliable time-series fields observed |
| Model monitoring / drift | Medium product fit | Production logs + dashboards | Later | Prediction persistence creates a starting point |

## Why Policy-Grounded Review Comes Next

- It preserves and extends the existing routing model.
- It uses current assets instead of requiring a new dataset.
- It creates a complete operator workflow rather than another isolated model.
- It demonstrates responsible AI: evidence, approval, override, and audit trail.
- It creates measurable analytics: review rate, override rate, evidence coverage,
  and eventually estimated time saved.

## Suggested Implementation Slice

1. `policy_documents` loader service over `data/policies/*.md`.
2. Keyword retrieval endpoint returning policy name, snippet, score, and matched
   terms.
3. Recommendation service that combines routing result, confidence, and policy
   evidence.
4. `human_reviews` table and CRUD routes.
5. Streamlit case-review workspace.
6. Dashboard metrics for pending reviews, approvals, rejections, overrides, and
   policy evidence coverage.

Avoid adding LLM drafting until policy retrieval and human review records are in
place.
