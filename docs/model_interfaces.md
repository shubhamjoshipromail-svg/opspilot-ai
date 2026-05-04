# Model Interfaces

This document describes planned future interfaces. These interfaces are not implemented yet.

## Design Goal

OpsPilot AI should avoid locking the workflow into a single algorithm or provider. Each model family should expose a stable interface while allowing implementations to change over time.

## Planned Triage Model Interface

Purpose: classify the ticket category.

Planned input:

- Ticket subject.
- Ticket description.
- Optional customer/account metadata.

Planned output:

- Predicted category.
- Confidence score.
- Candidate categories.
- Model version.
- Explanation or key features.

Future versions:

- `triage_model_v1`: simple baseline classifier.
- `triage_model_v2`: embeddings plus classifier.
- `triage_model_v3`: transformer/deep learning.

## Planned Risk Model Interface

Purpose: estimate priority and escalation risk.

Planned input:

- Ticket text.
- Category.
- Customer sentiment.
- Account history signals.
- Policy-sensitive keywords.

Planned output:

- Risk score.
- Priority level.
- Risk drivers.
- Confidence score.
- Model version.

Future versions:

- `risk_model_v1`: rules.
- `risk_model_v2`: supervised ML.

## Planned Retriever Interface

Purpose: find relevant policy and SOP evidence.

Planned input:

- Ticket text.
- Predicted category.
- Query terms.

Planned output:

- Ranked documents.
- Snippets.
- Relevance scores.
- Retriever version.

Future versions:

- `retriever_v1`: keyword search.
- `retriever_v2`: embeddings/RAG.

## Planned Recommendation Interface

Purpose: suggest the next best action.

Planned input:

- Ticket.
- Triage result.
- Risk score.
- Retrieved policy evidence.
- Historical outcomes.

Planned output:

- Recommended action.
- Rationale.
- Required human review flag.
- Confidence score.
- Recommender version.

Future versions:

- `recommender_v1`: rules.
- `recommender_v2`: contextual bandit.

## Planned Evaluator Interface

Purpose: evaluate AI output quality and safety.

Planned input:

- Ticket.
- Retrieved evidence.
- Recommendation.
- Draft response.
- Applicable policy text.

Planned output:

- Quality score.
- Groundedness score.
- Hallucination-risk score.
- Policy compliance score.
- Escalation correctness score.
- Rubric notes.
- Evaluator version.

Future versions:

- `judge_v1`: LLM rubric scoring.
- `judge_v2`: pairwise preference or reward model.

## Model Versioning Strategy

Every model output should include:

- Model family.
- Version name.
- Configuration metadata.
- Prompt version, when applicable.
- Provider, when applicable.
- Timestamp.
- Raw or structured output.

This enables comparison, auditing, regression analysis, and rollback.
