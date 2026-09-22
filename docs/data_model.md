# Data Model

This document describes the planned data model. The actual database tables are not implemented yet.

## Planned Tables

## tickets

Stores incoming cases.

Planned fields:

- `id`
- `external_ticket_id`
- `source`
- `customer_id`
- `subject`
- `description`
- `raw_text`
- `status`
- `created_at`
- `updated_at`

Why it exists: central record for every case processed by OpsPilot AI.

## ticket_events

Stores a timeline of actions and status changes.

Planned fields:

- `id`
- `ticket_id`
- `event_type`
- `event_payload`
- `created_by`
- `created_at`

Why it exists: supports auditability and workflow history.

## triage_predictions

Stores category predictions.

Planned fields:

- `id`
- `ticket_id`
- `model_version`
- `predicted_category`
- `confidence`
- `raw_output`
- `created_at`

Why it exists: records model output and version for later evaluation.

## risk_scores

Stores priority and escalation-risk output.

Planned fields:

- `id`
- `ticket_id`
- `model_version`
- `risk_score`
- `priority`
- `risk_drivers`
- `confidence`
- `created_at`

Why it exists: supports prioritization, review routing, and risk analytics.

## retrieval_results

Stores policy or SOP evidence returned for a ticket.

Planned fields:

- `id`
- `ticket_id`
- `retriever_version`
- `document_id`
- `document_title`
- `snippet`
- `score`
- `created_at`

Why it exists: creates an evidence trail for recommendations and AI drafts.

## recommendations

Stores next-best-action recommendations.

Planned fields:

- `id`
- `ticket_id`
- `recommender_version`
- `recommended_action`
- `rationale`
- `supporting_evidence_ids`
- `confidence`
- `created_at`

Why it exists: records the action suggested to the human reviewer.

## draft_responses

Stores AI-generated customer or internal response drafts.

Planned fields:

- `id`
- `ticket_id`
- `drafting_version`
- `draft_text`
- `tone`
- `evidence_ids`
- `created_at`

Why it exists: separates suggested response text from final human-approved output.

## evaluation_traces

Stores AI quality and compliance checks.

Planned fields:

- `id`
- `ticket_id`
- `target_type`
- `target_id`
- `evaluator_version`
- `quality_score`
- `groundedness_score`
- `hallucination_risk`
- `policy_compliance_score`
- `escalation_correctness_score`
- `rubric_notes`
- `created_at`

Why it exists: makes AI output measurable and auditable.

## human_reviews

Stores human decisions.

Planned fields:

- `id`
- `ticket_id`
- `reviewer_id`
- `decision`
- `edited_response`
- `override_category`
- `override_priority`
- `override_risk_score`
- `review_notes`
- `created_at`

Why it exists: keeps humans in control and captures feedback for improvement.

## model_versions

Stores model and prompt metadata.

Planned fields:

- `id`
- `model_family`
- `version_name`
- `description`
- `provider`
- `configuration`
- `created_at`

Why it exists: supports reproducibility, comparison, and rollback.

## business_metrics

Stores aggregated workflow and ROI metrics.

Planned fields:

- `id`
- `metric_date`
- `ticket_volume`
- `average_handling_time`
- `estimated_minutes_saved`
- `escalations_avoided`
- `approval_rate`
- `override_rate`
- `estimated_cost_savings`

Why it exists: connects AI workflow performance to business value.

## Auditability Strategy

OpsPilot AI should store:

- Original ticket text.
- Model outputs.
- Model versions.
- Retrieved evidence.
- Recommended actions.
- Draft responses.
- Evaluation scores.
- Human decisions.
- Override reasons.
- Final outcomes.

This makes it possible to inspect why the system recommended an action and how a human responded.
