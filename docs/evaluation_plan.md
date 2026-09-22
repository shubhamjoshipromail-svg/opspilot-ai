# Evaluation Plan

## Classification Evaluation

For triage classification:

- Accuracy.
- Precision, recall, and F1.
- Confusion matrix.
- Per-category performance.
- Confidence calibration.
- Manual review of common errors.

## Risk Scoring Evaluation

For escalation-risk scoring:

- ROC-AUC or PR-AUC when labeled data exists.
- Calibration curves.
- High-risk false negative rate.
- Review queue precision.
- Correlation with actual escalations.
- Segment analysis by category and channel.

## Retrieval Evaluation

For policy retrieval:

- Top-k relevance.
- Recall of known applicable policies.
- Evidence coverage.
- Citation quality.
- Irrelevant evidence rate.
- Human reviewer usefulness rating.

## Response Quality Evaluation

For drafted responses:

- Clarity.
- Completeness.
- Tone appropriateness.
- Policy grounding.
- Correctness.
- Escalation sensitivity.
- Need for human edits.

## AI Judge Plan

Future AI judge versions may score:

- Groundedness.
- Hallucination risk.
- Policy compliance.
- Customer empathy.
- Actionability.
- Escalation correctness.

AI judge outputs should be validated against human review samples. The judge should not be treated as automatically correct.

## Human Feedback Plan

Human reviewers should provide:

- Approval or rejection.
- Edited response.
- Override category.
- Override risk score or priority.
- Override reason.
- Notes on missing evidence.
- Final outcome.

This feedback can later support supervised learning, prompt improvement, policy retrieval tuning, and quality dashboards.

## Responsible AI Checks

Planned checks:

- Flag low-confidence outputs.
- Require approval for high-risk cases.
- Detect missing policy evidence.
- Detect unsupported claims in drafts.
- Track hallucination-risk scores.
- Monitor error rates across customer segments when data permits.
- Preserve audit trails for decisions.
- Avoid fully automated sensitive decisions without review.
