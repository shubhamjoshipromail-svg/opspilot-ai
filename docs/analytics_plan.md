# Analytics Plan

## Operational Analytics

Planned metrics:

- Ticket volume by day, channel, and category.
- Backlog size and aging.
- Average handling time.
- First response time.
- Resolution time.
- Escalation rate.
- Approval rate.
- Override rate.
- Reopen rate.
- SLA risk.

Dashboard ideas:

- Operations overview.
- Case volume trends.
- Category mix.
- Escalation queue.
- Human review throughput.
- Policy-heavy case segments.

## Model Analytics

Planned metrics:

- Triage accuracy.
- Precision, recall, and F1 by category.
- Risk score calibration.
- High-risk false negatives.
- Confidence distribution.
- Retrieval relevance.
- Recommendation acceptance rate.
- AI draft edit distance.
- Evaluation score trends.
- Error slices by category, channel, and policy type.

Dashboard ideas:

- Model performance overview.
- Error analysis table.
- Confidence and calibration view.
- Recommendation acceptance trends.
- Human override reasons.

## Business Analytics

Planned metrics:

- Estimated minutes saved.
- Cost per case.
- Manual review time avoided.
- Escalations avoided.
- Improved first-contact resolution.
- Rework reduction.
- Agent productivity.
- Quality improvement.

Dashboard ideas:

- ROI summary.
- Time savings over time.
- Estimated cost savings.
- Escalation prevention.
- Scenario analysis for adoption rates.

## ROI Metrics

Possible ROI formula:

```text
estimated_savings =
  (minutes_saved_per_case * cases_assisted * labor_cost_per_minute)
  + estimated_escalation_cost_avoided
  - estimated_ai_and_infrastructure_cost
```

These calculations should be presented as estimates with assumptions clearly shown.

## Dashboard Ideas

Future Streamlit dashboards:

- Executive ROI summary.
- Operations manager dashboard.
- Model quality dashboard.
- Human review dashboard.
- Evaluation trace explorer.
- Policy grounding dashboard.
