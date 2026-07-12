# Dataset Audit

Last reviewed: 2026-07-12

This audit is read-only. It describes local regenerated files that are ignored
by Git and can be rebuilt from the public Hugging Face source dataset.

## Source

- Dataset: `Tobi-Bueck/customer-support-tickets`
- Scope used by this project: English rows only
- Local raw export: `data/raw/customer_support_tickets_en.csv`
- Local normalized dataset: `data/processed/tickets_en_normalized.csv`
- Fixed split files: `verticals/ticket-intelligence/data/processed/train.csv`,
  `val.csv`, and `test.csv`

These CSV files are intentionally ignored and should not be committed by
default.

## Raw English Export

| Property | Value |
| --- | ---: |
| Rows | 28,261 |
| Columns | 16 |
| Duplicate full rows | 0 |

Columns:

```text
subject, body, answer, type, queue, priority, language, version,
tag_1, tag_2, tag_3, tag_4, tag_5, tag_6, tag_7, tag_8
```

Observed queue distribution:

| Queue | Rows |
| --- | ---: |
| Technical Support | 8,149 |
| Product Support | 5,305 |
| Customer Service | 4,269 |
| IT Support | 3,333 |
| Billing and Payments | 2,897 |
| Returns and Exchanges | 1,402 |
| Service Outages and Maintenance | 1,106 |
| Sales and Pre-Sales | 843 |
| Human Resources | 553 |
| General Inquiry | 404 |

Observed priority distribution:

| Priority | Rows |
| --- | ---: |
| medium | 11,570 |
| high | 10,917 |
| low | 5,774 |

## Normalized Dataset

| Property | Value |
| --- | ---: |
| Rows | 23,747 |
| Columns | 28 |
| Duplicate full rows | 0 |

Columns:

```text
external_id, subject, body, answer, type, queue, priority, language,
version, tag_1, tag_2, tag_3, tag_4, tag_5, tag_6, tag_7, tag_8,
combined_tags, customer_message, model_text, true_category, true_priority,
status, channel, source, ticket_type, reference_answer, tags
```

Key missing values:

| Column | Missing |
| --- | ---: |
| subject | 2,827 |
| body | 0 |
| answer / reference_answer | 5 |
| true_category | 0 |
| true_priority | 0 |
| customer_message | 0 |
| model_text | 0 |
| ticket_type | 0 |
| language | 0 |

Normalized queue distribution:

| Queue | Rows |
| --- | ---: |
| Technical Support | 6,858 |
| Product Support | 4,429 |
| Customer Service | 3,570 |
| IT Support | 2,833 |
| Billing and Payments | 2,420 |
| Returns and Exchanges | 1,173 |
| Service Outages and Maintenance | 937 |
| Sales and Pre-Sales | 724 |
| Human Resources | 461 |
| General Inquiry | 342 |

Normalized priority distribution:

| Priority | Rows |
| --- | ---: |
| medium | 9,753 |
| high | 9,149 |
| low | 4,845 |

Ticket type distribution:

| Ticket type | Rows |
| --- | ---: |
| Incident | 9,433 |
| Request | 6,880 |
| Problem | 4,931 |
| Change | 2,503 |

## Fixed Splits

Split strategy observed in `split_metadata.json`:

- `category_stratified`
- `random_state=42`

| Split | Rows |
| --- | ---: |
| Train | 16,622 |
| Validation | 3,562 |
| Test | 3,563 |

Test category distribution:

| Category | Rows |
| --- | ---: |
| `technical_support` | 1,029 |
| `product_support` | 665 |
| `customer_service` | 536 |
| `it_support` | 425 |
| `billing_and_payments` | 363 |
| `returns_and_exchanges` | 176 |
| `service_outages_and_maintenance` | 141 |
| `sales_and_pre_sales` | 108 |
| `human_resources` | 69 |
| `general_inquiry` | 51 |

Test priority distribution:

| Priority | Rows |
| --- | ---: |
| medium | 1,490 |
| high | 1,353 |
| low | 720 |

## Leakage And Splitting Risks

- `reference_answer` / `answer` is an outcome field and should not be used as an
  input for classification, priority, risk, or routing models.
- There are no observed customer identifiers, agent identifiers, SLA timestamps,
  satisfaction outcomes, resolution times, or escalation outcomes.
- Chronological splitting is not currently possible from the observed fields.
- The current split is category-stratified, not customer-grouped. Because no
  customer identifier is available, customer-level leakage cannot be tested.
- Tags and ticket type may be useful features, but they may encode operational
  labeling conventions. Treat them as metadata features and evaluate carefully.
- Priority labels exist, but priority is likely under-specified from text alone
  because real priority often depends on customer tier, SLA, business impact,
  incident severity, and account context.

## Privacy Notes

The dataset is public/source benchmark data, not private customer data. Still,
local raw/processed files should remain ignored by default because they are
regenerated data artifacts and may grow or change over time.

## Additional ML Opportunity Fit

The current dataset supports:

- Supervised category / queue classification.
- Supervised priority classification, with caution.
- Supervised ticket type classification.
- Retrieval or response evaluation experiments using `reference_answer`, without
  using it as an input to predictive models.
- Unsupervised topic clustering or duplicate/similarity exploration using text.

The current dataset does not directly support:

- SLA-breach prediction.
- Resolution-time prediction.
- Customer churn prediction.
- Customer satisfaction prediction.
- Agent performance analytics.
- Workload forecasting over time.
- True escalation-risk prediction from observed escalation outcomes.
- Drift monitoring without a live or time-stamped production stream.
