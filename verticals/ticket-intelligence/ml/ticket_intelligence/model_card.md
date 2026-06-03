# Model Card: OpsPilot Ticket Intelligence v1

## Model Details

- Model name: OpsPilot Ticket Intelligence v1
- Version: `ticket-intelligence-v1`
- Model type: TF-IDF vectorization plus Logistic Regression classifiers
- Tasks:
  - Ticket category / queue classification
  - Priority prediction
  - Escalation-risk scoring and routing via deterministic rules
- Language scope: English-only v1

## Intended Use

OpsPilot Ticket Intelligence v1 is intended for AI-assisted triage of English-language support and operations tickets. It suggests a category, predicts priority, estimates escalation risk, and recommends whether a ticket can be auto-triaged or should go to human/supervisor review.

## Not Intended Use

This model is not intended for automatic final decisions, refunds, legal or compliance judgments, account enforcement, or customer-facing action without human review.

## Dataset

The dataset is regenerated from the public labeled customer-support ticket dataset `Tobi-Bueck/customer-support-tickets`. The pipeline filters to English tickets only, normalizes fields, removes exact duplicate customer messages, and writes:

- `data/raw/customer_support_tickets_en.csv`
- `data/processed/tickets_en_normalized.csv`

The normalized dataset includes:

- `external_id`
- `customer_message`
- `true_category`
- `true_priority`
- `status`
- `channel`
- `source`
- `ticket_type`
- `language`
- `reference_answer`
- `tags`

This is a public external benchmark/source dataset, not private customer data.

## Labels

- Category label: normalized from the dataset queue/category field.
- Priority label: normalized from the dataset priority field.

## Current Metrics

Current fixed-split test results:

- Category accuracy: `0.4415`
- Category macro-F1: `0.4133`
- Category weighted-F1: `0.4441`
- Priority accuracy: `0.5240`
- Priority macro-F1: `0.5048`
- Priority weighted-F1: `0.5236`

Priority is a text-only baseline. In real operations systems, priority often depends on metadata such as customer tier, SLA, recurrence, account value, incident severity, or operational impact.

## Human Oversight Policy

The system routes tickets to human or supervisor review when:

- category confidence is low
- escalation risk is high
- policy-sensitive language is detected
- high-risk terms such as legal, fraud, chargeback, regulator, safety, or compliance appear

The output is decision support, not an autonomous operations decision.

## Limitations

- English-only v1.
- TF-IDF features cannot deeply understand context or long-range semantics.
- Labels may overlap, especially between technical support, product support, IT support, and customer service.
- Priority prediction is limited because many priority decisions require non-text metadata.
- Rule-based escalation risk may miss subtle dissatisfaction, sarcasm, or organization-specific risk signals.
- Dataset distribution may not match a specific company’s live support queue.

## Failure Modes

- Confusing overlapping support queues.
- Over-prioritizing messages with urgent wording but low operational impact.
- Under-prioritizing tickets that look neutral but come from high-value or SLA-bound customers.
- Mistaking policy-sensitive terms as ordinary support requests.
- Learning shortcuts from tags, phrases, or queue-specific vocabulary.

## Monitoring Plan

Track:

- category and priority macro-F1
- per-class recall for high-risk queues
- confidence distribution
- human-review rate
- override rate
- escalation misses
- high-risk tickets auto-triaged by mistake
- drift in category, priority, and language distributions

## Future Work

- Calibrate model confidence.
- Add operational metadata for priority and escalation risk.
- Fine-tune a lightweight transformer benchmark.
- Add multilingual support with either language detection plus language-specific models or multilingual transformer models such as XLM-R or multilingual DistilBERT.
- Integrate with FastAPI ticket creation and human review tables.
- Add policy retrieval/RAG after the core triage pipeline is stable.
