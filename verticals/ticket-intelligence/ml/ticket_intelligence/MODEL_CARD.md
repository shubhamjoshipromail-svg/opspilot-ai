# Model Card: OpsPilot Ticket Intelligence

## Model Details

- Model name: OpsPilot Ticket Intelligence
- Current committed baseline version: `ticket-intelligence-v1`
- V1 model types:
  - TF-IDF vectorization plus Logistic Regression classifiers
  - TensorFlow/Keras CNN text classifiers when trained from the Colab workflow
- V2 benchmark model type:
  - Transformer sequence classifier fine-tuning, beginning with `answerdotai/ModernBERT-base`
- Tasks:
  - Ticket category / queue classification
  - Priority prediction
  - Escalation-risk scoring and routing via deterministic rules
  - V2 parent_queue routing benchmark
- Language scope: English-only

## Intended Use

OpsPilot Ticket Intelligence is intended for AI-assisted triage of English-language support and operations tickets. It suggests a category, predicts priority, estimates escalation risk, and recommends whether a ticket can be auto-triaged or should go to human/supervisor review.

## Not Intended Use

This model is not intended for automatic final decisions, refunds, legal or compliance judgments, account enforcement, or customer-facing action without human review.

## Dataset

The dataset is regenerated from the public labeled customer-support ticket dataset `Tobi-Bueck/customer-support-tickets`. The pipeline filters to English tickets only, normalizes fields, removes exact duplicate customer messages, and writes:

- `data/raw/customer_support_tickets_en.csv`
- `data/processed/tickets_en_normalized.csv`

The normalized dataset includes:

- `external_id`
- `subject`
- `body`
- `customer_message`
- `model_text`
- `true_category`
- `true_priority`
- `queue`
- `priority`
- `status`
- `channel`
- `source`
- `ticket_type`
- `language`
- `reference_answer`
- `tags`
- `combined_tags`

This is a public external benchmark/source dataset, not private customer data.

`answer` / `reference_answer` is not used as input for category, priority, parent_queue, or risk prediction because it is post-resolution information and would cause leakage.

## Labels

- Category label: normalized from the dataset queue/category field.
- Priority label: normalized from the dataset priority field.

## Current Metrics

Current fixed-split test results:

- Category accuracy: `0.4409`
- Category macro-F1: `0.4121`
- Category weighted-F1: `0.4430`
- Priority accuracy: `0.5240`
- Priority macro-F1: `0.5042`
- Priority weighted-F1: `0.5235`

Priority is a text-only baseline. In real operations systems, priority often depends on metadata such as customer tier, SLA, recurrence, account value, incident severity, or operational impact.

## V2 Transformer Benchmark

The v2 transformer benchmark is additive and does not replace v1 until results justify it.

First target:

```text
parent_queue classification
```

Input:

```text
subject + "\n\n" + body
```

Label:

```text
category / true_category
```

Primary model:

```text
answerdotai/ModernBERT-base
```

Fallbacks:

- `microsoft/deberta-v3-base`
- `microsoft/deberta-v3-small`
- `distilbert-base-uncased`

The v2 script saves model artifacts, tokenizer, label mappings, metrics, classification report, confusion matrix, predictions, error analysis, and run summary. It uses the same fixed train/validation/test splits as v1 for fair comparison.

This benchmark is an enterprise AI workflow prototype for ticket routing. It is not production-ready automation.

## Human Oversight Policy

The system routes tickets to human or supervisor review when:

- category confidence is low
- escalation risk is high
- policy-sensitive language is detected
- high-risk terms such as legal, fraud, chargeback, regulator, safety, or compliance appear

The output is decision support, not an autonomous operations decision.

## Limitations

- English-only.
- TF-IDF features cannot deeply understand context or long-range semantics.
- TensorFlow and transformer models still depend on the quality and representativeness of the public source labels.
- Labels may overlap, especially between technical support, product support, IT support, and customer service.
- Priority prediction is limited because many priority decisions require non-text metadata.
- Rule-based escalation risk may miss subtle dissatisfaction, sarcasm, or organization-specific risk signals.
- Dataset distribution may not match a specific company’s live support queue.
- The weak-risk label workflow creates policy-derived labels, not true observed escalation outcomes.

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
- Run and compare the ModernBERT parent_queue benchmark on Colab/GPU.
- Add specialized transformer models for priority, ticket_type, and raw_queue only after parent_queue is documented.
- Consider a shared-encoder multitask model later.
- Add multilingual support with either language detection plus language-specific models or multilingual transformer models such as XLM-R or multilingual DistilBERT.
- Integrate with FastAPI ticket creation and human review tables.
- Add policy retrieval/RAG after the core triage pipeline is stable.
