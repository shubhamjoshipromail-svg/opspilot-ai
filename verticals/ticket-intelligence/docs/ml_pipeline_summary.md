# OpsPilot Ticket Intelligence ML Pipeline Summary

## Purpose

OpsPilot Ticket Intelligence is a reproducible English-language ML baseline for operations ticket triage. It classifies ticket category, predicts priority, scores escalation risk, and routes uncertain or high-risk tickets to human review.

The model assists triage. It does not make final customer, refund, legal, compliance, or account decisions.

## Dataset Source

The pipeline uses the public labeled customer-support dataset:

```text
Tobi-Bueck/customer-support-tickets
```

This dataset was selected because it is ticket/helpdesk oriented and includes text, queue/category labels, priority labels, language, ticket type, tags, and support answers. It fits OpsPilot’s routing and operations-review use case better than chatbot-intent-only datasets.

This is a public external benchmark/source dataset, not real private customer data.

## English-Only Scope

Version 1 filters to English tickets only. This keeps TF-IDF features interpretable and makes category/priority metrics easier to explain.

Future multilingual support can use language detection with language-specific models or multilingual transformers such as XLM-R or multilingual DistilBERT.

## Data Regeneration

From the repo root:

```bash
python scripts/build_ticket_dataset.py
```

This writes:

```text
data/raw/customer_support_tickets_en.csv
data/processed/tickets_en_normalized.csv
```

The normalized file includes:

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

The build script drops rows missing `customer_message`, removes exact duplicate customer messages, validates English-only rows when `language` exists, and fails clearly if labels cannot be created.

Current regenerated dataset:

- Source rows: `61,765`
- English rows before duplicate removal: `28,261`
- Normalized rows after duplicate removal: `23,747`

## Splits

From `verticals/ticket-intelligence`:

```bash
python ml/ticket_intelligence/create_splits.py --input ../../data/processed/tickets_en_normalized.csv
```

This writes fixed splits:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

Current split:

- Train: `16,622`
- Validation: `3,562`
- Test: `3,563`

The split uses `random_state=42` and category stratification where possible.

## Model

Two baseline classifiers are trained:

- category classifier
- priority classifier

Model type:

```text
TF-IDF + Logistic Regression
```

TF-IDF settings:

- lowercase: `True`
- n-grams: `(1, 2)`
- min_df: `1`
- max_features: `5000`
- strip_accents: `unicode`

Logistic regression settings:

- solver: `liblinear`
- class_weight: `balanced`
- max_iter: `1000`
- random_state: `42`

Artifacts:

- `ml/ticket_intelligence/artifacts/category_model.pkl`
- `ml/ticket_intelligence/artifacts/priority_model.pkl`
- `ml/ticket_intelligence/artifacts/model_metadata.json`

## Evaluation

Run:

```bash
python ml/ticket_intelligence/evaluate.py
```

Outputs:

- `outputs/metrics.json`
- `outputs/category_classification_report.csv`
- `outputs/priority_classification_report.csv`
- `outputs/category_confusion_matrix.png`
- `outputs/priority_confusion_matrix.png`
- `outputs/error_analysis.csv`

Current test metrics:

| Target | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| Category | 0.4409 | 0.4121 | 0.4430 |
| Priority | 0.5240 | 0.5042 | 0.5235 |

These are honest baseline metrics. Category labels overlap in support data, and text-only priority prediction is limited because real priority often depends on metadata.

## Feature Inspection

Run:

```bash
python ml/ticket_intelligence/feature_inspection.py
```

Outputs:

- `outputs/top_features_by_category.csv`
- `outputs/top_features_by_priority.csv`

This helps explain what terms the TF-IDF/logistic model associates with each class.

## Error Analysis

Run:

```bash
python ml/ticket_intelligence/error_analysis.py
```

Output:

- `outputs/error_analysis.csv`

The file includes true/predicted labels, confidences, and simple likely-reason tags such as `low_confidence`, `label_overlap`, `multiple_intents`, and `priority_depends_on_metadata`.

## Routing And Thresholds

Routing combines:

- category confidence
- priority confidence
- risk keywords
- policy-sensitive signals
- high-priority prediction

Run:

```bash
python ml/ticket_intelligence/threshold_sweep.py
```

Output:

- `outputs/threshold_sweep.csv`

The sweep shows the operational tradeoff between auto-triage rate and human-review rate. At higher thresholds, fewer tickets are auto-triaged, but accuracy on auto-triaged tickets improves.

## Structured Prediction

Run:

```bash
python ml/ticket_intelligence/predict.py
```

The output is model-versioned JSON with nested category, priority, risk, routing, and metadata fields.

## V2 Transformer Benchmark

V2 adds a transformer fine-tuning path without replacing the v1 baselines. The first target is `parent_queue` routing:

```text
input = subject + "\n\n" + body
target = category / true_category
```

The input excludes `answer` and `reference_answer` because those fields are post-resolution information and would leak future knowledge into routing, priority, or risk models.

Primary model:

```text
answerdotai/ModernBERT-base
```

Fallbacks:

- `microsoft/deberta-v3-base`
- `microsoft/deberta-v3-small`
- `distilbert-base-uncased`

Run a smoke test from `verticals/ticket-intelligence`:

```bash
python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name answerdotai/ModernBERT-base \
  --epochs 1 \
  --batch-size 4 \
  --max-length 256 \
  --sample-size 500
```

Run full parent_queue training on Colab/GPU:

```bash
python ml/ticket_intelligence/train_modernbert.py \
  --task parent_queue \
  --model-name answerdotai/ModernBERT-base \
  --epochs 3 \
  --batch-size 8 \
  --max-length 512 \
  --learning-rate 2e-5
```

Outputs:

- `artifacts/modernbert_parent_queue/`
- `outputs/modernbert_parent_queue_metrics.json`
- `outputs/modernbert_parent_queue_classification_report.csv`
- `outputs/modernbert_parent_queue_confusion_matrix.png`
- `outputs/modernbert_parent_queue_predictions.csv`
- `outputs/modernbert_parent_queue_error_analysis.csv`
- `outputs/modernbert_parent_queue_run_summary.md`

Build the model leaderboard:

```bash
python ml/ticket_intelligence/compare_models.py
```

This writes:

- `outputs/model_leaderboard.csv`
- `outputs/model_leaderboard.json`

## Synthetic Data

`ml/ticket_intelligence/data/synthetic_tickets.csv` is reserved for smoke tests and README demos only:

```bash
python ml/ticket_intelligence/train_baseline.py --smoke-test
```

Synthetic smoke metrics should not be presented as model performance.

## Future Work

- run the ModernBERT v2 parent_queue benchmark on Colab/GPU
- calibrate confidence
- add metadata-aware priority/risk models
- benchmark DistilBERT/MiniLM after the baseline is stable
- add multilingual support later
- integrate with FastAPI ticket workflows
- add policy retrieval/RAG
- add human review tables and override feedback
