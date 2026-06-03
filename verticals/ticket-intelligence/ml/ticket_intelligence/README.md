# Ticket Intelligence ML

This folder contains the canonical scripts for the OpsPilot Ticket Intelligence baseline pipeline.

## Source Data

The source dataset is the public Hugging Face dataset:

```text
Tobi-Bueck/customer-support-tickets
```

The project uses an English-only v1 scope. Regenerate the raw and normalized data from the repository root:

```bash
python scripts/build_ticket_dataset.py
```

Generated files:

- `data/raw/customer_support_tickets_en.csv`
- `data/processed/tickets_en_normalized.csv`

These files are intentionally gitignored.

## Full Pipeline

From `verticals/ticket-intelligence`:

```bash
python ml/ticket_intelligence/data_audit.py
python ml/ticket_intelligence/create_splits.py --input ../../data/processed/tickets_en_normalized.csv
python ml/ticket_intelligence/train_baseline.py
python ml/ticket_intelligence/evaluate.py
python ml/ticket_intelligence/feature_inspection.py
python ml/ticket_intelligence/error_analysis.py
python ml/ticket_intelligence/threshold_sweep.py
python ml/ticket_intelligence/predict.py
```

## Scripts

- `data_audit.py`: reports dataset paths, row counts, columns, missing key fields, label distributions, duplicate message count, language distribution, and seed/load scripts.
- `create_splits.py`: creates fixed train/validation/test splits with `random_state=42`.
- `train_baseline.py`: trains TF-IDF + Logistic Regression category and priority models.
- `evaluate.py`: evaluates saved models on the fixed test split.
- `feature_inspection.py`: exports top positive TF-IDF features per class.
- `error_analysis.py`: exports test-set mistakes with confidence and likely reason tags.
- `threshold_sweep.py`: estimates auto-triage versus human-review tradeoffs.
- `predict.py`: emits structured JSON for a sample or provided ticket.
- `routing.py`: risk scoring and human-review routing logic.
- `train_transformer.py`: dependency-gated transformer-ready script for future benchmarking.

## Artifacts

Saved under `artifacts/`:

- `category_model.pkl`
- `priority_model.pkl`
- `model_metadata.json`

## Outputs

Saved under `outputs/`:

- `metrics.json`
- `category_classification_report.csv`
- `priority_classification_report.csv`
- `category_confusion_matrix.png`
- `priority_confusion_matrix.png`
- `top_features_by_category.csv`
- `top_features_by_priority.csv`
- `error_analysis.csv`
- `threshold_sweep.csv`
- `predictions.jsonl`

## Current Baseline Metrics

Fixed English-only test split:

- Category accuracy: `0.4415`
- Category macro-F1: `0.4133`
- Category weighted-F1: `0.4441`
- Priority accuracy: `0.5240`
- Priority macro-F1: `0.5048`
- Priority weighted-F1: `0.5236`

Priority is a text-only baseline and should be interpreted carefully. In production, priority should include operational metadata.

## Smoke Tests

The 60-row synthetic dataset is only for smoke tests:

```bash
python ml/ticket_intelligence/train_baseline.py --smoke-test
```

Do not present synthetic metrics as the main model result.
