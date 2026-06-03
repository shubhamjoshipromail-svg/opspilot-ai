# OpsPilot Ticket Intelligence

OpsPilot Ticket Intelligence is the ML vertical for OpsPilot AI: an AI-assisted triage layer for English-language support and operations tickets.

It classifies ticket category, predicts priority, estimates escalation risk, and routes low-confidence or high-risk tickets to human review. It is decision support, not an automatic final-decision system.

## Dataset

Version 1 uses the public labeled customer-support ticket dataset `Tobi-Bueck/customer-support-tickets`.

Why this dataset:

- ticket/helpdesk oriented
- includes queue/category labels
- includes priority labels
- includes language, ticket type, tags, and support answers
- supports later reference-answer evaluation

This is a public external benchmark/source dataset, not real private customer data.

Version 1 is English-only. The build script filters `language == "en"` when the language column exists. German/multilingual support is future work.

Raw and processed data are regenerated locally and are not committed because root `data/raw/*` and `data/processed/*` are gitignored.

## Reproduce

From the repository root:

```bash
python scripts/build_ticket_dataset.py
```

Then:

```bash
cd verticals/ticket-intelligence
python ml/ticket_intelligence/data_audit.py
python ml/ticket_intelligence/create_splits.py --input ../../data/processed/tickets_en_normalized.csv
python ml/ticket_intelligence/train_baseline.py
python ml/ticket_intelligence/evaluate.py
python ml/ticket_intelligence/feature_inspection.py
python ml/ticket_intelligence/error_analysis.py
python ml/ticket_intelligence/threshold_sweep.py
python ml/ticket_intelligence/predict.py
```

Synthetic data is only for smoke tests:

```bash
python ml/ticket_intelligence/train_baseline.py --smoke-test
```

## Outputs

Model artifacts:

- `ml/ticket_intelligence/artifacts/category_model.pkl`
- `ml/ticket_intelligence/artifacts/priority_model.pkl`
- `ml/ticket_intelligence/artifacts/model_metadata.json`

Evaluation and analysis:

- `ml/ticket_intelligence/outputs/metrics.json`
- `ml/ticket_intelligence/outputs/category_classification_report.csv`
- `ml/ticket_intelligence/outputs/priority_classification_report.csv`
- `ml/ticket_intelligence/outputs/category_confusion_matrix.png`
- `ml/ticket_intelligence/outputs/priority_confusion_matrix.png`
- `ml/ticket_intelligence/outputs/top_features_by_category.csv`
- `ml/ticket_intelligence/outputs/top_features_by_priority.csv`
- `ml/ticket_intelligence/outputs/error_analysis.csv`
- `ml/ticket_intelligence/outputs/threshold_sweep.csv`
- `ml/ticket_intelligence/outputs/predictions.jsonl`

## Current Baseline Metrics

Fixed English-only test split:

- Train rows: `16,623`
- Validation rows: `3,562`
- Test rows: `3,563`

Category model:

- Accuracy: `0.4415`
- Macro-F1: `0.4133`
- Weighted-F1: `0.4441`

Priority model:

- Accuracy: `0.5240`
- Macro-F1: `0.5048`
- Weighted-F1: `0.5236`

Priority is a text-only baseline. In real operations workflows, priority often depends on metadata such as SLA, account tier, incident severity, recurrence, and business impact.

## Notebook

The notebook is a portfolio walkthrough, not the source of truth:

- `notebooks/ticket_intelligence_walkthrough.ipynb`

Scripts are the canonical pipeline.

## Responsible AI

The router sends uncertain, high-risk, or policy-sensitive tickets to human/supervisor review. OpsPilot Ticket Intelligence should not make final refund, legal, compliance, account, or customer-facing decisions without human oversight.

## Future Work

- confidence calibration
- transformer benchmark
- multilingual extension with language-specific models or XLM-R/multilingual DistilBERT
- policy retrieval/RAG
- FastAPI integration
- human review database tables
- monitoring for drift, override rate, and escalation misses
