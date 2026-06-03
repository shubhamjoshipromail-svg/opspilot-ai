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

- Train rows: `16,622`
- Validation rows: `3,562`
- Test rows: `3,563`

Category model:

- Accuracy: `0.4409`
- Macro-F1: `0.4121`
- Weighted-F1: `0.4430`

Priority model:

- Accuracy: `0.5240`
- Macro-F1: `0.5042`
- Weighted-F1: `0.5235`

Priority is a text-only baseline. In real operations workflows, priority often depends on metadata such as SLA, account tier, incident severity, recurrence, and business impact.

## Notebook

The notebook is a portfolio walkthrough, not the source of truth:

- `notebooks/ticket_intelligence_walkthrough.ipynb`

Scripts are the canonical pipeline.

## Running TensorFlow Training in Colab

The TensorFlow workflow is designed for Google Colab GPU training. It does not replace the TF-IDF + Logistic Regression baseline; it adds deep learning experiments for comparison.

Notebook:

- `notebooks/tensorflow_colab_training.ipynb`

In Colab:

1. Open the notebook.
2. Set `Runtime > Change runtime type > GPU`.
3. Clone the repo or upload the repo folder.
4. Run the cells from top to bottom.

The notebook installs:

```bash
tensorflow pandas numpy scikit-learn matplotlib datasets joblib
```

The notebook runs:

```bash
python scripts/build_ticket_dataset.py
cd verticals/ticket-intelligence
python ml/ticket_intelligence/create_splits.py --input ../../data/processed/tickets_en_normalized.csv
python ml/ticket_intelligence/train_tensorflow_category.py --epochs 10 --batch-size 64
python ml/ticket_intelligence/train_tensorflow_priority.py --epochs 10 --batch-size 64
python ml/ticket_intelligence/create_risk_labels.py
python ml/ticket_intelligence/train_tensorflow_risk.py --epochs 10 --batch-size 64
```

TensorFlow input feature:

- `model_text = subject + body + type + combined_tags`

Labels:

- Category: `queue` normalized as `true_category` / `category`
- Priority: `priority` normalized as `true_priority` / `priority`
- Optional risk: `high_risk_policy_label`, a weak policy-derived label

Leakage prevention:

- `answer` is saved only as `reference_answer`.
- `answer` / `reference_answer` is not included in `model_text`.
- `TextVectorization.adapt()` uses training text only.
- Validation data is used only for early stopping.
- Test data is used only once for final evaluation.

TensorFlow artifacts:

- `ml/ticket_intelligence/artifacts/tensorflow_category_model.keras`
- `ml/ticket_intelligence/artifacts/tensorflow_category_label_mapping.json`
- `ml/ticket_intelligence/artifacts/tensorflow_priority_model.keras`
- `ml/ticket_intelligence/artifacts/tensorflow_priority_label_mapping.json`
- `ml/ticket_intelligence/artifacts/tensorflow_risk_model.keras` if the optional risk model is run

TensorFlow outputs:

- `ml/ticket_intelligence/outputs/tensorflow_category_metrics.json`
- `ml/ticket_intelligence/outputs/tensorflow_priority_metrics.json`
- `ml/ticket_intelligence/outputs/tensorflow_risk_metrics.json` if run
- `ml/ticket_intelligence/outputs/model_comparison.json`
- classification reports, confusion matrices, and error analysis CSVs
- `ml/ticket_intelligence/outputs/risk_label_audit.csv`
- `ml/ticket_intelligence/outputs/risk_label_summary.json`

The notebook creates:

```text
tensorflow_training_results.zip
```

Download this zip from Colab or copy it to Google Drive. It contains the `.keras` models, label mappings, metrics, reports, confusion matrices, error analyses, risk-label audit files, model comparison, and ML report.

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
