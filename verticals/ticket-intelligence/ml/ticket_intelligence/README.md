# OpsPilot Ticket Intelligence

AI-assisted triage and escalation-risk routing for messy operations tickets.

## Problem

Operations teams receive noisy tickets that mix billing disputes, account access issues, shipping failures, technical incidents, cancellations, and compliance questions. The goal is not to fully automate support. The goal is to classify the ticket, estimate risk, and route uncertain or sensitive cases to the right human queue quickly.

## Solution

This module predicts:

- `category`
- `priority`
- `confidence`
- `escalation_risk`
- `routing_decision`
- `reason`
- `model_version`

The current production-style vertical slice uses a baseline model for category and priority, applies a conservative low-confidence lexical fallback for obvious category cues, then applies deterministic confidence and escalation-risk routing rules.

## Architecture

```text
Ticket CRUD backend
        |
        v
Ticket Intelligence Model
        |
        v
Confidence + Risk Router
        |
        v
Human Review Decision
        |
        v
Evaluation / Model Card / README
```

## Model Layers

### Baseline

- TF-IDF features
- Logistic regression classifier
- Predicts category and priority
- Saves model artifacts as `.pkl`

### Transformer-Ready Layer

`train_transformer.py` is prepared for DistilBERT/MiniLM-style fine-tuning. It checks dependencies and writes a skipped status when `torch`, `transformers`, `datasets`, or `evaluate` are not installed.

## Routing Logic

The router combines model confidence with operational risk signals:

- angry language
- refund, billing, or dispute terms
- urgency and business impact
- repeat issue language
- legal and compliance terms
- sensitive categories

When category confidence is very low, the prediction layer can override the raw model category only if multiple category-specific keywords point to a different class. The raw model category and fallback source are retained in metadata for auditability.

Routing decisions include:

- `auto_triage_suggestion`
- `human_review`
- `priority_queue`
- `supervisor_review`

## Run

From `verticals/ticket-intelligence/`:

```bash
python ml/ticket_intelligence/data_audit.py
python ml/ticket_intelligence/create_splits.py
python ml/ticket_intelligence/train_baseline.py
python ml/ticket_intelligence/evaluate.py
python ml/ticket_intelligence/predict.py "I was charged twice and need a refund today."
uvicorn app.api.main:app --reload
```

The main training path expects real normalized split files:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

`data_audit.py` scans repository CSV/TSV/JSON/JSONL files and identifies the largest real ticket dataset with usable text/category columns. `create_splits.py` writes fixed train/validation/test splits with `random_state=42` and category stratification where possible.

Audit outputs are saved to:

- `data/processed/data_audit.json`
- `data/processed/data_audit_summary.csv`
- `data/processed/seed_load_scripts.csv`

The 60-row `ml/ticket_intelligence/data/synthetic_tickets.csv` file is now reserved for smoke tests and README demos:

```bash
python ml/ticket_intelligence/train_baseline.py --smoke-test
python ml/ticket_intelligence/train_transformer.py --smoke-test
python ml/ticket_intelligence/create_splits.py --allow-synthetic-smoke
```

When `create_splits.py --allow-synthetic-smoke` is used, synthetic splits are written to `data/smoke_processed/` instead of the real `data/processed/` training location.

Notebook workflow:

```bash
jupyter notebook notebooks/ticket_intelligence_walkthrough.ipynb
```

The notebook keeps the ML pipeline in one place so you can edit normalization, model parameters, thresholds, and routing logic cell by cell.

Example API request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_message":"I was charged twice and support keeps closing my tickets. If this refund is not handled today I am escalating to legal."}'
```

## Artifacts

- `outputs/metrics.json`
- `outputs/confusion_matrix.png`
- `outputs/error_analysis.csv`
- `outputs/predictions.jsonl`
- `model_card.md`

## Current Baseline Results

The checked-in starter run was produced from the synthetic smoke dataset before real training data was available:

- Category accuracy: `0.60`
- Category macro-F1: `0.60`
- Priority accuracy: `0.4667`
- Priority macro-F1: `0.2741`

These numbers are smoke-test benchmarks only. Main model results should be regenerated after adding a real normalized dataset and creating `data/processed/train.csv`, `val.csv`, and `test.csv`.

## Responsible AI

OpsPilot Ticket Intelligence is designed for human-in-the-loop operations. It should not make final refund, cancellation, legal, compliance, or account enforcement decisions. Low-confidence, high-risk, and policy-sensitive tickets are routed to humans by default.

## Roadmap

- Replace synthetic data with historical ticket exports.
- Fine-tune the transformer layer and compare macro-F1 against the baseline.
- Calibrate confidence scores.
- Add priority as a true multi-task transformer head.
- Store predictions in the main OpsPilot database.
- Add monitoring for drift, override rates, and escalation misses.
