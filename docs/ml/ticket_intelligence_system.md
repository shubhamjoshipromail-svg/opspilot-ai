# Ticket Intelligence System

Last reviewed: 2026-07-12

Ticket Intelligence is the first major ML capability in OpsPilot. It should be
preserved carefully because it contains both the product runtime integration and
the research/training history.

## Product Runtime

The canonical product runtime is implemented in:

- `backend/app/routes/ticket_intelligence.py`
- `backend/app/services/ticket_intelligence.py`
- `backend/app/services/routing_predictions.py`
- `backend/app/schemas/ticket_intelligence.py`
- `backend/app/models/ticket_routing_prediction.py`

Runtime endpoint:

```text
POST /api/ticket-intelligence/route
```

Input:

```json
{
  "subject": "Invoice payment issue",
  "body": "I was charged twice for my subscription."
}
```

Output fields:

- `predicted_queue`
- `confidence`
- `decision`
- `threshold`
- `probabilities`
- `model_id`
- `latency_ms`
- `timestamp`

The default decision threshold is `0.80`.

- Confidence `>= 0.80`: `auto_route`
- Confidence `< 0.80`: `human_review`

The threshold can be changed with `OPSPILOT_ROUTING_THRESHOLD`. Invalid values
fall back to `0.80`, and valid values are clamped to `[0.0, 1.0]`.

## Hosted Runtime Model

The canonical runtime model ID is:

```text
shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1
```

The backend loads the tokenizer and sequence-classification model from Hugging
Face, caches the loaded model, uses CUDA when available, and CPU otherwise. If
the model or tokenizer cannot be loaded, the API returns HTTP `503` and does not
persist a successful prediction.

Supported runtime queue labels:

- `billing_and_payments`
- `customer_general`
- `human_resources`
- `returns_and_exchanges`
- `sales_and_pre_sales`
- `service_outages_and_maintenance`
- `technical_product_support`

The response schema requires the probability map to contain exactly these seven
labels.

## Product Persistence

Successful predictions are stored in `ticket_routing_predictions` with:

- optional `ticket_id`
- `subject`
- `body`
- `predicted_queue`
- `confidence`
- `threshold`
- `decision`
- full `probabilities` JSON
- `model_id`
- `latency_ms`
- `created_at`

Model-unavailable failures are intentionally not saved as successful prediction
records.

## Research And Training Lab

The research/training code lives under:

```text
verticals/ticket-intelligence/ml/ticket_intelligence/
```

Important scripts:

- `data_audit.py`
- `create_splits.py`
- `train_baseline.py`
- `evaluate.py`
- `feature_inspection.py`
- `error_analysis.py`
- `threshold_sweep.py`
- `predict.py`
- `routing.py`
- `train_modernbert.py`
- `train_tensorflow_category.py`
- `train_tensorflow_priority.py`
- `train_tensorflow_risk.py`
- `train_transformer.py`

The local baseline artifacts are intentionally ignored by Git:

```text
verticals/ticket-intelligence/ml/ticket_intelligence/artifacts/
verticals/ticket-intelligence/ml/ticket_intelligence/outputs/
```

Local artifact inventory observed during audit:

| File | Approx size | Purpose |
| --- | ---: | --- |
| `artifacts/category_model.pkl` | 596 KB | TF-IDF + Logistic Regression category model |
| `artifacts/priority_model.pkl` | 320 KB | TF-IDF + Logistic Regression priority model |
| `artifacts/model_metadata.json` | 8 KB | Baseline metadata and validation metrics |
| `outputs/metrics.json` | 8 KB | Fixed test-set metrics |
| `outputs/error_analysis.csv` | 1.4 MB | Error analysis output |
| `outputs/risk_label_audit.csv` | 24 MB | Weak-risk label audit |

These artifacts should not be deleted. They should also not be committed unless
the project intentionally changes its artifact policy.

## Verified Local Baseline Metrics

The following values were read from local
`verticals/ticket-intelligence/ml/ticket_intelligence/outputs/metrics.json`.

| Task | Accuracy | Macro-F1 | Weighted-F1 | Test rows |
| --- | ---: | ---: | ---: | ---: |
| Category | 0.4409 | 0.4121 | 0.4430 | 3,563 |
| Priority | 0.5240 | 0.5042 | 0.5235 | 3,563 |

These metrics describe the local 10-label TF-IDF baseline, not the hosted
seven-label ModernBERT runtime model.

Local category labels:

- `billing_and_payments`
- `customer_service`
- `general_inquiry`
- `human_resources`
- `it_support`
- `product_support`
- `returns_and_exchanges`
- `sales_and_pre_sales`
- `service_outages_and_maintenance`
- `technical_support`

Local priority labels:

- `high`
- `low`
- `medium`

## Data Leakage Rules

- `customer_message` and `model_text` are valid model inputs.
- `model_text` includes subject/body plus non-outcome metadata such as ticket
  type and tags.
- `answer` and `reference_answer` are post-resolution fields and must not be
  used as inputs for category, priority, risk, or routing prediction.
- `reference_answer` may be useful later for response-quality evaluation or
  suggested-response retrieval.

## How To Run Runtime Checks

Run tests from the repository root:

```bash
pytest
```

Run the backend:

```bash
uvicorn backend.app.main:app --reload
```

Call the routing endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/ticket-intelligence/route \
  -H 'Content-Type: application/json' \
  -d '{"subject":"Invoice payment issue","body":"I was charged twice for my subscription."}'
```

## How To Retrain Intentionally

The local ML pipeline should be run intentionally and reviewed before replacing
any runtime model:

```bash
python scripts/build_ticket_dataset.py
cd verticals/ticket-intelligence
python ml/ticket_intelligence/data_audit.py
python ml/ticket_intelligence/create_splits.py --input ../../data/processed/tickets_en_normalized.csv
python ml/ticket_intelligence/train_baseline.py
python ml/ticket_intelligence/evaluate.py
python ml/ticket_intelligence/feature_inspection.py
python ml/ticket_intelligence/error_analysis.py
python ml/ticket_intelligence/threshold_sweep.py
```

Do not present smoke-test or synthetic-data metrics as production results.

## Preservation Rules

- Do not delete, overwrite, or retrain the existing model artifacts as part of
  cleanup work.
- Do not change the runtime endpoint or response schema without updating tests
  and downstream frontend usage.
- Do not rename the seven runtime labels without updating the hosted model,
  schema validation, dashboard display, and tests.
- Keep generated datasets and artifacts ignored unless there is a deliberate
  artifact-versioning decision.
