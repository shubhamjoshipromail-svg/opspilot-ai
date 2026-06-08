# OpsPilot Ticket Intelligence — ML Results Report

## 1. Executive Summary

OpsPilot Ticket Intelligence v1 is an AI-assisted triage baseline for English-language support and operations tickets. It takes a messy customer ticket message, predicts a ticket category, predicts a priority level, estimates escalation risk with rules, and recommends a routing action such as human review, supervisor review, priority queue, or auto-triage suggestion.

The current committed baseline is an interpretable classic machine learning system using TF-IDF text features and Logistic Regression. A Colab-ready TensorFlow/Keras extension has also been added so category, priority, and optional weak-risk deep learning models can be trained on GPU without replacing the baseline benchmark.

Current test-set results:

| Task | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Category prediction | 0.4409 | 0.4121 | 0.4430 |
| Priority prediction | 0.5240 | 0.5042 | 0.5235 |

The results are reasonable for a first baseline, especially because the model only uses ticket text and does not yet use structured metadata such as ticket type, tags, channel, or customer/account context.

## 2. Dataset

The dataset source is `Tobi-Bueck/customer-support-tickets`, loaded from Hugging Face by `scripts/build_ticket_dataset.py`. This should be described as a public labeled customer-support ticket dataset or public benchmark/source dataset, not private real customer data.

The script downloads the dataset, uses the `train` split, filters to English rows where `language == "en"`, removes empty messages, removes exact duplicate `customer_message` rows, and writes two local regenerated files:

| File | Purpose |
|---|---|
| `data/raw/customer_support_tickets_en.csv` | Raw English-only export from the source dataset |
| `data/processed/tickets_en_normalized.csv` | Normalized ticket-intelligence dataset |

Those root `data/raw` and `data/processed` files are regenerated locally and gitignored. The fixed vertical splits are saved under `verticals/ticket-intelligence/data/processed/`.

Normalized dataset size:

| Dataset | Rows |
|---|---:|
| Normalized English dataset | 23,747 |
| Train split | 16,622 |
| Validation split | 3,562 |
| Test split | 3,563 |

The split strategy is `category_stratified` with `random_state=42`.

Available normalized columns:

| Column | Used by model? | Notes |
|---|---|---|
| `external_id` | No | Stable ticket identifier |
| `customer_message` | Yes | Main text input feature |
| `true_category` | Yes, label | Category/queue label |
| `true_priority` | Yes, label | Priority label |
| `status` | No | Currently constant/default metadata |
| `channel` | No | Currently metadata only |
| `source` | No | Dataset source |
| `ticket_type` | No | Useful future metadata feature |
| `language` | No | Used for filtering to English |
| `reference_answer` | No | Useful future response/evaluation field |
| `tags` | No | Useful future metadata feature |

The fixed split files also include canonical aliases:

| Column | Purpose |
|---|---|
| `category` | Normalized model label derived from `true_category` |
| `priority` | Normalized model label derived from `true_priority` |
| `ticket_id` | Internal identifier, usually copied from `external_id` |

Current normalized class distribution:

| Category | Rows |
|---|---:|
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

Priority distribution:

| Priority | Rows |
|---|---:|
| medium | 9,753 |
| high | 9,149 |
| low | 4,845 |

## 3. ML Task Definition

The pipeline trains two supervised text classification tasks:

| Task | Input | Label |
|---|---|---|
| Category classification | `customer_message` | `true_category` / `category` |
| Priority classification | `customer_message` | `true_priority` / `priority` |

For category classification, the model tries to learn which operational queue a ticket belongs to from the text alone. Example labels include `billing_and_payments`, `technical_support`, `product_support`, and `customer_service`.

For priority classification, the model tries to learn whether the ticket is `high`, `medium`, or `low` priority from the text alone.

The model does not currently learn escalation risk directly. Escalation risk and routing are handled by deterministic rules in `routing.py` using model confidence, predicted priority, policy-sensitive signals, and escalation keywords.

## 4. Models Used

The current committed baseline model family is classic machine learning. The TensorFlow/Keras scripts added for Colab are deep learning experiments that should be compared against this baseline after GPU training.

Two models are trained:

| Model | Target | Artifact |
|---|---|---|
| TF-IDF + Logistic Regression | Category | `ml/ticket_intelligence/artifacts/category_model.pkl` |
| TF-IDF + Logistic Regression | Priority | `ml/ticket_intelligence/artifacts/priority_model.pkl` |

TF-IDF settings:

| Setting | Value |
|---|---|
| `lowercase` | `true` |
| `ngram_range` | `[1, 2]` |
| `min_df` | `1` |
| `max_features` | `5000` |
| `strip_accents` | `unicode` |

Logistic Regression settings:

| Setting | Value |
|---|---|
| `max_iter` | `1000` |
| `class_weight` | `balanced` |
| `solver` | `liblinear` |
| `random_state` | `42` |

No transformer or fine-tuned language model is currently trained or used. TensorFlow/Keras training scripts are available for Colab GPU training, but they do not replace the committed TF-IDF baseline until their held-out metrics are reviewed.

Model artifacts are saved in:

```text
verticals/ticket-intelligence/ml/ticket_intelligence/artifacts/
```

The metadata file is:

```text
verticals/ticket-intelligence/ml/ticket_intelligence/artifacts/model_metadata.json
```

## 5. Features Used

The ML models currently use only one input feature:

```text
customer_message
```

The text is converted into TF-IDF features using single words and two-word phrases. No structured metadata is used yet. That means the current model ignores `ticket_type`, `tags`, `channel`, `status`, `reference_answer`, and any customer/account context.

Top learned category features show that the model is picking up sensible lexical signals:

| Category | Example top features |
|---|---|
| `billing_and_payments` | `billing`, `payment`, `pricing`, `invoice`, `charges`, `subscription` |
| `it_support` | `website`, `network`, `internet`, `upgrade`, `digital tools`, `connection` |
| `service_outages_and_maintenance` | `service`, `maintenance`, `outage`, `outages`, `disruption`, `servers` |
| `technical_support` | `technical`, `crashes`, `postgresql`, `website`, `firewall` |
| `returns_and_exchanges` | `returns`, `return`, `products`, `product`, `incorrect` |
| `sales_and_pre_sales` | `sales`, `solutions`, `pricing`, `financial`, `feature` |

Top learned priority features:

| Priority | Example top features |
|---|---|
| `high` | `critical`, `technical`, `immediate`, `system`, `urgent`, `outage`, `loss` |
| `low` | `sporadic`, `outdated`, `alert`, `intermittent`, `desktop` |
| `medium` | `agency has`, `customer interaction`, `import`, `connections and`, `antivirus` |

Overall, the category model appears to learn obvious queue-specific words well, especially for billing and outages. The priority model learns urgency and impact words, but priority is harder because true priority often depends on metadata, business impact, SLA rules, customer tier, or operational context that is not present in `customer_message`.

## 5A. Rich TensorFlow Feature Design

The TensorFlow Colab workflow uses a richer non-leaky text feature named `model_text`.

Source fields available from `Tobi-Bueck/customer-support-tickets` include:

```text
subject, body, answer, type, queue, priority, language, version, tag_1 ... tag_8
```

The regenerated normalized dataset now creates:

| Field | Definition |
|---|---|
| `customer_message` | `subject + "\n\n" + body` |
| `combined_tags` | Non-empty `tag_1` through `tag_8`, joined with comma separators |
| `model_text` | `subject + "\n\n" + body + "\n\nType: " + type + "\nTags: " + combined_tags` |
| `true_category` | Source `queue` |
| `true_priority` | Source `priority` |
| `reference_answer` | Source `answer` |

The TensorFlow scripts use `model_text` as input. This gives the neural models more useful signal than message body alone while still avoiding leakage.

`answer` / `reference_answer` is intentionally excluded from `model_text`. The answer is created after ticket handling, so using it to predict category, priority, or risk would leak future information into training. It is preserved only for future response drafting or reference-answer evaluation.

## 6. Evaluation Results

Final test metrics from `outputs/metrics.json`:

| Task | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Category | 0.4409 | 0.4121 | 0.4430 |
| Priority | 0.5240 | 0.5042 | 0.5235 |

Category per-class highlights:

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `billing_and_payments` | 0.7464 | 0.7052 | 0.7252 | 363 |
| `service_outages_and_maintenance` | 0.4769 | 0.6596 | 0.5536 | 141 |
| `technical_support` | 0.5172 | 0.4976 | 0.5072 | 1,029 |
| `product_support` | 0.4301 | 0.3098 | 0.3601 | 665 |
| `customer_service` | 0.3830 | 0.3451 | 0.3631 | 536 |
| `general_inquiry` | 0.2079 | 0.4118 | 0.2763 | 51 |

Priority per-class results:

| Priority | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `high` | 0.5663 | 0.5647 | 0.5655 | 1,353 |
| `medium` | 0.5382 | 0.5483 | 0.5432 | 1,490 |
| `low` | 0.4109 | 0.3972 | 0.4040 | 720 |

These results are not production-strong, but they are reasonable for an initial baseline. The model is solving 10-way category classification and 3-way priority classification using text only. The data is imbalanced, and several classes overlap conceptually. For example, `technical_support`, `it_support`, and `product_support` can contain similar technical language.

The strongest category is `billing_and_payments`, likely because billing language has distinctive terms like invoice, payment, charges, and subscription. The weaker categories are broader or smaller classes such as `general_inquiry`, `human_resources`, and `sales_and_pre_sales`.

## 7. Confusion Matrix Summary

The largest category confusions are:

| True category | Predicted category | Count |
|---|---|---:|
| `product_support` | `technical_support` | 162 |
| `it_support` | `technical_support` | 116 |
| `technical_support` | `customer_service` | 103 |
| `customer_service` | `technical_support` | 102 |
| `technical_support` | `it_support` | 101 |
| `technical_support` | `product_support` | 100 |
| `product_support` | `customer_service` | 75 |
| `customer_service` | `product_support` | 70 |

This shows that the model struggles most where support queues naturally overlap. Technical, IT, product, and customer-service tickets often share vocabulary such as issue, support, system, product, website, account, or access.

The largest priority confusions are:

| True priority | Predicted priority | Count |
|---|---|---:|
| `high` | `medium` | 436 |
| `medium` | `high` | 416 |
| `low` | `medium` | 265 |
| `medium` | `low` | 257 |
| `low` | `high` | 169 |
| `high` | `low` | 153 |

Priority is especially noisy because text alone is not enough to know business urgency. A short message can describe a severe incident, and a long message can describe a low-priority request. Priority likely needs metadata and rules in addition to text classification.

## 8. Error Analysis

`outputs/error_analysis.csv` contains 2,680 rows where either category, priority, or both were incorrect.

Error type distribution:

| Error type | Count |
|---|---:|
| `low_confidence` | 2,483 |
| `priority_depends_on_metadata` | 75 |
| `possible_label_noise` | 52 |
| `ambiguous_text` | 27 |
| `label_overlap` | 24 |
| `multiple_intents` | 17 |
| `insufficient_examples` | 2 |

The dominant pattern is low confidence, which supports the decision to use human review routing. The model often knows it is uncertain, especially on tickets that mix multiple topics or use broad business language.

Representative errors:

| Ticket | True category | Predicted category | True priority | Predicted priority | Confidence notes | Error type |
|---|---|---|---|---|---|---|
| `hf-tobi-005476` | `product_support` | `returns_and_exchanges` | `medium` | `medium` | Category confidence 0.2158 | `low_confidence` |
| `hf-tobi-018348` | `returns_and_exchanges` | `product_support` | `low` | `medium` | Category confidence 0.2122 | `low_confidence` |
| `hf-tobi-012549` | `billing_and_payments` | `billing_and_payments` | `medium` | `high` | Priority confidence 0.4009 | `priority_depends_on_metadata` |
| `hf-tobi-012221` | `it_support` | `technical_support` | `high` | `medium` | Category confidence 0.3705 | `low_confidence` |
| `hf-tobi-012189` | `service_outages_and_maintenance` | `sales_and_pre_sales` | `high` | `low` | Priority confidence 0.3534 | `low_confidence` |

The mistakes are mostly explainable:

- Some labels overlap, especially technical/product/IT/customer-service tickets.
- Some ticket text is ambiguous or contains multiple intents.
- Priority often depends on missing metadata such as customer tier, SLA, outage scope, revenue impact, or historical repeats.
- Some labels may be noisy or benchmark-style rather than perfectly operational.

## 9. Threshold Sweep / Human Review Tradeoff

Thresholding means the system only allows auto-triage suggestions when model confidence is above a chosen cutoff. Lower thresholds automate more tickets but risk more mistakes. Higher thresholds send more tickets to human review but improve precision on the small set that remains automated.

Current threshold sweep:

| Threshold | Auto-triage rate | Human-review rate | Accuracy on auto-triaged |
|---:|---:|---:|---:|
| 0.50 | 0.0286 | 0.9714 | 0.7843 |
| 0.55 | 0.0182 | 0.9818 | 0.8615 |
| 0.60 | 0.0143 | 0.9857 | 0.9412 |
| 0.65 | 0.0104 | 0.9896 | 0.9730 |
| 0.70 | 0.0056 | 0.9944 | 1.0000 |
| 0.75 | 0.0017 | 0.9983 | 1.0000 |
| 0.80 | 0.0003 | 0.9997 | 1.0000 |
| 0.85+ | 0.0000 | 1.0000 | N/A |

For a demo, a threshold around `0.60` to `0.65` is reasonable because it shows the intended operating pattern: the model can auto-suggest only when it is very confident, while most uncertain tickets go to human review.

The low auto-triage rate is not necessarily a weakness. In an operations setting, human-in-the-loop routing is valuable because it prevents the system from pretending to be certain on messy or high-risk tickets.

## 10. Routing and Risk Logic

`routing.py` combines model outputs with rule-based risk scoring.

The router considers:

- Category prediction
- Priority prediction
- Category confidence
- Priority confidence
- Escalation-risk keywords
- Policy-sensitive keywords
- Sensitive categories

Escalation keywords include:

```text
legal, lawsuit, chargeback, fraud, regulator, complaint, angry, cancel,
refund, charged twice, duplicate charge, manager, escalate, escalating,
urgent, today, account locked, cannot access, safety, compliance
```

Policy-sensitive keywords include:

```text
legal, lawsuit, regulator, compliance, privacy, gdpr, hipaa, fraud,
chargeback, safety, contract
```

Sensitive categories include:

```text
billing_and_payments, returns_and_exchanges, human_resources
```

Routing rules:

| Condition | Route |
|---|---|
| Escalation risk >= 0.80 | `human_review` |
| Policy-sensitive language/category | `supervisor_review` |
| Category confidence < 0.65 | `human_review` |
| High-priority prediction with sufficient confidence | `priority_queue` |
| Otherwise | `auto_triage_suggestion` |

The model does not make final decisions. It supports triage by producing a recommendation and routing high-risk or uncertain cases to a human.

## 11. Structured Prediction Example

Latest example from `outputs/predictions.jsonl`:

```json
{
  "ticket_id": "demo-001",
  "customer_message": "I was charged twice and support keeps closing my tickets. If this refund is not handled today I am escalating to legal.",
  "category": {
    "prediction": "it_support",
    "confidence": 0.2764,
    "raw_model_prediction": "it_support",
    "fallback_applied": false
  },
  "priority": {
    "prediction": "medium",
    "confidence": 0.3725
  },
  "risk": {
    "escalation_risk": 0.87,
    "risk_signals": [
      "low_category_confidence",
      "low_priority_confidence",
      "medium_priority",
      "policy:legal",
      "risk_keyword:charged twice",
      "risk_keyword:escalating",
      "risk_keyword:legal",
      "risk_keyword:refund",
      "risk_keyword:today"
    ],
    "policy_sensitive": true
  },
  "routing": {
    "decision": "human_review",
    "reason": "High escalation risk requires human review before action"
  },
  "model_metadata": {
    "model_version": "ticket-intelligence-v1",
    "model_type": "tfidf_logistic_regression_baseline",
    "dataset_source": "Tobi-Bueck/customer-support-tickets"
  }
}
```

Plain-English interpretation: the classifier is not confident and predicts `it_support`, which is probably not the best category for this billing/refund-style ticket. However, the risk router correctly catches several escalation signals: charged twice, refund, today, escalating, and legal. Because escalation risk is high, the system routes the ticket to `human_review`. This is exactly why the pipeline combines model confidence with risk rules instead of relying only on the classifier.

## 12. Current Limitations

- The model uses only `customer_message` text.
- It does not use structured metadata such as `ticket_type`, `tags`, `channel`, `status`, account history, SLA, customer tier, or previous ticket count.
- No transformer model is currently trained or used.
- No Railway/Postgres/FastAPI integration is active in this ML pipeline yet.
- Priority prediction is likely underpowered because priority often depends on metadata and policy rules.
- The dataset is a public labeled benchmark/source dataset, not private real customer data.
- The current scope is English-only v1.
- Some categories overlap heavily, especially `technical_support`, `it_support`, `product_support`, and `customer_service`.
- `predictions.jsonl` contains older historical demo rows plus the latest current-schema row, so the latest line should be treated as the current structured prediction example.

## 13. What This Project Demonstrates

Even though the current model is intentionally simple, the project demonstrates a serious ML product workflow:

- Reproducible dataset regeneration from a public source.
- English-only filtering and normalized schema creation.
- Fixed train/validation/test splits with `random_state=42`.
- Interpretable baseline model before adding complex models.
- Separate category and priority classifiers.
- Saved model artifacts and model metadata.
- Accuracy, macro-F1, weighted-F1, per-class reports, and confusion matrices.
- Feature inspection for model explainability.
- Error analysis with likely failure reasons.
- Confidence threshold sweep for automation versus human review.
- Rule-based risk scoring for escalation-sensitive tickets.
- Human-in-the-loop routing and responsible AI framing.
- Model card and documentation around intended use and limitations.

This is stronger than a notebook-only classifier because it shows the surrounding operational system: data pipeline, evaluation, artifacts, model versioning, risk routing, and oversight.

## 14. Recommended Next Steps

Immediate next step:

- Inspect this report and verify that the current baseline results and limitations are understood before adding new modeling work.

Next ML improvement:

- Add metadata features if available, especially `ticket_type`, `tags`, and possibly source/channel fields.
- Improve priority prediction as a hybrid model plus rules problem rather than text-only classification.
- Later, compare this baseline against a lightweight transformer model such as DistilBERT, MiniLM, or another small encoder.

Next product integration:

- Expose `predict.py` through a FastAPI endpoint later.
- Optionally load sample tickets and predictions into Railway/Postgres for an app demo.
- Keep human review as the default for low-confidence or high-risk cases.

## 15. TensorFlow Deep Learning Category Model

The Colab workflow adds `ml/ticket_intelligence/train_tensorflow_category.py`.

Task:

```text
model_text -> category
```

Preprocessing:

- `TextVectorization(max_tokens=30000, output_sequence_length=250)`
- `TextVectorization.adapt()` runs only on training `model_text`.
- Validation text is not used to fit preprocessing.
- Test text is used only once for final evaluation.

Model architecture:

```text
TextVectorization
Embedding(input_dim=30000, output_dim=128)
Conv1D(128, kernel_size=5, activation="relu")
GlobalMaxPooling1D
Dropout(0.3)
Dense(64, activation="relu")
Dropout(0.3)
Dense(num_category_classes, activation="softmax")
```

Training setup:

- Optimizer: Adam
- Loss: sparse categorical crossentropy
- Metric: accuracy
- Batch size: 64 by default
- Max epochs: 10
- Early stopping on validation loss with `restore_best_weights=True`

Outputs after running in Colab:

```text
ml/ticket_intelligence/artifacts/tensorflow_category_model.keras
ml/ticket_intelligence/artifacts/tensorflow_category_label_mapping.json
ml/ticket_intelligence/outputs/tensorflow_category_metrics.json
ml/ticket_intelligence/outputs/tensorflow_category_classification_report.csv
ml/ticket_intelligence/outputs/tensorflow_category_confusion_matrix.png
ml/ticket_intelligence/outputs/tensorflow_category_error_analysis.csv
```

The deep model should be compared against the TF-IDF category baseline in `outputs/model_comparison.json`. It should not be assumed better until the held-out test metrics prove it.

## 16. TensorFlow Deep Learning Priority Model

The Colab workflow also adds `ml/ticket_intelligence/train_tensorflow_priority.py`.

Task:

```text
model_text -> priority
```

It uses the same Keras text architecture as the category model, but the output classes are source priority labels such as `high`, `medium`, and `low`.

Priority is supervised because the dataset includes priority labels. However, priority may still be noisy or context-dependent because real operational priority often depends on SLA, customer tier, outage scope, account impact, recurrence, or business rules that are not fully represented in `model_text`.

Outputs after running in Colab:

```text
ml/ticket_intelligence/artifacts/tensorflow_priority_model.keras
ml/ticket_intelligence/artifacts/tensorflow_priority_label_mapping.json
ml/ticket_intelligence/outputs/tensorflow_priority_metrics.json
ml/ticket_intelligence/outputs/tensorflow_priority_classification_report.csv
ml/ticket_intelligence/outputs/tensorflow_priority_confusion_matrix.png
ml/ticket_intelligence/outputs/tensorflow_priority_error_analysis.csv
```

## 17. Weak Risk Labels

The workflow adds `ml/ticket_intelligence/create_risk_labels.py`.

It creates candidate weak labels:

- `urgency_signal`
- `policy_sensitive_signal`
- `escalation_keyword_signal`
- `high_risk_policy_label`

Signals come from fields such as `model_text`, `type`, `priority`, `queue/category`, and `combined_tags`. Rules include high priority, incident type, security/outage/compliance/legal/fraud tags, and text keywords such as legal, lawsuit, chargeback, fraud, regulator, complaint, urgent, outage, data breach, account locked, cannot access, safety, compliance, refund, duplicate charge, and charged twice.

Outputs:

```text
ml/ticket_intelligence/outputs/risk_label_audit.csv
ml/ticket_intelligence/outputs/risk_label_summary.json
```

These are weak policy-derived labels, not real observed escalation outcomes. They are useful for controlled experiments and routing-risk analysis, but they should not be presented as ground truth customer escalation.

## 18. Optional TensorFlow Risk Model

The optional risk experiment is `ml/ticket_intelligence/train_tensorflow_risk.py`.

Task:

```text
model_text -> high_risk_policy_label
```

This is binary classification. The model predicts whether a ticket matches the weak policy-derived high-risk label. It does not predict true escalation because the source dataset does not contain observed escalation outcomes.

Outputs after running in Colab:

```text
ml/ticket_intelligence/artifacts/tensorflow_risk_model.keras
ml/ticket_intelligence/outputs/tensorflow_risk_metrics.json
ml/ticket_intelligence/outputs/tensorflow_risk_classification_report.csv
ml/ticket_intelligence/outputs/tensorflow_risk_confusion_matrix.png
ml/ticket_intelligence/outputs/tensorflow_risk_error_analysis.csv
```

## 19. TensorFlow Colab Workflow

Notebook path:

```text
verticals/ticket-intelligence/notebooks/tensorflow_colab_training.ipynb
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

It then writes:

```text
tensorflow_training_results.zip
```

The zip includes `.keras` model files, label mappings, metrics JSON, classification reports, confusion matrices, error analyses, risk-label audit files, model comparison JSON, and this ML report.

## 20. What The Final Product Still Needs

- Compare TensorFlow metrics against the existing TF-IDF baseline after Colab training finishes.
- Decide whether the neural model is actually better before replacing any default baseline.
- Calibrate confidence thresholds if TensorFlow predictions are used for routing.
- Add real escalation outcomes before treating risk prediction as a true escalation model.
- Keep human review for low-confidence, policy-sensitive, and high-risk tickets.
- Consider transformers only after this TensorFlow extension is measured and understood.

## 21. V2 Transformer Benchmark: ModernBERT Parent Queue

V2 adds a transformer fine-tuning benchmark as an additive upgrade. It does not replace v1. The purpose is to compare a serious pretrained transformer against the existing TF-IDF and TensorFlow CNN models using the same fixed train/validation/test splits.

First v2 task:

```text
parent_queue classification
```

Input:

```text
subject + "\n\n" + body
```

Target:

```text
category / true_category
```

The script uses `category` when available because it is the normalized parent queue label. If the dataset schema changes, it can fall back to `true_category` or another configured task label.

The input deliberately excludes:

```text
answer
reference_answer
```

Those fields are post-resolution answers, so using them to predict routing, priority, or risk would leak future information.

Primary model:

```text
answerdotai/ModernBERT-base
```

Fallbacks:

```text
microsoft/deberta-v3-base
microsoft/deberta-v3-small
distilbert-base-uncased
```

Script:

```text
ml/ticket_intelligence/train_modernbert.py
```

The script supports:

- `--task parent_queue / priority / ticket_type / raw_queue`
- `--model-name`
- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--max-length`
- `--weight-decay`
- `--warmup-ratio`
- `--sample-size`
- `--output-dir`
- `--seed`

It prints a split audit before training:

- row counts
- columns
- target label counts
- missing values
- sample input text
- whitespace token-length estimates
- selected input and target columns

It is robust to transformer models that do or do not accept `token_type_ids`.

Expected outputs after Colab/GPU training:

```text
ml/ticket_intelligence/artifacts/modernbert_parent_queue/
ml/ticket_intelligence/outputs/modernbert_parent_queue_metrics.json
ml/ticket_intelligence/outputs/modernbert_parent_queue_classification_report.csv
ml/ticket_intelligence/outputs/modernbert_parent_queue_confusion_matrix.png
ml/ticket_intelligence/outputs/modernbert_parent_queue_predictions.csv
ml/ticket_intelligence/outputs/modernbert_parent_queue_error_analysis.csv
ml/ticket_intelligence/outputs/modernbert_parent_queue_run_summary.md
```

Local status: the local Anaconda environment used in this workspace does not currently have the full transformer stack installed, so the sample command fails clearly with install instructions. The intended training environment is Colab GPU.

## 22. Model Leaderboard

Leaderboard script:

```text
ml/ticket_intelligence/compare_models.py
```

Outputs:

```text
ml/ticket_intelligence/outputs/model_leaderboard.csv
ml/ticket_intelligence/outputs/model_leaderboard.json
```

The leaderboard reads metrics when available from:

- TF-IDF Logistic Regression category/priority baselines
- TensorFlow CNN category/priority models
- ModernBERT parent_queue model

Current committed leaderboard contains the available TF-IDF rows. TensorFlow and ModernBERT rows appear automatically after their metric JSON files are generated in Colab and copied back into the repo.

This should be described as a transformer-based ticket-routing benchmark and enterprise AI workflow prototype, not as production-ready automation.

## 23. V2 Experiment Upgrade Plan

The first full ModernBERT-base `parent_queue` result is stronger than v1 but still not strong enough for a final routing model:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| ModernBERT-base parent_queue | 0.5229 | 0.4725 | 0.5158 |

The next v2 experiments are designed to improve routing quality without deleting or overwriting existing results.

New trainer flags:

```bash
--run-name
--class-weighting none|balanced|sqrt_balanced
--label-map clean_v1
```

`--run-name` saves artifacts and outputs under unique names:

```text
artifacts/{run_name}/
outputs/{run_name}_metrics.json
outputs/{run_name}_classification_report.csv
outputs/{run_name}_predictions.csv
outputs/{run_name}_error_analysis.csv
outputs/{run_name}_confusion_matrix.png
```

`--class-weighting` adds weighted cross entropy:

- `none`: unweighted baseline
- `balanced`: inverse-frequency class weights
- `sqrt_balanced`: softened inverse-frequency weights

`--label-map clean_v1` merges overlapping support labels:

- `technical_support`, `it_support`, `product_support` -> `technical_product_support`
- `customer_service`, `general_inquiry` -> `customer_general`

This tests whether cleaner operational routing groups outperform the original noisy queue taxonomy.

Recommended bakeoff:

1. ModernBERT-base baseline
2. ModernBERT-base + `sqrt_balanced`
3. ModernBERT-large + `sqrt_balanced`
4. DeBERTa-v3-large + `sqrt_balanced`
5. ModernBERT-base + `clean_v1` + `sqrt_balanced`

Threshold analysis is now available:

```bash
python ml/ticket_intelligence/threshold_analysis.py \
  --predictions ml/ticket_intelligence/outputs/{run_name}_predictions.csv
```

It reports coverage, accuracy, and macro-F1 at confidence thresholds from 0.00 to 0.95. This supports the human-in-the-loop routing design by showing how many tickets can be auto-routed at each confidence level and how many should be sent to review.
