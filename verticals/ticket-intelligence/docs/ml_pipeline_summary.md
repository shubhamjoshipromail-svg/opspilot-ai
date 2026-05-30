# OpsPilot Ticket Intelligence ML Pipeline Summary

This document explains the current Ticket Intelligence vertical in plain English: what models exist, how data flows through them, what cleaning and normalization happens, current outcomes, and how to improve the pipeline over time.

## Current Goal

OpsPilot Ticket Intelligence is a decision-support layer for messy operations tickets. Given a raw customer message, it returns:

- predicted ticket category
- predicted priority
- escalation risk
- confidence score
- routing decision
- short routing explanation
- model/version metadata

The important design choice is that the ML model does **not** make every decision. The model predicts category and priority. Then rule-based risk and confidence logic routes the ticket to human review, supervisor review, a priority queue, or an auto-triage suggestion.

## Data

The current dataset is [synthetic_tickets.csv](../ml/ticket_intelligence/data/synthetic_tickets.csv). It has 60 rows and four columns:

- `ticket_id`: stable ticket identifier
- `customer_message`: messy customer-facing ticket text
- `category`: supervised category label
- `priority`: supervised priority label

Current categories:

- `account_access`
- `billing_dispute`
- `cancellation`
- `compliance_request`
- `shipping_delay`
- `technical_issue`

Current priorities:

- `low`
- `medium`
- `high`

This is a starter dataset, not production data. It exists so the full pipeline can run end to end. The next real improvement is to replace or augment it with historical ticket exports.

## Data Connection

The ML scripts and notebook load the CSV directly from:

```text
verticals/ticket-intelligence/ml/ticket_intelligence/data/synthetic_tickets.csv
```

The pipeline does not yet connect to the main FastAPI ticket database. That is intentional for this vertical slice: the model can be trained and evaluated offline first. In a later integration step, the backend can call the same prediction code after a ticket is created or updated.

## Cleaning And Normalization Used Today

Current baseline cleaning is intentionally light:

- `TfidfVectorizer(lowercase=True)` lowercases text inside the vectorizer.
- `strip_accents="unicode"` normalizes accented characters.
- `ngram_range=(1, 2)` captures unigrams and two-word phrases.
- No stemming or lemmatization is currently used.
- No stop-word removal is currently used.
- No aggressive punctuation stripping is used before modeling.

The routing layer separately normalizes text with:

```python
normalized = text.lower()
```

That lowercased text is used to detect risk keywords such as refund, dispute, urgent, legal, GDPR, repeat issue terms, and angry language.

## Why Cleaning Is Light

For short support tickets, too much cleaning can remove useful signal. Words like "not", "again", "today", "refund", "legal", "blocked", and "cancel" matter. A simple TF-IDF baseline often works better when the original text shape is mostly preserved.

That said, the notebook includes an optional `normalize_text()` function so you can experiment with:

- lowercasing
- Unicode normalization
- URL replacement
- email replacement
- number replacement
- punctuation spacing
- whitespace cleanup

## Baseline Model

The main trained model is:

```text
TF-IDF + Logistic Regression
```

There are two separate classifiers:

1. Category classifier: predicts `category`
2. Priority classifier: predicts `priority`

The pipeline is:

```text
customer_message
  -> TF-IDF vectorizer
  -> LogisticRegression(class_weight="balanced")
  -> predicted label + class probabilities
```

Important settings:

- `ngram_range=(1, 2)`
- `min_df=1`
- `max_features=5000`
- `strip_accents="unicode"`
- `class_weight="balanced"`
- `solver="liblinear"`
- `max_iter=1000`
- `random_state=42`

The train/test split is:

- 75% train
- 25% test
- stratified by category
- `random_state=42`

## Confidence

The baseline confidence is the maximum predicted probability from logistic regression. The final confidence returned by the prediction function is:

```python
min(category_confidence, priority_confidence)
```

That is conservative. If either category or priority is uncertain, the whole triage recommendation is treated as uncertain.

Important limitation: logistic regression probabilities are not calibrated yet. They are useful for rough routing thresholds, but they should be calibrated before production use.

## Low-Confidence Lexical Fallback

The prediction layer includes a small safety fallback:

If the category model confidence is below `0.40`, and multiple category-specific keywords strongly indicate a different category, the output category can be overridden.

Example: if the raw model predicts `technical_issue` with very low confidence, but the text contains terms like `charged`, `refund`, and `billing`, the final category may become `billing_dispute`.

The raw model output is still preserved in metadata:

- `raw_model_category`
- `category_source`
- `category_hint_score`
- `category_distribution`

This makes the fallback auditable instead of hidden.

## Escalation Risk Model

Escalation risk is rule-based today. It starts with a base score of `0.12`, then adds weight for signals:

- angry language: `+0.19`
- billing/refund dispute terms: `+0.15`
- urgency or business impact: `+0.18`
- repeat issue language: `+0.14`
- legal/compliance sensitivity: `+0.20`
- sensitive category: `+0.08`
- high predicted priority: `+0.12`
- medium predicted priority: `+0.05`

The final score is capped at `0.99`.

This is not a trained risk model yet. It is a transparent first version that works well for human-in-the-loop routing and makes it easy to explain why a ticket was flagged.

## Routing Logic

The router uses category, priority, confidence, and escalation risk.

Current decisions:

- `supervisor_review`: compliance-sensitive ticket with enough risk
- `human_review`: low confidence or high priority
- `priority_queue`: high escalation risk
- `auto_triage_suggestion`: high enough confidence and low risk

Current thresholds:

- low confidence threshold: `0.58`
- high risk threshold: `0.72`

## Current Outcomes

The checked-in baseline run uses 45 training rows and 15 test rows.

Category model:

- accuracy: `0.60`
- macro-F1: `0.60`
- weighted-F1: `0.5978`

Priority model:

- accuracy: `0.4667`
- macro-F1: `0.2741`
- weighted-F1: `0.3585`

Interpretation:

- Category prediction is a reasonable first benchmark for a tiny synthetic dataset.
- Priority prediction is weak and over-predicts `high`.
- The priority labels need more examples and clearer labeling rules.
- The confidence values are low, which is actually useful for this demo because the router sends uncertain cases to human review.

## What The Confusion Matrix Shows

The category model correctly predicts all three `account_access` test examples, but it confuses several other classes with `account_access`. This likely happens because the dataset is tiny and many messages share generic support language.

The priority model predicts most test examples as `high`. This is common when:

- the dataset is small
- class labels are imbalanced
- many messages contain urgency-like words
- the model has too little evidence to separate `medium` from `high`

## Transformer Layer

The repo includes [train_transformer.py](../ml/ticket_intelligence/train_transformer.py). It is transformer-ready but was not trained in this environment because `torch` and `transformers` were not installed.

The intended second layer is:

```text
DistilBERT or MiniLM sequence classifier
```

Input:

```text
customer_message
```

Output:

```text
category
```

Priority can be added later as:

- a second classifier
- a multi-task model head
- a rule-based mapping from category/risk/business impact

## How To Improve The Pipeline

### 1. Better Data

The biggest improvement will come from real data:

- export historical tickets
- remove duplicates
- keep original customer language
- add final resolved category
- add true priority based on SLA or business outcome
- include escalation outcome if available
- include metadata such as account tier, ticket age, repeat count, channel, and product area

### 2. Better Labeling

Priority needs a clearer definition. For example:

- `high`: production outage, legal threat, refund dispute with escalation, executive customer, SLA breach
- `medium`: important but not immediately business-blocking
- `low`: informational, admin, simple update

Without strict labeling rules, the model cannot learn stable priority boundaries.

### 3. Better Normalization

Try controlled normalization experiments:

- replace URLs with `<URL>`
- replace emails with `<EMAIL>`
- replace ticket/order/invoice numbers with `<NUMBER>`
- normalize repeated punctuation
- keep negation words
- keep domain terms
- compare with and without stop-word removal

Do not blindly stem or remove too much punctuation until metrics prove it helps.

### 4. Hyperparameter Tuning

Good baseline experiments:

- `ngram_range`: `(1, 1)`, `(1, 2)`, `(1, 3)`
- `max_features`: `1000`, `3000`, `5000`, `10000`
- `min_df`: `1`, `2`
- logistic regression `C`: `0.3`, `1`, `3`, `10`
- solvers: `liblinear`, `lbfgs`
- compare `class_weight=None` vs `balanced`

Also compare:

- Linear SVM
- Complement Naive Bayes
- calibrated logistic regression

### 5. Confidence Calibration

Use calibration before relying on thresholds:

- `CalibratedClassifierCV`
- reliability curves
- expected calibration error
- threshold tuning on validation data

The goal is for a `0.75` confidence prediction to be correct roughly 75% of the time.

### 6. Risk Model Upgrade

Move from fixed rules to supervised risk prediction once labels exist:

- target: escalated/not escalated
- features: text, category, priority, customer tier, repeat count, ticket age, sentiment, policy-sensitive terms
- models: logistic regression, gradient boosting, random forest, XGBoost/LightGBM if available

Keep rules as guardrails even after adding ML risk scoring.

### 7. Transformer Fine-Tuning

Once dependencies and data are available:

- start with `distilbert-base-uncased` or a small MiniLM classifier
- fine-tune category first
- compare macro-F1 against baseline
- use early stopping
- track per-class recall
- avoid overfitting tiny synthetic data

Transformer fine-tuning is most valuable after you have at least a few hundred labeled examples per major class.

### 8. Pretraining Or Domain Adaptation

If you eventually have lots of unlabeled ticket text:

- continue masked-language-model pretraining on internal ticket text
- then fine-tune on category labels
- compare against plain DistilBERT/MiniLM

This is useful when support language contains lots of domain-specific product terms, acronyms, and workflow phrases.

### 9. Post-Training Evaluation

Add model checks after training:

- error review by category
- confidence distribution by class
- threshold sweep for human review
- false-negative escalation review
- slice metrics by ticket channel, customer tier, language, and category
- monitor override rate after humans correct predictions

### 10. Product Integration

The next integration step is to connect the vertical to the existing backend:

```text
POST /tickets
  -> save ticket
  -> run triage prediction
  -> save triage result
  -> return ticket + triage
```

Longer term, add database tables for:

- `triage_predictions`
- `risk_scores`
- `human_reviews`
- `model_versions`
- `evaluation_traces`

## Bottom Line

The current version is a strong portfolio slice because it is more than a classifier:

- it trains a baseline model
- saves metrics and artifacts
- exposes prediction outputs
- scores escalation risk
- routes cases to humans
- includes responsible-AI documentation
- is structured so a transformer can replace or compete with the baseline later

The most important next improvements are real labels, clearer priority definitions, confidence calibration, threshold tuning, and then transformer fine-tuning once there is enough data.
