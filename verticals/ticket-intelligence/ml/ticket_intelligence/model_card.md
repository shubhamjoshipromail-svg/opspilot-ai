# Model Card: OpsPilot Ticket Intelligence v1

## Model Details

- Model version: `ticket-intelligence-v1`
- Baseline: TF-IDF vectorizer plus logistic regression
- Transformer layer: `train_transformer.py` supports DistilBERT/MiniLM-style fine-tuning when optional dependencies are installed
- Outputs: ticket category, priority, confidence, escalation risk, routing decision, explanation, metadata

## Intended Use

This module is intended to assist operations teams with messy support-ticket triage. It can suggest a category, estimate urgency, flag escalation risk, and route cases to a queue such as human review, supervisor review, or auto-triage suggestion.

## Not Intended For

- Automatic final decisions on refunds, cancellations, compliance responses, or account enforcement
- Replacing trained support, legal, compliance, or operations staff
- High-stakes use without human oversight and production monitoring

## Training Data

The starter dataset is synthetic and designed to exercise operational scenarios across billing disputes, technical issues, account access, cancellations, shipping delays, and compliance requests. It should be replaced or augmented with real historical tickets before production use.

## Evaluation

The baseline produces:

- Accuracy
- Macro-F1
- Per-class precision, recall, and F1
- Confusion matrix
- Error examples

Artifacts are saved under `ml/ticket_intelligence/outputs/`.

## Human Oversight Policy

The router sends tickets to human review when model confidence is low, escalation risk is high, priority is high, or the ticket contains policy-sensitive compliance/legal language. The model should be treated as decision support, not an autonomous operations agent.

## Limitations

- Synthetic training data is not representative of all customer language.
- Confidence scores from logistic regression are useful for routing thresholds but are not calibrated probabilities.
- Rule-based escalation scoring may miss implicit anger, sarcasm, or organization-specific risk.
- The transformer script requires optional dependencies and real compute/network access to fine-tune.

## Responsible AI Notes

Production use should include drift monitoring, calibration, periodic human review of routed outcomes, bias checks across customer segments, and an appeals path for customer-impacting decisions.
