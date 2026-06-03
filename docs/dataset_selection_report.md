# Dataset Selection Report

OpsPilot Ticket Intelligence v1 uses `Tobi-Bueck/customer-support-tickets` as its public source dataset.

## Why This Dataset

This dataset is aligned with an operations-ticket workflow. It includes ticket-style fields such as subject/body text, queue/category, priority, language, tags, type, and support answers. That makes it useful for triage classification, priority prediction, routing-risk analysis, and later reference-answer evaluation.

## Alternatives Considered

- `bitext/Bitext-customer-support-llm-chatbot-training-dataset`: strong for chatbot intent and response generation patterns, but less aligned with operational ticket routing and priority modeling.
- `gorkemsevinc/customer_support_tickets`: appears smaller and less directly aligned with the richer queue/priority/language/tag/answer workflow needed for OpsPilot.

## Scope Decision

Version 1 is English-only. German/multilingual tickets are intentionally excluded to keep the TF-IDF/logistic-regression baseline interpretable and the evaluation story clean.

Future multilingual support could use language detection with language-specific models or multilingual transformers such as XLM-R or multilingual DistilBERT.
