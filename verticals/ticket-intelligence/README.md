# OpsPilot AI

This repo currently contains one polished vertical slice: **OpsPilot Ticket Intelligence**, an AI-assisted triage and escalation-risk routing module for operations tickets.

See [ml/ticket_intelligence/README.md](ml/ticket_intelligence/README.md) for the case study, architecture, run commands, evaluation artifacts, and responsible AI notes.

For hands-on experimentation, open [notebooks/ticket_intelligence_walkthrough.ipynb](notebooks/ticket_intelligence_walkthrough.ipynb). It contains the current ML code in one notebook with markdown explanations, evaluation, routing, normalization experiments, and hyperparameter tuning ideas.

For a plain-English explanation of the model/data/routing pipeline, read [docs/ml_pipeline_summary.md](docs/ml_pipeline_summary.md).

## Resume Bullet

Built OpsPilot Ticket Intelligence, an AI operations triage module that classifies messy support tickets, estimates escalation risk, and routes low-confidence or high-risk cases to human review using baseline text classification, model-versioned outputs, evaluation reports, and human-in-the-loop routing.

Synthetic smoke-test benchmark: category macro-F1 `0.60`; priority macro-F1 `0.2741`. Main training now expects real normalized splits under `data/processed/`.
