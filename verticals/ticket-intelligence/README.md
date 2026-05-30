# OpsPilot AI

This repo currently contains one polished vertical slice: **OpsPilot Ticket Intelligence**, an AI-assisted triage and escalation-risk routing module for operations tickets.

See [ml/ticket_intelligence/README.md](ml/ticket_intelligence/README.md) for the case study, architecture, run commands, evaluation artifacts, and responsible AI notes.

## Resume Bullet

Built OpsPilot Ticket Intelligence, an AI operations triage module that classifies messy support tickets, estimates escalation risk, and routes low-confidence or high-risk cases to human review using baseline text classification, model-versioned outputs, evaluation reports, and human-in-the-loop routing.

Current starter benchmark on synthetic tickets: category macro-F1 `0.60`; priority macro-F1 `0.2741`.
