# Docs

OpsPilot documentation lives in this folder.

## Core References

- [Product Architecture](PRODUCT_ARCHITECTURE.md)
- [Product Spec](product_spec.md)
- [Roadmap](roadmap.md)
- [Repository Audit](repository_audit.md)
- [Development Guide](setup/development.md)

## ML And Data

- [Ticket Intelligence System](ml/ticket_intelligence_system.md)
- [Dataset Audit](data/dataset_audit.md)
- [Model Interfaces](model_interfaces.md)
- [Evaluation Plan](evaluation_plan.md)
- [Dataset Selection Report](dataset_selection_report.md)

## Product Planning

- [Module Opportunity Matrix](product/module_opportunity_matrix.md)
- [Analytics Plan](analytics_plan.md)
- [Data Model](data_model.md)

## Dataset Notes

- Routing/category label: `true_category`
- Priority label: `true_priority`
- Ticket type: `ticket_type`
- Reference answer: `reference_answer`
- Tags are stored as text in `tags` for the MVP.

The normalized English dataset is regenerated locally at
`data/processed/tickets_en_normalized.csv` and is intentionally ignored by Git.
