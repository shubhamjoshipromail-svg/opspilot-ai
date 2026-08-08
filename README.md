# OpsPilot AI

Support-ticket routing. A fine-tuned ModernBERT classifier assigns an incoming
ticket to one of seven operational queues, and routes low-confidence tickets to a
human instead of guessing.

## What exists

**The model.** ModernBERT fine-tuned for sequence classification, deployed at
[huggingface.co/shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1](https://huggingface.co/shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1).
Routing accuracy went from ~52% to ~73%. The seven labels are
`billing_and_payments`, `customer_general`, `human_resources`,
`returns_and_exchanges`, `sales_and_pre_sales`,
`service_outages_and_maintenance`, and `technical_product_support`.

**The finding.** Most of that gain came from redesigning the label taxonomy, not
from the model. The original categories overlapped enough that no classifier could
separate them cleanly — two labels could both be correct for the same ticket, so
the ceiling was in the labels, not the architecture. Collapsing and redrawing the
categories moved accuracy further than any amount of tuning did.

**Confidence gating.** Below the confidence threshold, a ticket goes to a human
queue rather than being silently misrouted. A wrong auto-route costs more than an
unrouted ticket, so the model is allowed to abstain.

**The scaffold.** A FastAPI service with ticket CRUD over SQLAlchemy/Postgres
(`backend/`), a normalized-ticket loader (`scripts/`), reference policy documents
(`data/policies/`), and pytest coverage for the health and ticket routes.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # set DATABASE_URL
uvicorn backend.app.main:app --reload
pytest
```

## Not built yet

The trained model is not wired into the API — inference lives on Hugging Face and
the FastAPI service does not call it yet. The Streamlit app in `frontend/` is a
placeholder page. There is no evaluation harness, no analytics, and no review UI
in the repo; the documents under `docs/` describe intended design, not shipped
behaviour.
