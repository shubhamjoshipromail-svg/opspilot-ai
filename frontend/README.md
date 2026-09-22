# Frontend

This folder contains the canonical Streamlit MVP for OpsPilot.

The frontend calls the canonical FastAPI backend over HTTP. It does not import
Ticket Intelligence research code or load model artifacts directly.

Configure the backend URL if it is not running on the default:

```bash
export OPSPILOT_API_URL=http://localhost:8000
```

Run the frontend from the repository root:

```bash
streamlit run frontend/streamlit_app.py
```

Current tabs:

- Dashboard.
- Ticket Routing Demo.
- Recent Predictions.
- Ticket Inbox.
- Model Analytics.
