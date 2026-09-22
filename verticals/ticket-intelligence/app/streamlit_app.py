from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TICKET_INTELLIGENCE_DIR = ROOT / "ml" / "ticket_intelligence"
sys.path.insert(0, str(TICKET_INTELLIGENCE_DIR))

import streamlit as st

from predict import predict_ticket


st.set_page_config(page_title="OpsPilot Ticket Intelligence", layout="wide")
st.title("OpsPilot Ticket Intelligence")

default_text = (
    "I was charged twice and support keeps closing my tickets. "
    "If this refund is not handled today I am escalating to legal."
)
message = st.text_area("Ticket text", value=default_text, height=180)

if st.button("Analyze ticket", type="primary"):
    result = predict_ticket(message)
    cols = st.columns(4)
    cols[0].metric("Category", result["category"])
    cols[1].metric("Priority", result["priority"])
    cols[2].metric("Escalation risk", result["escalation_risk"])
    cols[3].metric("Confidence", result["confidence"])
    st.subheader("Routing")
    st.write(result["routing_decision"])
    st.write(result["reason"])
    st.subheader("Risk signals")
    st.write(result["risk_signals"])
    st.subheader("Raw output")
    st.json(result)
