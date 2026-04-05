from __future__ import annotations

import os

import requests
import streamlit as st

from fraud_detection.ui_helpers import (
    default_transaction_payload,
    load_local_status,
    parse_batch_csv,
)

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _api_get(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _api_post(path: str, payload):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Fraud Ops Dashboard", layout="wide")
st.title("Fraud Detection Operations Dashboard")
st.caption("Prediction, monitoring, and model promotion controls for the PaySim fraud project")

tabs = st.tabs(["Single Prediction", "Batch Scoring", "Model Card", "Drift", "Retraining"])

with tabs[0]:
    st.subheader("Single transaction scoring")
    defaults = default_transaction_payload()
    col1, col2, col3 = st.columns(3)
    with col1:
        step = st.number_input("Step", min_value=1, value=int(defaults["step"]))
        amount = st.number_input("Amount", min_value=0.0, value=float(defaults["amount"]))
        transaction_type = st.selectbox(
            "Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"], index=0
        )
    with col2:
        name_orig = st.text_input("Origin account", value=str(defaults["nameOrig"]))
        old_balance_org = st.number_input(
            "Old origin balance", min_value=0.0, value=float(defaults["oldbalanceOrg"])
        )
        new_balance_orig = st.number_input(
            "New origin balance", min_value=0.0, value=float(defaults["newbalanceOrig"])
        )
    with col3:
        name_dest = st.text_input("Destination account", value=str(defaults["nameDest"]))
        old_balance_dest = st.number_input(
            "Old destination balance", min_value=0.0, value=float(defaults["oldbalanceDest"])
        )
        new_balance_dest = st.number_input(
            "New destination balance", min_value=0.0, value=float(defaults["newbalanceDest"])
        )

    if st.button("Score Transaction", use_container_width=True):
        payload = {
            "step": int(step),
            "type": transaction_type,
            "amount": float(amount),
            "nameOrig": name_orig,
            "oldbalanceOrg": float(old_balance_org),
            "newbalanceOrig": float(new_balance_orig),
            "nameDest": name_dest,
            "oldbalanceDest": float(old_balance_dest),
            "newbalanceDest": float(new_balance_dest),
        }
        try:
            result = _api_post("/api/v1/predict", payload)
            st.success("Prediction complete")
            st.json(result)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tabs[1]:
    st.subheader("Batch CSV scoring")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            frame = parse_batch_csv(uploaded.getvalue())
            st.dataframe(frame.head(10), use_container_width=True)
            if st.button("Score Batch", use_container_width=True):
                result = _api_post("/api/v1/predict/batch", frame.to_dict(orient="records"))
                st.success(f"Scored {len(result)} rows")
                st.dataframe(result, use_container_width=True)
        except Exception as exc:
            st.error(f"Batch scoring failed: {exc}")

with tabs[2]:
    st.subheader("Active model card")
    try:
        st.json(_api_get("/api/v1/model"))
    except Exception as exc:
        st.error(f"Could not load model metadata: {exc}")

with tabs[3]:
    st.subheader("Latest drift report")
    try:
        st.json(_api_get("/api/v1/drift"))
    except Exception as exc:
        st.error(f"Could not load drift report: {exc}")

with tabs[4]:
    st.subheader("Training and promotion status")
    status = load_local_status()
    st.write("Candidate")
    st.json(status.get("candidate", {}))
    st.write("Champion")
    st.json(status.get("champion", {}))
    st.write("Last promotion decision")
    st.json(status.get("promotion", {}))
    st.write("Latest training summary")
    st.json(status.get("training", {}))
