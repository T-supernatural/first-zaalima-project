"""Run with: streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.predict import FEATURES, load_model, predict_transactions

st.set_page_config(page_title="Zalima Fraud Detection", page_icon="🔎")
st.title("Transaction fraud detection")
st.info("Educational demonstration. This is not a production financial security system.")
st.write("Upload a CSV with Time, V1–V28 and Amount, without Class. V1–V28 must already be the dataset's anonymised features; raw card details cannot be converted here.")
st.caption("Prediction 1 flags potential fraud; 0 indicates a score below 0.5. Scores are not calibrated probabilities, especially with class weighting.")
st.download_button("Download input header", ",".join(FEATURES) + "\n", "transaction_header.csv", "text/csv")
uploaded = st.file_uploader("Transaction CSV", type="csv")
if uploaded is not None:
    try:
        transactions = pd.read_csv(uploaded)
        st.dataframe(predict_transactions(transactions, load_model()), use_container_width=True)
    except (ValueError, TypeError, FileNotFoundError, pd.errors.ParserError) as exc:
        st.error(str(exc))
