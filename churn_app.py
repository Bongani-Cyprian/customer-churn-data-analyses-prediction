import streamlit as st
import pandas as pd
import joblib

import numpy as np
import plotly.express as px
from sklearn.metrics import classification_report

# Import the custom transformer
from feature_engineering import ChurnFeatureEngineer

st.set_page_config(page_title="Churn Risk Checker", layout="wide")
st.title("📉 Customer Churn Risk Checker")

@st.cache(allow_output_mutation=True)
def load_model(path):
    return joblib.load(path)

pipe = load_model("model_artifacts/telenova_churn_xgb.joblib")

uploaded_file = st.file_uploader("🔼 Drop a CSV with the feature columns", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview", df.head())
    st.write("### Data Columns", df.columns.tolist())

    # drop any extra cols
    for col in ["customer_id", "churn", "churn_flag"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # predict probabilities
    probs = pipe.predict_proba(df)[:, 1]
    df["churn_probability"] = probs.round(3)

    # Quick static metric at 50%
    num_high50 = (df["churn_probability"] >= 0.5).sum()
    st.metric("Accounts flagged ≥ 50% risk", num_high50)

    # ─── Interactive Plotly Section ─────────────────────────────
    st.subheader("📊 Churn Probability Distribution")

    # 1) slider for threshold
    threshold = st.slider(
        "Select churn threshold", 
        min_value=0.0, max_value=1.0, value=0.6, step=0.01
    )

    # 2) label at that threshold
    df["predicted_label"] = np.where(
        df["churn_probability"] >= threshold, "Churn", "Stay"
    )

    # 3) interactive histogram
    fig = px.histogram(
        df,
        x="churn_probability",
        color="predicted_label",
        nbins=50,
        histnorm="density",
        opacity=0.6,
        barmode="overlay",
        labels={"churn_probability": "Churn Probability"},
        title=f"Churn Probability @ threshold {threshold:.2f}"
    )

    # 4) vertical line at threshold
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"{threshold:.2f}",
        annotation_position="top right"
    )

    # 5) render Plotly chart
    st.plotly_chart(fig, use_container_width=True)

    # ─── Live Metrics ────────────────────────────────────────────
   
    y_true = (df["churn_probability"] >= 0.5).astype(int)
    y_pred = (df["churn_probability"] >= threshold).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Recall", f"{report['1']['recall']:.2f}")
    col2.metric("Churn Precision", f"{report['1']['precision']:.2f}")
    col3.metric("Overall F1", f"{report['accuracy']:.2f}")

    # ─── Download scored CSV as before ──────────────────────────
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download scored CSV",
        data=csv_data,
        file_name="customer_churn_scored.csv",
        mime="text/csv"
    )

else:
    st.info("Awaiting CSV upload…")
