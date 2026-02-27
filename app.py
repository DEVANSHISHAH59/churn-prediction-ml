import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")
st.title("📉 Customer Churn Prediction")
st.write("Fill the customer details and click **Predict** to get churn probability.")

# ---- Load model pipeline ----
# IMPORTANT: we will upload this file next (from Colab)
MODEL_PATH = "churn_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# If model is not uploaded yet, show a friendly error
try:
    model = load_model()
except Exception as e:
    st.error("Model file not found. Please upload `churn_pipeline.joblib` to this GitHub repo first.")
    st.stop()

# ---- User inputs (basic but useful) ----
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("SeniorCitizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

with col2:
    phone = st.selectbox("PhoneService", ["Yes", "No"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("PaymentMethod", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.number_input("MonthlyCharges", min_value=0.0, value=70.0)

# TotalCharges (approx) + engineered feature
total = st.number_input("TotalCharges", min_value=0.0, value=float(monthly * max(tenure, 1)))
avg = total / tenure if tenure > 0 else 0.0

# ---- Build input row with the SAME column names as dataset ----
# We will fill remaining columns with common defaults so the pipeline works.
input_row = {
    "gender": gender,
    "SeniorCitizen": int(senior),
    "Partner": partner,
    "Dependents": dependents,
    "tenure": int(tenure),
    "PhoneService": phone,
    "MultipleLines": "No",  # default
    "InternetService": internet,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": payment,
    "MonthlyCharges": float(monthly),
    "TotalCharges": float(total),
    "AvgCharges": float(avg),
}

X_input = pd.DataFrame([input_row])

if st.button("Predict"):
    proba = model.predict_proba(X_input)[0][1]
    pred = int(proba >= 0.5)

    st.subheader("Result")
    st.write(f"**Churn probability:** {proba:.3f}")

    if pred == 1:
        st.error("⚠️ Prediction: Customer is likely to churn.")
    else:
        st.success("✅ Prediction: Customer is not likely to churn.")
import os, joblib, streamlit as st

MODEL_PATH = "churn_pipeline.joblib"

st.write("Current folder files:", os.listdir("."))  # temporary debug

if not os.path.exists(MODEL_PATH):
    st.error(f"Missing {MODEL_PATH}. Please upload it to the repo root.")
    st.stop()

model = joblib.load(MODEL_PATH)
