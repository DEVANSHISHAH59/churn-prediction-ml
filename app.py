import os
import joblib
import streamlit as st
import pandas as pd

# ------------------ Page UI ------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")
st.title("📉 Customer Churn Prediction")
st.write("Fill the customer details and click **Predict** to get churn probability.")

# ------------------ Load Model ------------------
MODEL_PATH = "churn_pipeline.joblib"

# (Temporary debug) show files Streamlit sees in the repo
st.write("Current folder files:", os.listdir("."))

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: `{MODEL_PATH}`. Make sure it is uploaded to the GitHub repo root.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model from `{MODEL_PATH}`. Error: {e}")
    st.stop()

# ------------------ Inputs ------------------
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
    payment = st.selectbox(
        "PaymentMethod",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = st.number_input("MonthlyCharges", min_value=0.0, value=70.0)

# TotalCharges (approx) + engineered feature
total = st.number_input("TotalCharges", min_value=0.0, value=float(monthly * max(int(tenure), 1)))
avg = total / tenure if tenure > 0 else 0.0

# ------------------ Build Input Row ------------------
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

# ------------------ Predict ------------------
if st.button("Predict"):
    try:
        proba = model.predict_proba(X_input)[0][1]
        pred = int(proba >= 0.5)

        st.subheader("Result")
        st.write(f"**Churn probability:** {proba:.3f}")

        if pred == 1:
            st.error("⚠️ Prediction: Customer is likely to churn.")
        else:
            st.success("✅ Prediction: Customer is not likely to churn.")
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
