import os
import json
import joblib
import streamlit as st
import pandas as pd
from datetime import datetime

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

# ------------------ BASIC AUTH ------------------
def check_login():
    st.sidebar.markdown("## 🔒 Login")

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_btn = st.sidebar.button("Login")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if login_btn:
        u_ok = username == st.secrets.get("APP_USERNAME", "")
        p_ok = password == st.secrets.get("APP_PASSWORD", "")
        if u_ok and p_ok:
            st.session_state.logged_in = True
        else:
            st.sidebar.error("Wrong username or password")

    return st.session_state.logged_in

if not check_login():
    st.title("📉 Customer Churn Prediction")
    st.warning("Please login from the sidebar to use the app.")
    st.stop()

# ------------------ TITLE ------------------
st.title("📉 Customer Churn Prediction")
st.caption("Fill the customer details in the sidebar and click **Predict**.")

# ------------------ MULTI-MODEL LOADER ------------------
# Add multiple model files here (upload them to repo root)
MODEL_OPTIONS = {
    "Main Model (Pipeline)": "churn_pipeline.joblib",
    # Add more when you have them:
    # "Logistic Regression": "churn_pipeline_lr.joblib",
    # "Random Forest": "churn_pipeline_rf.joblib",
}

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

st.sidebar.markdown("## 🧠 Model Selection")
model_choice = st.sidebar.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
MODEL_PATH = MODEL_OPTIONS[model_choice]

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: `{MODEL_PATH}`. Upload it to the repo root.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model `{MODEL_PATH}`.\n\nError: {e}")
    st.stop()

# ------------------ INPUTS (SIDEBAR) ------------------
st.sidebar.markdown("## 🧾 Customer Inputs")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("SeniorCitizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

phone = st.sidebar.selectbox("PhoneService", ["Yes", "No"])
internet = st.sidebar.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox(
    "PaymentMethod",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)
monthly = st.sidebar.number_input("MonthlyCharges", min_value=0.0, value=70.0)

total = st.sidebar.number_input("TotalCharges", min_value=0.0, value=float(monthly * max(int(tenure), 1)))
avg = total / tenure if tenure > 0 else 0.0

# Defaults for missing columns (must match training features)
input_row = {
    "gender": gender,
    "SeniorCitizen": int(senior),
    "Partner": partner,
    "Dependents": dependents,
    "tenure": int(tenure),
    "PhoneService": phone,
    "MultipleLines": "No",
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

# ------------------ MAIN LAYOUT ------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Input Summary")
    st.dataframe(X_input, use_container_width=True)

    predict_btn = st.button("Predict", type="primary")

with right:
    st.subheader("Prediction")

    if predict_btn:
        try:
            proba = float(model.predict_proba(X_input)[0][1])
            pred = int(proba >= 0.5)

            st.metric("Churn probability", f"{proba:.3f}")
            st.progress(min(max(proba, 0.0), 1.0))

            if pred == 1:
                st.error("⚠️ Prediction: Customer is likely to churn.")
            else:
                st.success("✅ Prediction: Customer is not likely to churn.")

            # ------------------ DOWNLOAD REPORT ------------------
            st.subheader("Download report")

            report = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_name": model_choice,
                "model_file": MODEL_PATH,
                "prediction_label": pred,
                "churn_probability": proba,
                "inputs": input_row,
            }

            csv_bytes = X_input.assign(
                churn_probability=proba,
                prediction_label=pred,
                model=model_choice
            ).to_csv(index=False).encode("utf-8")

            json_bytes = json.dumps(report, indent=2).encode("utf-8")

            st.download_button(
                "⬇️ Download CSV report",
                data=csv_bytes,
                file_name="churn_prediction_report.csv",
                mime="text/csv",
            )
            st.download_button(
                "⬇️ Download JSON report",
                data=json_bytes,
                file_name="churn_prediction_report.json",
                mime="application/json",
            )

            # ------------------ FEATURE IMPORTANCE ------------------
            st.subheader("Feature importance (if available)")

            # Try to locate final estimator inside a Pipeline
            estimator = model
            if hasattr(model, "named_steps") and len(model.named_steps) > 0:
                # last step is usually the estimator
                estimator = list(model.named_steps.values())[-1]

            # Works for linear models / tree models (if feature names can be derived)
            if hasattr(estimator, "feature_importances_") or hasattr(estimator, "coef_"):
                # Try to get feature names from preprocessing
                feature_names = None
                try:
                    if hasattr(model, "get_feature_names_out"):
                        feature_names = model.get_feature_names_out()
                except Exception:
                    feature_names = None

                if hasattr(estimator, "feature_importances_"):
                    importances = estimator.feature_importances_
                else:
                    # coef_ may be 2D for classifiers
                    coef = estimator.coef_
                    importances = coef[0] if hasattr(coef, "__len__") and len(coef) > 0 else coef

                fi = pd.DataFrame({
                    "feature": feature_names if feature_names is not None else [f"f{i}" for i in range(len(importances))],
                    "importance": importances
                }).sort_values("importance", ascending=False).head(20)

                st.dataframe(fi, use_container_width=True)
            else:
                st.info("This model type does not expose feature importance (or it can't be extracted from the pipeline).")

        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")
    else:
        st.info("Click **Predict** to see results.")
