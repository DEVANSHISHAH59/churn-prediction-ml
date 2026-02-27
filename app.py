import os
import json
import joblib
import streamlit as st
import pandas as pd
from datetime import datetime

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")


# ------------------ LOGIN + LOGOUT ------------------
def check_login() -> bool:
    st.sidebar.markdown("## 🔒 Login")

    # Session state init
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # If NOT logged in → show login form
    if not st.session_state.logged_in:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            u_ok = username == st.secrets.get("APP_USERNAME", "")
            p_ok = password == st.secrets.get("APP_PASSWORD", "")

            if u_ok and p_ok:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.sidebar.error("Wrong username or password")

    # If logged in → show logout button
    else:
        st.sidebar.success("Logged in ✅")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    return st.session_state.logged_in


# Stop app if not logged in
if not check_login():
    st.title("📉 Customer Churn Prediction")
    st.warning("Please login from the sidebar to use the app.")
    st.stop()


# ------------------ TITLE ------------------
st.title("📉 Customer Churn Prediction")
st.caption("Fill the customer details in the sidebar and click **Predict** to get churn probability.")
st.markdown("---")


# ------------------ MULTI-MODEL SETUP ------------------
MODEL_OPTIONS = {
    "Main Model (Pipeline)": "churn_pipeline.joblib",
    "Logistic Regression": "churn_pipeline_lr.joblib",
    "Random Forest": "churn_pipeline_rf.joblib",
}

st.sidebar.markdown("## 🧠 Model Selection")
model_choice = st.sidebar.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
MODEL_PATH = MODEL_OPTIONS[model_choice]


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: `{MODEL_PATH}`. Upload it to the GitHub repo root.")
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

# Must match training feature columns (defaults for missing ones)
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

    if not predict_btn:
        st.info("Click **Predict** to see results.")
    else:
        try:
            proba = float(model.predict_proba(X_input)[0][1])
            pred = int(proba >= 0.5)

            # ---- Probability Dashboard ----
            st.metric("Churn Probability", f"{proba*100:.1f}%")
            st.progress(min(max(proba, 0.0), 1.0))

            if proba < 0.30:
                st.success("Low Risk 🟢")
            elif proba < 0.60:
                st.warning("Medium Risk 🟡")
            else:
                st.error("High Risk 🔴")

            # ---- Final prediction message ----
            if pred == 1:
                st.error("⚠️ Prediction: Customer is likely to churn.")
            else:
                st.success("✅ Prediction: Customer is not likely to churn.")

            st.markdown("---")

            # ------------------ DOWNLOAD REPORT ------------------
            st.subheader("Download Prediction Report")

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
                model=model_choice,
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

            st.markdown("---")

            # ------------------ FEATURE IMPORTANCE ------------------
            st.subheader("Feature Importance (if available)")

            estimator = model
            if hasattr(model, "named_steps") and len(model.named_steps) > 0:
                estimator = list(model.named_steps.values())[-1]

            if hasattr(estimator, "feature_importances_"):
                fi = pd.DataFrame({
                    "feature": [f"f{i}" for i in range(len(estimator.feature_importances_))],
                    "importance": estimator.feature_importances_,
                }).sort_values("importance", ascending=False).head(15)

                st.bar_chart(fi.set_index("feature"))

            elif hasattr(estimator, "coef_"):
                coef = estimator.coef_
                coef_1d = coef[0] if hasattr(coef, "shape") and len(coef.shape) > 1 else coef
                st.bar_chart(pd.DataFrame({"coef": coef_1d}))

            else:
                st.info("Feature importance not available for this model.")

        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")
