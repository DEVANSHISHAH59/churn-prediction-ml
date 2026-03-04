import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Churn Risk Insights", page_icon="📉", layout="wide")
st.title("📉 Churn Risk Insights Dashboard")

st.markdown(
    """
**What is churn?**  
Churn is when a customer stops using a service (cancels subscription, disconnects, stops purchasing).

**Real-world problem solved:**  
Retention teams need to know **who is likely to churn**, **why**, and **what action to take** (win-back offer, proactive support, plan change).
"""
)

# =========================================================
# SAMPLE DATA GENERATOR
# =========================================================
@st.cache_data(show_spinner=False)
def make_sample_churn_data(n=2500, seed=7):
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], size=n, p=[0.35, 0.25, 0.25, 0.15])
    internet = rng.choice(["DSL", "Fiber optic", "None"], size=n, p=[0.35, 0.50, 0.15])

    tenure_months = rng.integers(1, 73, size=n)
    monthly_charges = np.clip(rng.normal(70, 25, size=n), 15, 130)
    support_tickets_90d = np.clip(rng.poisson(1.2, size=n), 0, 10)
    late_payments_6m = np.clip(rng.poisson(0.8, size=n), 0, 6)
    avg_usage_hours = np.clip(rng.normal(3.2, 1.4, size=n), 0.2, 10.0)

    has_partner = rng.choice([0, 1], size=n, p=[0.55, 0.45])
    has_dependents = rng.choice([0, 1], size=n, p=[0.70, 0.30])
    paperless_billing = rng.choice([0, 1], size=n, p=[0.40, 0.60])

    # Churn propensity (synthetic logic)
    # Higher churn for month-to-month, fiber optic (often higher price), more tickets, more late payments, low tenure
    contract_weight = np.where(contract == "Month-to-month", 1.0, np.where(contract == "One year", 0.35, 0.2))
    internet_weight = np.where(internet == "Fiber optic", 0.45, np.where(internet == "DSL", 0.2, 0.05))
    price_weight = (monthly_charges - 50) / 80  # normalized-ish
    tenure_weight = (25 - np.minimum(tenure_months, 25)) / 25  # newer customers higher risk
    tickets_weight = support_tickets_90d / 8
    late_weight = late_payments_6m / 6

    logit = (
        -1.1
        + 1.25 * contract_weight
        + 0.55 * internet_weight
        + 0.95 * tenure_weight
        + 0.85 * tickets_weight
        + 0.75 * late_weight
        + 0.40 * price_weight
        - 0.20 * has_partner
        - 0.15 * has_dependents
    )

    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, np.clip(prob, 0.02, 0.90), size=n)

    df = pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(n)],
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges.round(2),
        "avg_usage_hours": avg_usage_hours.round(2),
        "support_tickets_90d": support_tickets_90d,
        "late_payments_6m": late_payments_6m,
        "contract_type": contract,
        "payment_method": payment,
        "internet_service": internet,
        "has_partner": has_partner,
        "has_dependents": has_dependents,
        "paperless_billing": paperless_billing,
        "churn": churn
    })
    return df


# =========================================================
# DATASET LOADING (NO UPLOAD NEEDED)
# =========================================================
st.sidebar.markdown("## 📂 Dataset")
use_demo = st.sidebar.checkbox("Use built-in sample dataset (recommended)", True)

if use_demo:
    df = make_sample_churn_data()
    st.sidebar.success("✅ Loaded sample churn dataset")
else:
    st.sidebar.info("For portfolio clarity, keep demo dataset enabled (no upload).")
    df = make_sample_churn_data()

st.sidebar.divider()
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Choose module",
    ["🏠 Overview", "🧠 Predict & Explain", "🎯 Risk Segments", "📊 Model Insights", "🧪 Decision Boundary (Optional)"],
    label_visibility="collapsed"
)

# =========================================================
# TRAIN MODEL
# =========================================================
target = "churn"
id_col = "customer_id"

X = df.drop(columns=[target])
y = df[target].astype(int)

num_cols = ["tenure_months", "monthly_charges", "avg_usage_hours", "support_tickets_90d", "late_payments_6m"]
bin_cols = ["has_partner", "has_dependents", "paperless_billing"]
cat_cols = ["contract_type", "payment_method", "internet_service"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("bin", "passthrough", bin_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

model = LogisticRegression(max_iter=1000)

pipe = Pipeline(steps=[("prep", preprocess), ("clf", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.22, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)
proba_test = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba_test)

# Pred for whole dataset (for segments)
df_scored = df.copy()
df_scored["churn_probability"] = pipe.predict_proba(df.drop(columns=[target]))[:, 1]

def risk_bucket(p):
    if p >= 0.70: return "High"
    if p >= 0.40: return "Medium"
    return "Low"

df_scored["risk_segment"] = df_scored["churn_probability"].apply(risk_bucket)


# =========================================================
# HELPERS: Feature importance (permutation)
# =========================================================
@st.cache_data(show_spinner=False)
def compute_perm_importance(_pipe, X_sample, y_sample):
    # Permutation importance works with pipelines
    r = permutation_importance(_pipe, X_sample, y_sample, n_repeats=5, random_state=42, scoring="roc_auc")
    return r

# =========================================================
# PAGE: OVERVIEW
# =========================================================
if page == "🏠 Overview":
    st.markdown("## 🏠 Overview")

    churn_rate = df[target].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{churn_rate*100:.1f}%")
    c3.metric("Model AUC (test)", f"{auc:.3f}")
    c4.metric("Avg churn probability", f"{df_scored['churn_probability'].mean():.2f}")

    st.markdown("### Data preview")
    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("### Churn probability distribution")
    fig, ax = plt.subplots()
    ax.hist(df_scored["churn_probability"], bins=30)
    ax.set_xlabel("Churn probability")
    ax.set_ylabel("Customers")
    st.pyplot(fig)

    st.info(
        "Tip: Go to **Risk Segments** to export a list of high-risk customers for retention outreach."
    )


# =========================================================
# PAGE: PREDICT & EXPLAIN
# =========================================================
elif page == "🧠 Predict & Explain":
    st.markdown("## 🧠 Predict & Explain (Single customer)")
    st.caption("Use this like a retention analyst: input customer attributes, get churn probability + explanation.")

    # pick an example customer to prefill
    example_id = st.selectbox("Choose an example customer", df_scored["customer_id"].head(50))
    base = df_scored[df_scored["customer_id"] == example_id].iloc[0]

    colA, colB, colC = st.columns(3)
    with colA:
        tenure = st.slider("Tenure (months)", 1, 72, int(base["tenure_months"]))
        monthly = st.slider("Monthly charges", 15.0, 130.0, float(base["monthly_charges"]))
        usage = st.slider("Avg usage (hours/day)", 0.2, 10.0, float(base["avg_usage_hours"]))
    with colB:
        tickets = st.slider("Support tickets (last 90 days)", 0, 10, int(base["support_tickets_90d"]))
        late = st.slider("Late payments (last 6 months)", 0, 6, int(base["late_payments_6m"]))
        contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(base["contract_type"]))
    with colC:
        payment = st.selectbox("Payment method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"], index=["Electronic check", "Credit card", "Bank transfer", "Mailed check"].index(base["payment_method"]))
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "None"], index=["DSL", "Fiber optic", "None"].index(base["internet_service"]))
        has_partner = st.selectbox("Has partner", [0, 1], index=int(base["has_partner"]))
        has_dependents = st.selectbox("Has dependents", [0, 1], index=int(base["has_dependents"]))
        paperless = st.selectbox("Paperless billing", [0, 1], index=int(base["paperless_billing"]))

    customer_row = pd.DataFrame([{
        "customer_id": "CUSTOM_INPUT",
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "avg_usage_hours": usage,
        "support_tickets_90d": tickets,
        "late_payments_6m": late,
        "contract_type": contract,
        "payment_method": payment,
        "internet_service": internet,
        "has_partner": has_partner,
        "has_dependents": has_dependents,
        "paperless_billing": paperless,
    }])

    p = pipe.predict_proba(customer_row)[:, 1][0]
    seg = risk_bucket(p)

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{p:.2f}")
    c2.metric("Risk segment", seg)
    c3.metric("Suggested action", "Win-back offer + proactive support" if seg == "High" else ("Monitor + targeted nudge" if seg == "Medium" else "Maintain"))

    st.progress(min(max(p, 0.0), 1.0))

    st.markdown("### Explanation (what usually drives churn in this model)")
    st.write(
        "- **High support tickets / late payments** increase churn risk\n"
        "- **Month-to-month contract** increases churn risk\n"
        "- **Low tenure** increases churn risk\n"
        "- **High charges** can increase churn risk\n"
        "- Partner/dependents tend to reduce churn risk (more stable customers)"
    )


# =========================================================
# PAGE: RISK SEGMENTS
# =========================================================
elif page == "🎯 Risk Segments":
    st.markdown("## 🎯 Risk Segments (Target lists)")
    st.caption("This is what a retention team exports for outreach campaigns.")

    seg_counts = df_scored["risk_segment"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    c1, c2, c3 = st.columns(3)
    c1.metric("High risk", f"{int(seg_counts.get('High',0)):,}")
    c2.metric("Medium risk", f"{int(seg_counts.get('Medium',0)):,}")
    c3.metric("Low risk", f"{int(seg_counts.get('Low',0)):,}")

    st.markdown("### High-risk customers (top 50)")
    high = df_scored[df_scored["risk_segment"] == "High"].sort_values("churn_probability", ascending=False)
    st.dataframe(high.head(50), use_container_width=True)

    st.download_button(
        "⬇️ Download high-risk target list (CSV)",
        data=high.to_csv(index=False).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("### Recommended outreach playbooks")
    st.write("- **High risk:** win-back offer, proactive support call, plan downgrade option")
    st.write("- **Medium risk:** product education, loyalty points, targeted discount")
    st.write("- **Low risk:** upsell/cross-sell, referral campaigns")


# =========================================================
# PAGE: MODEL INSIGHTS
# =========================================================
elif page == "📊 Model Insights":
    st.markdown("## 📊 Model Insights")
    st.caption("Explainable ML: feature importance, correlations, and probability distribution.")

    # Probability histogram
    st.markdown("### Churn probability histogram")
    fig, ax = plt.subplots()
    ax.hist(df_scored["churn_probability"], bins=30)
    ax.set_xlabel("Churn probability")
    ax.set_ylabel("Customers")
    st.pyplot(fig)

    # Correlation heatmap (numeric only)
    st.markdown("### Correlation heatmap (numeric features)")
    corr = df[num_cols + bin_cols + ["churn"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr.values)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

    # Feature importance (permutation)
    st.markdown("### Feature importance (permutation on test set)")
    imp = compute_perm_importance(pipe, X_test, y_test)

    # Map permutation importances back to ORIGINAL columns (approx)
    # Permutation importance returns per input column of X_test (pre-transform)
    imp_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": imp.importances_mean
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots()
    top = imp_df.head(10)[::-1]
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importance (Δ AUC)")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

    st.markdown("### Model performance snapshot")
    pred_test = (proba_test >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred_test)
    st.write("Confusion matrix (threshold=0.5):")
    st.write(cm)
    st.text("Classification report:")
    st.text(classification_report(y_test, pred_test))


# =========================================================
# PAGE: DECISION BOUNDARY (OPTIONAL)
# =========================================================
else:
    st.markdown("## 🧪 Decision Boundary (Optional)")
    st.caption("This is a visual intuition tool. It only works for 2 numeric features at a time.")

    f1 = st.selectbox("Feature 1", ["tenure_months", "monthly_charges", "avg_usage_hours", "support_tickets_90d", "late_payments_6m"], index=0)
    f2 = st.selectbox("Feature 2", ["tenure_months", "monthly_charges", "avg_usage_hours", "support_tickets_90d", "late_payments_6m"], index=1)

    if f1 == f2:
        st.warning("Choose two different features.")
        st.stop()

    # Train a simple 2D model for boundary visualization only
    X2 = df[[f1, f2]]
    y2 = df["churn"].astype(int)

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.22, random_state=42, stratify=y2)
    pipe2 = Pipeline(steps=[("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    pipe2.fit(X2_train, y2_train)

    # Create mesh
    x_min, x_max = X2[f1].min(), X2[f1].max()
    y_min, y_max = X2[f2].min(), X2[f2].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = pd.DataFrame({f1: xx.ravel(), f2: yy.ravel()})
    zz = pipe2.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(xx, yy, zz, levels=20)
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    ax.scatter(X2_test[f1], X2_test[f2], s=10, alpha=0.6)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title("Churn probability decision surface (2D)")
    st.pyplot(fig)
