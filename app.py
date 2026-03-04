import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Churn Risk Predictor", page_icon="📉", layout="wide")

st.title("📉 Churn Risk Predictor — Risk Segments & Explainable ML")
st.markdown(
    """
**What is churn?**  
Churn is when a customer stops using a service (cancels a subscription, closes an account, or stops purchasing).

**Real-world problem solved:**  
Retention and customer success teams need to know **who is likely to churn**, **why**, and **what action to take** (win-back offers, proactive support, plan changes).

**Quick start**
1) Use the sample dataset (or upload your own CSV/XLSX)  
2) Select the **Churn column (target)**  
3) Click **Train model** → review risk segments + download high-risk customers
"""
)


# =========================================================
# SAMPLE DATA GENERATOR (Netflix-style)
# =========================================================
@st.cache_data(show_spinner=False)
def make_sample_churn_data(n=2500, seed=7):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(n)],
        "tenure_months": rng.integers(1, 73, size=n),
        "monthly_fee": np.clip(rng.normal(14, 4, size=n), 5, 30).round(2),
        "watch_hours_week": np.clip(rng.normal(8, 4, size=n), 0, 40).round(1),
        "plan_type": rng.choice(["Basic", "Standard", "Premium"], size=n, p=[0.35, 0.45, 0.20]),
        "device_count": rng.integers(1, 6, size=n),
        "support_contacts_90d": np.clip(rng.poisson(0.8, size=n), 0, 8),
        "payment_issues_6m": np.clip(rng.poisson(0.5, size=n), 0, 6),
        "region": rng.choice(["NA", "EU", "APAC", "LATAM"], size=n, p=[0.35, 0.30, 0.25, 0.10]),
        "auto_pay": rng.choice([0, 1], size=n, p=[0.45, 0.55]),
        "promo_user": rng.choice([0, 1], size=n, p=[0.65, 0.35]),
    })

    # synthetic churn logic
    logit = (
        -1.0
        + 0.95 * (df["plan_type"].eq("Basic")).astype(int)
        + 0.70 * (df["payment_issues_6m"] / 6)
        + 0.55 * (df["support_contacts_90d"] / 8)
        + 0.75 * ((12 - np.minimum(df["tenure_months"], 12)) / 12)
        + 0.35 * ((df["monthly_fee"] - 10) / 20)
        - 0.18 * (df["watch_hours_week"] / 40)
        - 0.25 * df["auto_pay"]
        - 0.10 * df["promo_user"]
    )
    prob = 1 / (1 + np.exp(-logit))
    df["churn"] = rng.binomial(1, np.clip(prob, 0.03, 0.85), size=n)
    return df


# =========================================================
# FILE READER
# =========================================================
def read_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Upload CSV or Excel only.")


# =========================================================
# LABEL NORMALIZATION (YES/NO, TRUE/FALSE, etc.)
# =========================================================
def normalize_churn(y: pd.Series) -> pd.Series:
    s = y.copy()

    if s.dtype == bool:
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        # treat >0 as churn
        return (s.fillna(0).astype(float) > 0).astype(int)

    s = s.astype(str).str.strip().str.lower()

    pos = {"1", "yes", "true", "churn", "churned", "cancelled", "canceled", "left", "terminated"}
    neg = {"0", "no", "false", "active", "retained", "stay", "stayed", "current"}

    out = []
    unknown = 0
    for v in s:
        if v in pos:
            out.append(1)
        elif v in neg:
            out.append(0)
        else:
            out.append(0)
            unknown += 1

    if unknown > 0:
        st.warning(f"Note: {unknown} rows had unknown churn labels and were treated as non-churn (0).")
    return pd.Series(out, index=y.index)


# =========================================================
# RISK SEGMENT BUCKETS
# =========================================================
def risk_bucket(p: float) -> str:
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


# =========================================================
# SIDEBAR: DATASET
# =========================================================
st.sidebar.header("📂 Dataset")
use_demo = st.sidebar.checkbox("Use built-in sample dataset (recommended)", True)

uploaded = None
if not use_demo:
    uploaded = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])

if use_demo:
    df = make_sample_churn_data()
    st.sidebar.success("✅ Loaded sample dataset")
else:
    if uploaded is None:
        st.info("Upload a dataset or enable the sample dataset.")
        st.stop()
    df = read_file(uploaded)
    st.sidebar.success("✅ Uploaded dataset loaded")

if df.shape[0] < 200:
    st.warning("This dataset is small (<200 rows). Model results may be unstable.")

st.markdown("### Dataset preview")
st.dataframe(df.head(20), use_container_width=True)

# =========================================================
# COLUMN MAPPING
# =========================================================
st.sidebar.header("🧩 Column mapping")
cols = df.columns.tolist()

# pick defaults if present
default_target_idx = cols.index("churn") if "churn" in cols else 0
target_col = st.sidebar.selectbox("Churn column (target)", cols, index=default_target_idx)

id_options = ["(none)"] + cols
default_id = "customer_id" if "customer_id" in cols else "(none)"
id_col = st.sidebar.selectbox("Customer ID column (optional)", id_options, index=id_options.index(default_id))

exclude_defaults = [c for c in cols if any(x in c.lower() for x in ["name", "email", "phone", "address"])]
exclude = st.sidebar.multiselect(
    "Exclude columns (PII/free text etc.)",
    options=[c for c in cols if c != target_col],
    default=exclude_defaults
)

work = df.drop(columns=exclude, errors="ignore").copy()

# y and X
y_raw = work[target_col]
y = normalize_churn(y_raw)

X = work.drop(columns=[target_col], errors="ignore")

if id_col != "(none)" and id_col in X.columns:
    X = X.drop(columns=[id_col])

# infer types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

st.sidebar.caption(f"Detected: {len(num_cols)} numeric + {len(cat_cols)} categorical features")

if len(num_cols) + len(cat_cols) < 2:
    st.error("Not enough usable features after exclusions. Please adjust excluded columns.")
    st.stop()

# =========================================================
# NAVIGATION
# =========================================================
st.sidebar.divider()
st.sidebar.header("🧭 Navigation")
page = st.sidebar.radio(
    "Choose module",
    ["🏠 Overview", "🚀 Train & Score", "🎯 Risk Segments", "📊 Explainable Insights"],
    label_visibility="collapsed"
)

# =========================================================
# STATE
# =========================================================
if "trained" not in st.session_state:
    st.session_state["trained"] = False
    st.session_state["pipe"] = None
    st.session_state["scored"] = None
    st.session_state["auc"] = None
    st.session_state["report"] = None
    st.session_state["cm"] = None
    st.session_state["importances"] = None


def train_model():
    # basic missing handling
    X2 = X.copy()
    for c in num_cols:
        X2[c] = X2[c].fillna(X2[c].median())
    for c in cat_cols:
        X2[c] = X2[c].fillna("Unknown").astype(str)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    # stratify only if both classes exist
    strat = y if y.nunique() == 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X2, y, test_size=0.22, random_state=42, stratify=strat
    )

    pipe.fit(X_train, y_train)

    # metrics
    if y.nunique() == 2:
        proba_test = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba_test)
        pred_test = (proba_test >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred_test)
        rep = classification_report(y_test, pred_test)
    else:
        auc, cm, rep = None, None, "Only one class present in target; classification metrics not available."

    # score all rows
    churn_prob = pipe.predict_proba(X2)[:, 1]
    scored = df.copy()
    scored["churn_probability"] = churn_prob
    scored["risk_segment"] = scored["churn_probability"].apply(risk_bucket)

    # permutation importance (only if two classes present)
    imp_df = None
    if y.nunique() == 2:
        pim = permutation_importance(
            pipe, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc"
        )
        imp_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance": pim.importances_mean
        }).sort_values("importance", ascending=False)

    st.session_state["trained"] = True
    st.session_state["pipe"] = pipe
    st.session_state["scored"] = scored
    st.session_state["auc"] = auc
    st.session_state["cm"] = cm
    st.session_state["report"] = rep
    st.session_state["importances"] = imp_df


# =========================================================
# PAGE: OVERVIEW
# =========================================================
if page == "🏠 Overview":
    st.markdown("## 🏠 Overview")

    churn_rate = y.mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Churn rate", f"{churn_rate*100:.1f}%")
    c3.metric("Features used", f"{len(num_cols)+len(cat_cols)}")
    c4.metric("Demo / Upload", "Demo" if use_demo else "Upload")

    st.info("Go to **Train & Score** to train the model and generate churn probabilities.")


# =========================================================
# PAGE: TRAIN & SCORE
# =========================================================
elif page == "🚀 Train & Score":
    st.markdown("## 🚀 Train & Score")
    st.caption("Train a churn model on your dataset and score customers by churn probability.")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### What will happen when you train?")
        st.write("- Train a logistic regression model")
        st.write("- Score each customer with churn probability")
        st.write("- Split customers into Low / Medium / High risk segments")
        st.write("- Generate explainable insights")

    with right:
        st.markdown("### Train")
        st.write("Click the button to train the model with your selected columns.")
        if st.button("🚀 Train model", type="primary", use_container_width=True):
            with st.spinner("Training model and scoring customers..."):
                train_model()
            st.success("Done! Go to **Risk Segments** and **Explainable Insights**.")

    if st.session_state["trained"]:
        st.divider()
        st.markdown("### Training summary")
        if st.session_state["auc"] is not None:
            st.metric("Test AUC", f"{st.session_state['auc']:.3f}")
        else:
            st.info("AUC not available (target column has only one class).")

        scored = st.session_state["scored"]
        st.markdown("### Churn probability distribution")
        fig, ax = plt.subplots()
        ax.hist(scored["churn_probability"], bins=30)
        ax.set_xlabel("Churn probability")
        ax.set_ylabel("Customers")
        st.pyplot(fig)


# =========================================================
# PAGE: RISK SEGMENTS
# =========================================================
elif page == "🎯 Risk Segments":
    st.markdown("## 🎯 Risk Segments")
    st.caption("Use this to generate a target list for retention campaigns.")

    if not st.session_state["trained"]:
        st.warning("Train the model first in **Train & Score**.")
        st.stop()

    scored = st.session_state["scored"]

    counts = scored["risk_segment"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    c1, c2, c3 = st.columns(3)
    c1.metric("High risk", f"{int(counts['High']):,}")
    c2.metric("Medium risk", f"{int(counts['Medium']):,}")
    c3.metric("Low risk", f"{int(counts['Low']):,}")

    st.markdown("### High-risk customers (top 50)")
    high = scored.sort_values("churn_probability", ascending=False)
    if id_col != "(none)" and id_col in df.columns:
        display_cols = [id_col, "churn_probability", "risk_segment"] + [c for c in df.columns if c not in [id_col, target_col]][:8]
    else:
        display_cols = ["churn_probability", "risk_segment"] + [c for c in df.columns if c != target_col][:10]

    st.dataframe(high[display_cols].head(50), use_container_width=True)

    st.download_button(
        "⬇️ Download high-risk target list (CSV)",
        data=high.to_csv(index=False).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("### Recommended playbooks")
    st.write("- **High risk:** win-back offer, proactive support call, downgrade option")
    st.write("- **Medium risk:** targeted nudge, product education, loyalty points")
    st.write("- **Low risk:** upsell/cross-sell, referral campaigns")


# =========================================================
# PAGE: EXPLAINABLE INSIGHTS
# =========================================================
else:
    st.markdown("## 📊 Explainable Insights")
    st.caption("Understand what drives churn risk and how the model behaves.")

    if not st.session_state["trained"]:
        st.warning("Train the model first in **Train & Score**.")
        st.stop()

    scored = st.session_state["scored"]

    st.markdown("### Churn probability histogram")
    fig, ax = plt.subplots()
    ax.hist(scored["churn_probability"], bins=30)
    ax.set_xlabel("Churn probability")
    ax.set_ylabel("Customers")
    st.pyplot(fig)

    st.markdown("### Feature importance (permutation)")
    imp_df = st.session_state["importances"]
    if imp_df is None or imp_df.empty:
        st.info("Feature importance not available (likely only one class in target).")
    else:
        fig, ax = plt.subplots()
        top = imp_df.head(12)[::-1]
        ax.barh(top["feature"], top["importance"])
        ax.set_xlabel("Importance (Δ AUC)")
        ax.set_ylabel("Feature")
        st.pyplot(fig)

    st.markdown("### Correlation heatmap (numeric features)")
    # Use numeric columns from original df; if columns missing, fallback
    numeric_for_corr = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_for_corr) >= 2:
        corr = df[numeric_for_corr].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(corr.values)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

    st.markdown("### Model performance")
    if st.session_state["auc"] is not None:
        st.metric("Test AUC", f"{st.session_state['auc']:.3f}")
        st.write("Confusion Matrix (threshold=0.5):")
        st.write(st.session_state["cm"])
        st.text("Classification Report:")
        st.text(st.session_state["report"])
    else:
        st.info("Classification metrics not available (target column may have only one class).")
