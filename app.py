import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    auc
)

import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ML Classification Dashboard | Vivek",
    page_icon="🧠",
    layout="wide"
)

# =========================
# SESSION STATE FOR THEME
# =========================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🎨 Appearance")

theme_toggle = st.sidebar.toggle("Dark Mode")

st.session_state.theme = "dark" if theme_toggle else "light"

# =========================
# THEME CSS + ANIMATIONS
# =========================
if st.session_state.theme == "dark":
    bg = "#0e1117"
    card = "#161b22"
    text = "#ffffff"
else:
    bg = "#f5f7fb"
    card = "#ffffff"
    text = "#000000"

st.markdown(f"""
<style>

/* GLOBAL */
html, body, [class*="css"] {{
    background-color: {bg};
    color: {text};
    transition: all 0.4s ease-in-out;
}}

/* HEADER */
.header {{
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    animation: slideDown 0.8s ease-out;
}}

/* CARDS */
.metric-card {{
    background-color: {card};
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    animation: fadeUp 0.8s ease;
}}

.metric-card:hover {{
    transform: scale(1.04);
    transition: 0.3s;
}}

/* ANIMATIONS */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes slideDown {{
    from {{ opacity: 0; transform: translateY(-30px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* FOOTER */
.footer {{
    text-align:center;
    color: gray;
    font-size: 14px;
    animation: fadeUp 1s ease;
}}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(
    "<div class='header'>"
    "<h1>🧠 ML Classification Dashboard</h1>"
    "<p>End-to-End ML | Training → Evaluation → Deployment</p>"
    "</div>",
    unsafe_allow_html=True
)

# =========================
# MODEL MAP
# =========================
MODEL_MAP = {
    "Logistic Regression": "model/logisticregression.pkl",
    "Decision Tree": "model/decisiontree.pkl",
    "KNN": "model/kneighbors.pkl",
    "Naive Bayes": "model/gaussiannb.pkl",
    "Random Forest": "model/randomforest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

MODEL_INFO = {
    "Logistic Regression": "Linear baseline, interpretable.",
    "Decision Tree": "Rule-based, may overfit.",
    "KNN": "Distance-based learner.",
    "Naive Bayes": "Fast probabilistic model.",
    "Random Forest": "Strong ensemble learner.",
    "XGBoost": "Boosted ensemble with best performance."
}

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("📂 Upload Test CSV", type=["csv"])
selected_model = st.sidebar.selectbox("🤖 Select Model", MODEL_MAP.keys())
st.sidebar.info(MODEL_INFO[selected_model])

# =========================
# MAIN LOGIC
# =========================
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    scaler = joblib.load("model/scaler.pkl")
    X_scaled = scaler.transform(X)

    model = joblib.load(MODEL_MAP[selected_model])
    y_pred = model.predict(X_scaled)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Metrics", "📉 Confusion Matrix", "📈 ROC Curve", "⬇️ Download"]
    )

    # ===== METRICS =====
    with tab1:
        cols = st.columns(3)
        metrics = [
            ("Accuracy", accuracy_score(y, y_pred)),
            ("Precision", precision_score(y, y_pred)),
            ("Recall", recall_score(y, y_pred)),
            ("F1 Score", f1_score(y, y_pred)),
            ("MCC", matthews_corrcoef(y, y_pred)),
            ("Samples", len(y))
        ]

        for col, (name, val) in zip(cols * 2, metrics):
            col.markdown(
                f"<div class='metric-card'><h3>{name}</h3><h2>{round(val,4)}</h2></div>",
                unsafe_allow_html=True
            )

    # ===== CONFUSION MATRIX =====
    with tab2:
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    # ===== ROC CURVE =====
    with tab3:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)

            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax2.plot([0,1],[0,1],'--')
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.warning("ROC Curve not supported.")

    # ===== DOWNLOAD =====
    with tab4:
        out = data.copy()
        out["Prediction"] = y_pred
        st.download_button(
            "⬇️ Download Predictions CSV",
            out.to_csv(index=False),
            "predictions.csv"
        )

else:
    st.info("⬅️ Upload a CSV file to start.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<div class='footer'>🚀 Built by Vivek | ML Assignment 2 | BITS Pilani</div>",
    unsafe_allow_html=True
)
