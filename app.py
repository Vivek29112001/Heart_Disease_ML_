import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="ML Assignment 2",
    layout="wide"
)

st.title("🧠 Machine Learning Model Evaluation – Classification App")

st.markdown(
    """
    Upload a **test CSV file**, select a **trained ML model** and
    view evaluation metrics and confusion matrix.
    """
)

# =========================
# Model File Mapping
# =========================
MODEL_MAP = {
    "Logistic Regression": "model/logisticregression.pkl",
    "Decision Tree": "model/decisiontree.pkl",
    "KNN": "model/kneighbors.pkl",
    "Naive Bayes": "model/gaussiannb.pkl",
    "Random Forest": "model/randomforest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_MAP.keys())
)

# =========================
# Main Logic
# =========================
if uploaded_file is not None:

    # Load data
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Features & Target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Load scaler
    scaler = joblib.load("model/scaler.pkl")
    X_scaled = scaler.transform(X)

    # Load model
    model_path = MODEL_MAP[selected_model]
    model = joblib.load(model_path)

    # Prediction
    y_pred = model.predict(X_scaled)

    # =========================
    # Metrics
    # =========================
    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
    col2.metric("Precision", round(precision_score(y, y_pred), 4))
    col3.metric("Recall", round(recall_score(y, y_pred), 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1_score(y, y_pred), 4))
    col5.metric("MCC", round(matthews_corrcoef(y, y_pred), 4))
    col6.metric("Total Samples", len(y))

    # =========================
    # Confusion Matrix
    # =========================
    st.subheader("📉 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{selected_model} – Confusion Matrix")

    st.pyplot(fig)

else:
    st.info("⬅️ Please upload a test CSV file from the sidebar.")
