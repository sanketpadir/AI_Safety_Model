# app.py

import os
import pickle
import re
import streamlit as st
import pandas as pd

# -------------------------
# Project directories
# -------------------------
project_root = os.path.abspath(os.path.dirname(__file__))
models_dir = os.path.join(project_root, "models")
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
ml_model_path = os.path.join(models_dir, "ml_model.pkl")
xgb_model_path = os.path.join(models_dir, "xgb_model.pkl")

# -------------------------
# Load vectorizer and models
# -------------------------
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

with open(ml_model_path, "rb") as f:
    ml_model = pickle.load(f)

with open(xgb_model_path, "rb") as f:
    xgb_model = pickle.load(f)

# -------------------------
# Preprocessing function
# -------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.strip()
    return text

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Safety POC", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Safety Models – Proof of Concept")
st.write(
    "This app detects Abuse, Escalation, Crisis, and Age-flag in user messages. "
    "Choose a model and enter a message to see predictions."
)

# Sidebar for model selection
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "XGBoost"])

st.markdown("---")

# Input
user_input = st.text_area("Enter message here:", height=100)

if st.button("Predict") and user_input:
    clean_text = preprocess(user_input)
    vec_text = vectorizer.transform([clean_text])

    if model_choice == "Logistic Regression":
        pred = ml_model.predict(vec_text)
    else:
        pred = xgb_model.predict(vec_text)

    labels = ["Abuse", "Escalation", "Crisis", "Age-flag"]
    result = {label: int(pred[0][i]) for i, label in enumerate(labels)}

    # Display results in columns with color coding
    st.subheader("Prediction Results")
    col1, col2, col3, col4 = st.columns(4)
    for i, (label, value) in enumerate(result.items()):
        color = "green" if value == 0 else "red"
        with [col1, col2, col3, col4][i]:
            st.markdown(f"<h3 style='color:{color}'>{label}: {value}</h3>", unsafe_allow_html=True)

st.markdown("---")
st.write("Developed as a Proof of Concept for AI Safety Models.")
