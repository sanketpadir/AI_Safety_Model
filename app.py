# app.py
import streamlit as st
import pickle
import re
import os

# Load model and vectorizer

current_dir = os.path.dirname(__file__)
models_dir = os.path.join(current_dir, "models")

with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(models_dir, "ml_model.pkl"), "rb") as f:
    model = pickle.load(f)

# Preprocessing function

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.strip()
    return text

# Streamlit UI

st.title("AI Safety Model Demo")
st.write("Enter a message to detect abuse, escalation, crisis, and age-appropriateness.")

user_input = st.text_area("Enter your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        # Preprocess and transform
        msg_clean = preprocess(user_input)
        msg_vec = vectorizer.transform([msg_clean])
        
        # Make prediction
        pred = model.predict(msg_vec)
        labels = ['abuse_label', 'escalation_label', 'crisis_label', 'age_flag']
        
        # Show results
        st.subheader("Prediction Results:")
        for label, value in zip(labels, pred[0]):
            st.write(f"**{label}**: {value}")
