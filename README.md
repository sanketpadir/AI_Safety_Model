# AI Safety Models – Proof of Concept (POC)

# Overview

This project is a Proof of Concept (POC) for AI Safety Models in a conversational AI platform.
The models perform real-time detection of:

Abuse Language Detection – harmful/inappropriate content

Escalation Pattern Recognition – emotionally dangerous conversations

Crisis Intervention – severe distress or self-harm

Content Filtering – age-appropriate filtering for minors

A Streamlit app demonstrates the POC with live predictions, now in an interactive and attractive layout with color-coded outputs.

# Project Structure

solulab_project/
│
├── data/                  # Dataset (ai_safety_dataset.csv)
├── models/                # Saved models and vectorizers
├── src/                   # Source code
│   ├── train.py           # Logistic Regression training
│   ├── train_xgboost.py   # XGBoost training
│   ├── evaluate.py        # Evaluation script
│   └── preprocess.py      # Preprocessing utilities
├── app.py                 # Streamlit app for inference (interactive & color-coded)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

# Setup Instructions

# 1. Create & activate virtual environment

python -m venv solulab_project
solulab_project\Scripts\activate   # Windows
# OR
source solulab_project/bin/activate  # Linux/Mac

# 2. Install dependencies

pip install -r requirements.txt

# 3. Ensure dataset is present

data/ai_safety_dataset.csv

# 4. Training Models

python src/train.py           # Train Logistic Regression model
python src/train_xgboost.py   # Train XGBoost model

# 5. Evaluating Models

python src/evaluate.py

# 6. Running the Streamlit App

streamlit run app.py
