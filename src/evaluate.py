import os
import pandas as pd
import pickle
import re
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Define project directories

script_dir = os.path.dirname(os.path.abspath(__file__))  # src/
project_root = os.path.abspath(os.path.join(script_dir, ".."))
data_path = os.path.join(project_root, "data", "ai_safety_dataset.csv")
models_dir = os.path.join(project_root, "models")

# Load dataset

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)

# Text preprocessing

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.strip()
    return text

df['clean_message'] = df['message'].apply(preprocess)

# Features and labels

X = df['clean_message']
y = df[['abuse_label', 'escalation_label', 'crisis_label', 'age_flag']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load vectorizer

vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

X_test_vec = vectorizer.transform(X_test)

# Load chosen model

model_name = input("Enter model to evaluate (ml_model.pkl / xgb_model.pkl): ").strip()
model_path = os.path.join(models_dir, model_name)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Predict & evaluate

y_pred = model.predict(X_test_vec)

print("\n--- Classification Report ---\n")
for i, col in enumerate(y.columns):
    print(f"Label: {col}")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))
    print("-" * 50)
