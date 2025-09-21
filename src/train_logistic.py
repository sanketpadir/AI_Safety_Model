# src/train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import re
import os

# Load dataset

df = pd.read_csv(r"data\ai_safety_dataset.csv")

# Text preprocessing

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.strip()
    return text

df['clean_message'] = df['message'].apply(preprocess)

# Train-test split

X = df['clean_message']
y = df[['abuse_label', 'escalation_label', 'crisis_label', 'age_flag']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train ML model

model = MultiOutputClassifier(LogisticRegression(max_iter=500))
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
for i, col in enumerate(y.columns):
    print(f"--- {col} ---")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

# Save model and vectorizer

# Create models folder if it doesn't exist
current_dir = os.path.dirname(__file__)           # src/
models_dir = os.path.join(current_dir, "..", "models")
os.makedirs(models_dir, exist_ok=True)

# Save ML model
with open(os.path.join(models_dir, "ml_model.pkl"), "wb") as f:
    pickle.dump(model, f)

# Save TF-IDF vectorizer
with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved in 'models/' folder successfully!")
