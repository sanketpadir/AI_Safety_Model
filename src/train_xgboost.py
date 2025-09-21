import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


# Load dataset

df = pd.read_csv("data/ai_safety_dataset.csv")

# Text preprocessing

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special chars
    text = text.strip()
    return text

df['clean_message'] = df['message'].apply(preprocess)

# Features & Targets
X = df['clean_message']
y = df[['abuse_label', 'escalation_label', 'crisis_label', 'age_flag']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train XGBoost model

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    max_depth=6,
    n_estimators=200,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model = MultiOutputClassifier(xgb_model, n_jobs=-1)
model.fit(X_train_vec, y_train)

# Evaluate model

y_pred = model.predict(X_test_vec)

for i, col in enumerate(y.columns):
    print(f"\n--- {col} ---")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

# Save model & vectorizer

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n XGBoost model and vectorizer saved in 'models/' folder.")
