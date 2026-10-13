# ===========================
# LAYER 1 – CORE ML ENGINE
# ===========================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from scipy.sparse import hstack
from textblob import TextBlob
from nlp_engine import analyze_review

# ===========================
# 1️⃣ LOAD DATA
# ===========================

print("Loading dataset...")

train_df = pd.read_csv("data/drugsComTrain_raw.csv")
test_df = pd.read_csv("data/drugsComTest_raw.csv")

df = pd.concat([train_df, test_df], ignore_index=True)
df = df.sample(n=60000, random_state=42)

print("Dataset loaded. Shape:", df.shape)

# ===========================
# 2️⃣ BASIC CLEANING
# ===========================

df = df.dropna(subset=["review", "condition", "drugName", "rating"])

df["outcome_effective"] = (df["rating"] >= 7).astype(int)
df["review"] = df["review"].str.lower()

# ===========================
# 3️⃣ NLP FEATURES
# ===========================

print("Computing NLP features...")

# Sentiment Score
df["sentiment_score"] = df["review"].apply(
    lambda x: (TextBlob(str(x)).sentiment.polarity + 1) / 2
)

# Severity Score
severity_map = {
    "None": 0.0,
    "Low": 0.2,
    "Moderate": 0.5,
    "High": 0.8
}

df["severity_score"] = df["review"].apply(
    lambda x: severity_map.get(
        analyze_review(str(x))["severity_level"], 0.3
    )
)

# ===========================
# 4️⃣ FEATURE ENGINEERING
# ===========================

df["review_length"] = df["review"].apply(lambda x: len(str(x).split()))
df["useful_ratio"] = df["usefulCount"] / df["usefulCount"].max()

le_drug = LabelEncoder()
le_condition = LabelEncoder()

df["drug_encoded"] = le_drug.fit_transform(df["drugName"])
df["condition_encoded"] = le_condition.fit_transform(df["condition"])

# ===========================
# 5️⃣ TF-IDF VECTORIZATION
# ===========================

print("Performing TF-IDF vectorization...")

tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_text = tfidf.fit_transform(df["review"])

# ===========================
# 6️⃣ STRUCTURED FEATURES (6 FEATURES)
# ===========================

structured_features = df[[
    "drug_encoded",
    "condition_encoded",
    "review_length",
    "useful_ratio",
    "sentiment_score",
    "severity_score"
]].values

X = hstack([X_text, structured_features])
y = df["outcome_effective"]

print("Final Feature Shape:", X.shape)

# ===========================
# 7️⃣ TRAIN-TEST SPLIT
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===========================
# 8️⃣ MODEL TRAINING
# ===========================

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ===========================
# 9️⃣ EVALUATION
# ===========================

print("Evaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===========================
# 🔟 SAVE MODEL
# ===========================

joblib.dump(model, "models/drug_effectiveness_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(le_drug, "models/label_encoder_drug.pkl")
joblib.dump(le_condition, "models/label_encoder_condition.pkl")

print("\n✅ Layer 1 Model Saved Successfully!")