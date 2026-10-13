# ===========================
# LAYER 2 – NLP ENGINE
# ===========================

import joblib
import re

# Load trained TF-IDF model from Layer 1
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# ===========================
# 1️⃣ Side Effect Dictionary
# ===========================

side_effect_keywords = [
    "nausea",
    "vomiting",
    "headache",
    "dizziness",
    "fatigue",
    "rash",
    "insomnia",
    "anxiety",
    "pain",
    "diarrhea",
    "dry mouth",
    "weight gain"
]

# ===========================
# 2️⃣ Text Cleaning
# ===========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# ===========================
# 3️⃣ Side Effect Extraction
# ===========================

def extract_side_effects(text):
    text = clean_text(text)
    detected = []

    for word in side_effect_keywords:
        if word in text:
            detected.append(word)

    return detected

# ===========================
# 4️⃣ Severity Score
# ===========================

def calculate_severity(side_effects):
    count = len(side_effects)

    if count == 0:
        return "None"
    elif count == 1:
        return "Low"
    elif count <= 3:
        return "Moderate"
    else:
        return "High"

# ===========================
# 5️⃣ NLP Analyzer Function
# ===========================

def analyze_review(review_text):
    cleaned = clean_text(review_text)
    side_effects = extract_side_effects(cleaned)
    severity = calculate_severity(side_effects)

    return {
        "side_effects_detected": side_effects,
        "severity_level": severity
    }

# ===========================
# TEST RUN
# ===========================

if __name__ == "__main__":
    sample = "I experienced severe nausea and dizziness after taking this drug."
    result = analyze_review(sample)
    print(result)
