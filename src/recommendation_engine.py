# ===========================
# CLEAN WORKING ML ENGINE
# ===========================

import joblib
import numpy as np
from scipy.sparse import hstack
from textblob import TextBlob
from nlp_engine import analyze_review

# Load trained objects
model = joblib.load("models/drug_effectiveness_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
le_drug = joblib.load("models/label_encoder_drug.pkl")
le_condition = joblib.load("models/label_encoder_condition.pkl")

severity_map = {
    "None": 0.0,
    "Low": 0.2,
    "Moderate": 0.5,
    "High": 0.8
}


def calculate_drug_impact_index(effectiveness_prob, sentiment_score, severity_score,
                                w_ml=0.6, w_sent=0.3, w_sev=0.3):

    ml_component = w_ml * effectiveness_prob
    sentiment_component = w_sent * sentiment_score
    severity_component = w_sev * severity_score

    impact = ml_component + sentiment_component - severity_component
    impact = np.clip(impact, 0, 1)

    return {
        "impact_score": round(float(impact), 3),
        "ml_component": round(float(ml_component), 3),
        "sentiment_component": round(float(sentiment_component), 3),
        "severity_component": round(float(severity_component), 3),
    }


def generate_recommendation(review_text, drug_name, condition):

    # NLP Analysis
    nlp_result = analyze_review(review_text)
    severity = nlp_result["severity_level"]
    side_effects = nlp_result["side_effects_detected"]

    sentiment_score = (TextBlob(review_text).sentiment.polarity + 1) / 2
    severity_score = severity_map.get(severity, 0.3)

    review_lower = review_text.lower()
    review_length = len(review_lower.split())
    useful_ratio = 0

    try:
        drug_encoded = le_drug.transform([drug_name])[0]
    except:
        drug_encoded = 0

    try:
        condition_encoded = le_condition.transform([condition])[0]
    except:
        condition_encoded = 0

    X_text = tfidf.transform([review_lower])

    structured_features = np.array([[
        drug_encoded,
        condition_encoded,
        review_length,
        useful_ratio,
        sentiment_score,
        severity_score
    ]])

    X = hstack([X_text, structured_features])

    effectiveness_prob = float(model.predict_proba(X)[0][1])

    impact_data = calculate_drug_impact_index(
        effectiveness_prob,
        sentiment_score,
        severity_score
    )

    drug_impact_index = impact_data["impact_score"]

    if drug_impact_index >= 0.6:
        recommendation = "Recommended"
    elif drug_impact_index >= 0.4:
        recommendation = "Monitor Closely"
    else:
        recommendation = "Not Recommended"

    return {
        "effectiveness_probability": round(effectiveness_prob, 3),
        "side_effects_detected": side_effects,
        "severity_level": severity,
        "drug_impact_index": drug_impact_index,
        "impact_breakdown": impact_data,
        "final_recommendation": recommendation
    }