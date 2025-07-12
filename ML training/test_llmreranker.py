import joblib
import numpy as np

# Load model and encoders
model = joblib.load("../lead_model.pkl")
label_encoders = joblib.load("../label_encoders.pkl")

# --- Define LLM-inspired Re-ranker ---
def llm_reranker(initial_score: int, comments: str) -> int:
    comments = comments.lower()
    adjustment = 0

    if "urgent" in comments:
        adjustment += 10
    if "call me now" in comments:
        adjustment += 5
    if "budget approved" in comments:
        adjustment += 8
    if "follow up" in comments:
        adjustment += 2
    if "not interested" in comments:
        adjustment -= 15
    if "price too high" in comments:
        adjustment -= 5

    return max(0, min(100, initial_score + adjustment))


# --- Simulate One Test Lead ---
sample_lead = {
    "credit_score": 320,
    "income": 8500,
    "age_group": "26–35",
    "family_background": "Married with Kids",
    "occupation": "Teacher",
    "comments": "Call me now, it's urgent",
}

# Encode fields
age_encoded = label_encoders["Age Group"].transform([sample_lead["age_group"]])[0]
family_encoded = label_encoders["Family Background"].transform([sample_lead["family_background"]])[0]
occupation_encoded = label_encoders["Occupation"].transform([sample_lead["occupation"]])[0]

# Create input vector
def comment_score(text: str) -> int:
    text = text.lower()
    score = 0
    if "urgent" in text: score += 10
    if "call me now" in text: score += 5
    if "budget approved" in text: score += 8
    if "not interested" in text: score -= 15
    if "price too high" in text: score -= 5
    if "follow up" in text: score += 2
    return score

comment_numeric_score = comment_score(sample_lead["comments"])

X_input = np.array([
    sample_lead["credit_score"],
    sample_lead["income"],
    age_encoded,
    family_encoded,
    occupation_encoded,
    comment_numeric_score
]).reshape(1, -1)

# Predict
initial_score = int(model.predict_proba(X_input)[0][1] * 100)
reranked_score = llm_reranker(initial_score, sample_lead["comments"])

# Print Results
print(f"✅ Initial Score from ML model: {initial_score}")
print(f"🤖 Reranked Score after LLM Re-ranker: {reranked_score}")
