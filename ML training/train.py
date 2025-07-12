import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("../Dataset/lead.csv")

# --- FEATURE ENGINEERING ---

# 1. Encode Comments into Numeric Scores
def comment_score(text):
    text = str(text).lower()
    score = 0
    if "urgent" in text: score += 10
    if "call me now" in text: score += 5
    if "budget approved" in text: score += 8
    if "not interested" in text: score -= 15
    if "price too high" in text: score -= 5
    if "follow up" in text: score += 2
    return score

df["Comment Score"] = df["Comments"].apply(comment_score)

# 2. Encode categorical features
label_encoders = {}
categorical_cols = ["Age Group", "Family Background", "Occupation"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- MODEL TRAINING ---

# Feature list (tuned)
features = [
    "Credit Score",
    "Income",
    "Age Group",
    "Family Background",
    "Occupation",
    "Comment Score"
]
X = df[features]
y = df["Lead Intent"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosted Model with hyperparameter tuning
model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate on training set
y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"✅ Training Accuracy: {train_accuracy:.4f}")

# Evaluate on test set (optional)
y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"🧪 Test Accuracy: {test_accuracy:.4f}")

# Save the model
joblib.dump(model, "../lead_model.pkl")
print("✅ Model saved as lead_model.pkl")

# (Optional) Save label encoders
joblib.dump(label_encoders, "../label_encoders.pkl")
print("✅ Label encoders saved as label_encoders.pkl")
