# 🧠 AI Lead Intent Scoring Dashboard

A web-based lead scoring system that predicts the intent score (0–100) of leads using a machine learning model and a rule-based LLM-style re-ranker.

---

## 🚀 Overview

**Problem:** Brokers spend time on low-intent leads, which slows conversions.  
**Solution:** This dashboard prioritizes high-intent leads using predictive analytics and comment-based heuristics.

---

## 📂 Project Structure

```
├── main.py                  # FastAPI backend
├── Frontend/
│   └── index.html           # HTML frontend
├── Dataset/
│   ├── lead.csv             # Synthetic dataset (~10,000 rows)
│   └── data.py              # Script to generate dataset
├── ML training/
│   ├── train.py             # Model training script
│   └── test_llmreranker.py  # Rule-based comment scoring test
├── lead_model.pkl           # Trained ML model
├── label_encoders.pkl       # Label encoders for categorical data
├── requirements.txt         # Python dependencies
└── README.md
```

---

## ⚙️ Tech Stack

- **Frontend:** HTML, CSS (minimal JavaScript)
- **Backend:** FastAPI (Python)
- **ML Model:** Scikit-learn's GradientBoostingClassifier
- **Reranker:** Rule-based LLM logic on comment text

---

## 📊 Dataset

- ✅ **Type:** Synthetic (~10,000 rows)
- ✅ **Generated using:** `random`, `numpy`, and `faker`
- ✅ **Fields:**
  - Email
  - Phone Number
  - Credit Score
  - Income
  - Age Group
  - Family Background
  - Occupation
  - Comments

Relationships were designed to simulate real-world lead behavior (e.g., higher income → better credit score).

---

## 🧠 Model Features

| Feature           | Source     |
|------------------|------------|
| `credit_score`    | Input      |
| `income`          | Input      |
| `age_group`       | Encoded    |
| `family_background` | Encoded |
| `occupation`      | Encoded    |
| `comment_score`   | From LLM reranker (keyword rules) |

---

## 🔄 Comment Score (LLM-style Re-ranker)

Applies rule-based scores:
```
"urgent" → +10
"important" → +5
"call back" → +5
"not interested" → -10
"later" → -5
"follow up" → +5
```
Extracted from the Comments field, passed as a numeric feature (`comment_score`) to the model.

---

## 📡 FastAPI API Endpoint

### `POST /score`

**Request Body:**
```json
{
  "email": "john@example.com",
  "phone_number": "+91-1234567890",
  "credit_score": 720,
  "income": 600000,
  "age_group": "26–35",
  "family_background": "Married",
  "occupation": "Doctor",
  "comments": "urgent call back"
}
```

**Response:**
```json
{
  "email": "john@example.com",
  "initial_score": 68,
  "reranked_score": 68,
  "comment_score": 15
}
```

---

## 🖥️ Run Locally

### 1. Clone & Install
```bash
git clone https://github.com/rajrounak21/AI-Lead-Intent-Scoring-Dashboard.git
cd AI-Lead-Intent-Scoring-Dashboard
pip install -r requirements.txt
```

### 2. Run FastAPI Server
```bash
uvicorn main:app --reload
```

### 3. Access the Dashboard
Go to [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## ✅ Assignment Compliance

- ✅ Synthetic dataset with realistic relationships
- ✅ Consent checkbox included
- ✅ Comment logic (LLM reranker)
- ✅ Intent Score (0–100)
- ✅ Rule-based + ML blended scoring
- ✅ Single page HTML dashboard + API

---

## 📩 Author

**Rounak Raj**  
📧 rajrounak366@gmail.com  
🔗 [GitHub](https://github.com/rajrounak21)

---

## 📝 License

MIT
