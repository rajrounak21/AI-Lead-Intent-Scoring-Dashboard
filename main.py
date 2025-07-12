from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
import joblib
import numpy as np
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust in production)
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model and encoders
model = joblib.load("lead_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Keyword scores for comment score (LLM-style reranker)
keyword_scores = {
    "urgent": 10,
    "important": 5,
    "call back": 5,
    "not interested": -10,
    "later": -5,
    "follow up": 5
}

# Templates (optional if serving HTML via Jinja2)
templates = Jinja2Templates(directory="templates")

# Serve frontend HTML
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Input model for lead
class Lead(BaseModel):
    email: EmailStr
    phone_number: str
    credit_score: int
    income: float
    age_group: str
    family_background: str
    occupation: str
    comments: str

@app.post("/score")
async def score_lead(lead: Lead):
    # Input validation
    if lead.credit_score < 0 or lead.income < 0:
        raise HTTPException(status_code=400, detail="Invalid numeric input")

    # Encode categorical features
    encoded_age = encoders["Age Group"].transform([lead.age_group])[0]
    encoded_family = encoders["Family Background"].transform([lead.family_background])[0]
    encoded_occupation = encoders["Occupation"].transform([lead.occupation])[0]

    # Calculate comment score using keywords
    comment_text = lead.comments.lower()
    comment_score = sum(v for k, v in keyword_scores.items() if k in comment_text)

    # Prepare input array (6 features)
    features = np.array([
        lead.credit_score,
        lead.income,
        encoded_age,
        encoded_family,
        encoded_occupation,
        comment_score
    ]).reshape(1, -1)

    # Predict intent score
    predicted_score = int(model.predict_proba(features)[0][1] * 100)

    return {
        "email": lead.email,
        "initial_score": predicted_score,
        "reranked_score": predicted_score,  # reranker already included
        "comment_score": comment_score
    }
