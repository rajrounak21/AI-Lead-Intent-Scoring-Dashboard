from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
import numpy as np
import joblib

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model and encoders
model = joblib.load("lead_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Templates (optional)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Input model
class Lead(BaseModel):
    email: EmailStr
    phone_number: str
    credit_score: int
    income: float
    age_group: str
    family_background: str
    occupation: str
    comments: str

# Keyword scores (LLM-style reranker)
keyword_scores = {
    "urgent": 10,
    "important": 5,
    "call back": 5,
    "not interested": -10,
    "later": -5,
    "follow up": 5
}

def llm_reranker(score: int, comment: str) -> int:
    comment = comment.lower()
    adjustment = sum(v for k, v in keyword_scores.items() if k in comment)
    return max(0, min(100, score + adjustment))

@app.post("/score")
async def score_lead(lead: Lead):
    # Input validation
    if lead.credit_score < 0 or lead.income < 0:
        raise HTTPException(status_code=400, detail="Invalid numeric input")

    # Safe category transform
    def safe_transform(encoder, value, field_name):
        if value not in encoder.classes_:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value '{value}' for {field_name}. Expected one of: {list(encoder.classes_)}"
            )
        return encoder.transform([value])[0]

    encoded_age = safe_transform(encoders["Age Group"], lead.age_group, "Age Group")
    encoded_family = safe_transform(encoders["Family Background"], lead.family_background, "Family Background")
    encoded_occupation = safe_transform(encoders["Occupation"], lead.occupation, "Occupation")

    # Comment score used as input feature
    comment_input_score = sum(v for k, v in keyword_scores.items() if k in lead.comments.lower())

    # Model input features
    features = np.array([
        lead.credit_score,
        lead.income,
        encoded_age,
        encoded_family,
        encoded_occupation,
        comment_input_score
    ]).reshape(1, -1)

    # Model prediction
    initial_score = int(model.predict_proba(features)[0][1] * 100)

    # Internal income logic (silent in frontend)
    if lead.income < 20000:
        adjusted_score = max(0, initial_score - 5)
        final_score = llm_reranker(adjusted_score, lead.comments)
    elif lead.income > 30000:
        final_score = llm_reranker(initial_score, lead.comments)
    else:
        final_score = initial_score

    return {
        "email": lead.email,
        "initial_score": initial_score,
        "reranked_score": final_score,
        "comment_score": comment_input_score
    }
