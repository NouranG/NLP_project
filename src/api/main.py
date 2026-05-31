from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ----------------------------------------------------
# Load artifacts once at startup
# ----------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

artifacts = joblib.load(
    "models/sentiment_pipeline.pkl"
)

model = artifacts["model"]
vectorizer = artifacts["vectorizer"]
preprocessor = artifacts["preprocessor"]


# ----------------------------------------------------
# FastAPI
# ----------------------------------------------------

app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0"
)


class ReviewRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API is running"
    }


@app.post("/predict")
def predict(request: ReviewRequest):

    cleaned_text = preprocessor.process(request.text)

    X = vectorizer.transform(
        pd.Series([cleaned_text])
    )

    prediction = int(
        model.predict(X)[0]
    )

    sentiment = (
        "positive"
        if prediction == 1
        else "negative"
    )

    return {
        "text": request.text,
        "cleaned_text": cleaned_text,
        "prediction": prediction,
        "sentiment": sentiment
    }