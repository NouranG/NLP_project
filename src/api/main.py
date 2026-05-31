from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# --------------------------------------------------
# Load artifacts
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

artifact_path = ROOT / "models" / "embeddings_random_forest.pkl"

if not artifact_path.exists():
    raise FileNotFoundError(
        f"Artifact not found: {artifact_path}"
    )

pipeline = joblib.load(artifact_path)

model = pipeline["model"]
vectorizer = pipeline["vectorizer"]
preprocessor = pipeline["preprocessor"]


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------

app = FastAPI(
    title="NLP Sentiment Analysis",
    version="1.0.0",
    description="Amazon/Twitter Sentiment Analysis API"
)


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: int
    sentiment: str
    cleaned_text: str


@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    try:

        cleaned_text = preprocessor.process(
            request.text
        )

        X = pd.Series([cleaned_text])

        X_vec = vectorizer.transform(X)

        prediction = int(
            model.predict(X_vec)[0]
        )

        sentiment = (
            "positive"
            if prediction == 1
            else "negative"
        )

        return PredictionResponse(
            prediction=prediction,
            sentiment=sentiment,
            cleaned_text=cleaned_text
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )