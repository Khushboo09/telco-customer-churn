# Day 2: FastAPI REST API for churn prediction.
#
# POST /predict:
#   1. Accepts customer information as JSON
#   2. Applies src.feature_engineering.engineer_features() to compute TotalServices/tenure_group
#      (the SAME function used by the notebook - do not reimplement this logic here)
#   3. Applies the saved preprocessing pipeline
#   4. Loads the trained model from model/churn_model.pkl
#   5. Returns the churn prediction and probability

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.feature_engineering import engineer_features
from src.schemas import CustomerData, PredictionResponse

logger = logging.getLogger("uvicorn.error")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "churn_model.pkl"

# Load the saved pipeline (preprocessing + Decision Tree) once, when the module
# is imported / the app process starts - not per-request, and never retrained here.
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Could not find saved model at {MODEL_PATH}. Run notebook/churn_analysis.ipynb "
        "through Section 12 to train and save it before starting the API."
    ) from exc

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predicts whether a customer will churn from raw customer attributes.",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict:
    """Liveness check confirming the app is running and the model is loaded."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData) -> PredictionResponse:
    """Predict churn for a single customer.

    Flow: validated request -> single-row DataFrame -> engineer_features()
    (adds TotalServices/tenure_group) -> saved pipeline's predict()/predict_proba().
    """
    try:
        raw_df = pd.DataFrame([customer.model_dump()])
        engineered_df = engineer_features(raw_df)

        prediction = model.predict(engineered_df)[0]
        probabilities = model.predict_proba(engineered_df)[0]

        classes = list(model.named_steps["classifier"].classes_)
        churn_probability = float(probabilities[classes.index("Yes")])
    except Exception as exc:
        logger.exception("Prediction failed for request: %s", customer.model_dump())
        raise HTTPException(
            status_code=400, detail="Could not generate a prediction for the given input."
        ) from exc

    return PredictionResponse(
        churn_prediction=str(prediction),
        churn_probability=round(churn_probability, 4),
    )

