"""FastAPI service for real-time churn prediction with SHAP-based explanations."""
import json
import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Churn Prediction API",
    description="Production-ready churn model with XGBoost + SHAP interpretability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file, so working directory doesn't matter)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")

# ---------------------------------------------------------------------------
# Load artifacts at startup
# ---------------------------------------------------------------------------
MODEL = joblib.load(os.path.join(MODEL_DIR, "churn_model.joblib"))
EXPLAINER = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.joblib"))

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    METADATA = json.load(f)

FEATURE_NAMES = METADATA["feature_names"]
CATEGORICAL_COLS = METADATA["categorical_cols"]
ENCODERS = METADATA["encoders"]
DEFAULTS = METADATA["defaults"]


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class CustomerData(BaseModel):
    gender: Optional[str] = Field(default=None, examples=["Female"])
    SeniorCitizen: Optional[int] = Field(default=None, examples=[0])
    Partner: Optional[str] = Field(default=None, examples=["Yes"])
    Dependents: Optional[str] = Field(default=None, examples=["No"])
    tenure: Optional[int] = Field(default=None, examples=[12])
    PhoneService: Optional[str] = Field(default=None, examples=["Yes"])
    MultipleLines: Optional[str] = Field(default=None, examples=["No"])
    InternetService: Optional[str] = Field(default=None, examples=["Fiber optic"])
    OnlineSecurity: Optional[str] = Field(default=None, examples=["No"])
    OnlineBackup: Optional[str] = Field(default=None, examples=["Yes"])
    DeviceProtection: Optional[str] = Field(default=None, examples=["No"])
    TechSupport: Optional[str] = Field(default=None, examples=["No"])
    StreamingTV: Optional[str] = Field(default=None, examples=["Yes"])
    StreamingMovies: Optional[str] = Field(default=None, examples=["Yes"])
    Contract: Optional[str] = Field(default=None, examples=["Month-to-month"])
    PaperlessBilling: Optional[str] = Field(default=None, examples=["Yes"])
    PaymentMethod: Optional[str] = Field(default=None, examples=["Electronic check"])
    MonthlyCharges: Optional[float] = Field(default=None, examples=[70.35])
    TotalCharges: Optional[float] = Field(default=None, examples=[845.5])


class Factor(BaseModel):
    feature: str
    value: Optional[float]
    shap_contribution: float
    effect: str


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: str
    top_factors: list[Factor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode_input(data: dict) -> pd.DataFrame:
    """Fill missing fields with defaults, label-encode categoricals."""
    row = {}
    for feat in FEATURE_NAMES:
        value = data.get(feat)
        if value is None:
            value = DEFAULTS.get(feat)
        row[feat] = value

    for col in CATEGORICAL_COLS:
        mapping = ENCODERS[col]
        raw_val = str(row[col])
        if raw_val not in mapping:
            raw_val = str(DEFAULTS.get(col))
        row[col] = mapping.get(raw_val, 0)

    df = pd.DataFrame([row])

    # Ensure numeric dtypes
    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[FEATURE_NAMES]


def explain_prediction(df: pd.DataFrame, top_n: int = 5):
    shap_values = EXPLAINER.shap_values(df)

    # Handle both SHAP API styles:
    # - list of arrays (one per class) -> use class 1 (churn), row 0
    # - single array of shape (n_samples, n_features) -> row 0
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    contributions = list(zip(FEATURE_NAMES, df.iloc[0].tolist(), values.tolist()))
    contributions.sort(key=lambda x: abs(x[2]), reverse=True)

    top_factors = [
        {
            "feature": feat,
            "value": val,
            "shap_contribution": round(shap_val, 4),
            "effect": "increases churn risk" if shap_val > 0 else "decreases churn risk",
        }
        for feat, val, shap_val in contributions[:top_n]
    ]
    return top_factors


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api")
def api_info():
    return {
        "message": "Churn Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "metrics": METADATA.get("metrics"),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        data = customer.model_dump()
        df = encode_input(data)

        proba = float(MODEL.predict_proba(df)[0, 1])
        prediction = "Churn" if proba >= 0.5 else "No Churn"
        top_factors = explain_prediction(df)

        return PredictionResponse(
            churn_probability=round(proba, 4),
            churn_prediction=prediction,
            top_factors=top_factors,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Mount static files after all API routes are defined
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")