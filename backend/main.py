"""
FastAPI backend for the Credit Card Fraud Detection system.

Endpoints:
  GET  /            - API info
  GET  /health       - health check
  GET  /model-info   - model metadata + metrics
  POST /predict      - single transaction prediction
  POST /predict-batch - batch prediction via CSV upload
"""
import io
import json
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.routers import users, prediction

app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(prediction.router, prefix="/prediction", tags=["Prediction"])

MODELS_DIR = "models"
MODEL_STATE: dict = {}


def load_artifacts():
    model = joblib.load(f"{MODELS_DIR}/model.joblib")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.joblib")
    freq_maps = joblib.load(f"{MODELS_DIR}/freq_maps.joblib")
    with open(f"{MODELS_DIR}/feature_cols.json") as f:
        cols_info = json.load(f)
    with open(f"{MODELS_DIR}/metrics.json") as f:
        metrics = json.load(f)
    return {
        "model": model,
        "scaler": scaler,
        "freq_maps": freq_maps,
        "feature_cols": cols_info["feature_cols"],
        "numeric_features": cols_info["numeric_features"],
        "category_values": cols_info["category_values"],
        "threshold": metrics["threshold"],
        "metrics": metrics,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_STATE.update(load_artifacts())
    print("Model artifacts loaded. Threshold =", MODEL_STATE["threshold"])
    yield
    MODEL_STATE.clear()


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Random Forest based real-time fraud scoring API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionRequest(BaseModel):
    trans_date_trans_time: str = Field(..., description="e.g. 2024-06-21 14:32:00")
    amt: float = Field(..., gt=0)
    gender: str = Field(..., pattern="^[MF]$")
    city_pop: float = Field(..., ge=0)
    merchant: str
    job: str
    state: str
    category: str
    dob: str = Field(..., description="YYYY-MM-DD")


class BatchPredictionRow(BaseModel):
    fraud_probability: float
    is_fraud_prediction: int
    risk_level: str


def risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.3:
        return "Medium"
    return "Low"


def build_single_feature_row(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    return build_feature_matrix_for_inference(df)


def build_feature_matrix_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    state = MODEL_STATE
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day"] = df["trans_date_trans_time"].dt.day
    df["trans_weekday"] = df["trans_date_trans_time"].dt.weekday
    df["trans_month"] = df["trans_date_trans_time"].dt.month
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    df["merchant_enc"] = df["merchant"].map(state["freq_maps"]["merchant"]).fillna(0.0)
    df["job_enc"] = df["job"].map(state["freq_maps"]["job"]).fillna(0.0)
    df["state_enc"] = df["state"].map(state["freq_maps"]["state"]).fillna(0.0)

    for cat in state["category_values"]:
        df[f"category_{cat}"] = (df["category"] == cat).astype(int)

    df["gender"] = (df["gender"].astype(str).str.upper() == "M").astype(int)

    X = df[state["feature_cols"]].copy()
    X[state["numeric_features"]] = state["scaler"].transform(X[state["numeric_features"]])
    return X


@app.get("/")
def root():
    return {
        "name": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "endpoints": ["/health", "/model-info", "/predict", "/predict-batch"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in MODEL_STATE}


@app.get("/model-info")
def model_info():
    m = MODEL_STATE["metrics"]
    return {
        "model_type": "RandomForestClassifier",
        "threshold": m["threshold"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1_score": m["f1_score"],
        "pr_auc": m["pr_auc"],
        "confusion_matrix": m["confusion_matrix"],
        "train_rows": m["train_rows"],
        "test_rows": m["test_rows"],
        "test_fraud_rate": m["test_fraud_rate"],
        "best_params": m["best_params"],
        "top_features": dict(
            sorted(m["feature_importances"].items(), key=lambda x: -x[1])[:10]
        ),
    }


@app.post("/predict")
def predict(tx: TransactionRequest):
    if "model" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        X = build_single_feature_row(tx.model_dump())
        proba = float(MODEL_STATE["model"].predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature processing error: {e}")

    threshold = MODEL_STATE["threshold"]
    return {
        "fraud_probability": round(proba, 6),
        "is_fraud_prediction": int(proba >= threshold),
        "risk_level": risk_level(proba),
        "threshold_used": threshold,
    }


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if "model" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    required = ["trans_date_trans_time", "amt", "gender", "city_pop",
                "merchant", "job", "state", "category", "dob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    try:
        X = build_feature_matrix_for_inference(df)
        proba = MODEL_STATE["model"].predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature processing error: {e}")

    threshold = MODEL_STATE["threshold"]
    df_out = df.copy()
    df_out["fraud_probability"] = np.round(proba, 6)
    df_out["is_fraud_prediction"] = (proba >= threshold).astype(int)
    df_out["risk_level"] = [risk_level(p) for p in proba]

    stream = io.StringIO()
    df_out.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
