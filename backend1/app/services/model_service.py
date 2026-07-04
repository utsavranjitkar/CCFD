from pathlib import Path
import json
import joblib
import io
import pandas as pd
import numpy as np

def get_form_options():

    return {
        "merchants": sorted(MODEL_STATE["freq_maps"]["merchant"].keys()),
        "jobs": sorted(MODEL_STATE["freq_maps"]["job"].keys()),
        "states": sorted(MODEL_STATE["freq_maps"]["state"].keys()),
        "categories": sorted(MODEL_STATE["category_values"])
    }

# Absolute path to app/model_artifacts
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "model_artifacts"

MODEL_STATE = {}


def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    freq_maps = joblib.load(MODELS_DIR / "freq_maps.joblib")

    with open(MODELS_DIR / "feature_cols.json") as f:
        cols_info = json.load(f)

    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    MODEL_STATE.update({
        "model": model,
        "scaler": scaler,
        "freq_maps": freq_maps,
        "feature_cols": cols_info["feature_cols"],
        "numeric_features": cols_info["numeric_features"],
        "category_values": cols_info["category_values"],
        "threshold": metrics["threshold"],
        "metrics": metrics,
    })


def risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.3:
        return "Medium"
    return "Low"


def build_single_feature_row(payload: dict):
    df = pd.DataFrame([payload])
    return build_feature_matrix_for_inference(df)


def build_feature_matrix_for_inference(df: pd.DataFrame):
    state = MODEL_STATE

    df = df.copy()

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day"] = df["trans_date_trans_time"].dt.day
    df["trans_weekday"] = df["trans_date_trans_time"].dt.weekday
    df["trans_month"] = df["trans_date_trans_time"].dt.month
    df["age"] = (
        df["trans_date_trans_time"] - df["dob"]
    ).dt.days // 365

    df["merchant_enc"] = (
        df["merchant"]
        .map(state["freq_maps"]["merchant"])
        .fillna(0.0)
    )

    df["job_enc"] = (
        df["job"]
        .map(state["freq_maps"]["job"])
        .fillna(0.0)
    )

    df["state_enc"] = (
        df["state"]
        .map(state["freq_maps"]["state"])
        .fillna(0.0)
    )

    for cat in state["category_values"]:
        df[f"category_{cat}"] = (
            df["category"] == cat
        ).astype(int)

    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.upper()
        .eq("M")
        .astype(int)
    )

    X = df[state["feature_cols"]].copy()

    X[state["numeric_features"]] = state["scaler"].transform(
        X[state["numeric_features"]]
    )

    return X

def predict_single(payload: dict):

    X = build_single_feature_row(payload)

    probability = float(
        MODEL_STATE["model"]
        .predict_proba(X)[0, 1]
    )

    threshold = MODEL_STATE["threshold"]

    return {
        "fraud_probability": round(probability, 6),
        "is_fraud_prediction": int(probability >= threshold),
        "risk_level": risk_level(probability),
        "threshold_used": threshold
    }


def get_model_info():

    return MODEL_STATE["metrics"]

def predict_batch_dataframe(df: pd.DataFrame):

    required = [
        "trans_date_trans_time",
        "amt",
        "gender",
        "city_pop",
        "merchant",
        "job",
        "state",
        "category",
        "dob",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = build_feature_matrix_for_inference(df)

    probabilities = MODEL_STATE["model"].predict_proba(X)[:, 1]

    threshold = MODEL_STATE["threshold"]

    output = df.copy()

    output["fraud_probability"] = np.round(probabilities, 6)

    output["is_fraud_prediction"] = (
        probabilities >= threshold
    ).astype(int)

    output["risk_level"] = [
        risk_level(p)
        for p in probabilities
    ]

    return output