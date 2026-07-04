import io
import pandas as pd

from datetime import datetime

from sqlalchemy.orm import Session
from app.database import get_db

from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse

from app.services.model_service import get_form_options

from fastapi import APIRouter, Depends, HTTPException

from app import schemas, models
from app.auth import get_current_user
from app.services.model_service import (
    MODEL_STATE,
    predict_single,
    predict_batch_dataframe,
    get_model_info
)

router = APIRouter()

@router.get("/form-options")
def form_options(
    current_user: models.User = Depends(get_current_user)
):
    return get_form_options()

@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in MODEL_STATE
    }


@router.get("/model-info")
def model_info(
    current_user: models.User = Depends(get_current_user)
):
    # Only admins can view model information
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admins only"
        )

    m = get_model_info()

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
            sorted(
                m["feature_importances"].items(),
                key=lambda x: -x[1]
            )[:10]
        ),
    }


@router.post(
    "/predict",
    response_model=schemas.PredictionResponse
)
def predict(
    tx: schemas.TransactionRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    try:

        result = predict_single(
            tx.model_dump()
        )

        prediction = models.Prediction(

            user_id=current_user.id,

            trans_date_trans_time=datetime.strptime(
                tx.trans_date_trans_time,
                "%Y-%m-%d %H:%M:%S"
            ),

            amount=tx.amt,

            gender=tx.gender,

            city_pop=tx.city_pop,

            merchant=tx.merchant,

            job=tx.job,

            state=tx.state,

            category=tx.category,

            dob=datetime.strptime(
                tx.dob,
                "%Y-%m-%d"
            ),

            fraud_probability=result["fraud_probability"],

            risk_level=result["risk_level"],

            prediction=result["is_fraud_prediction"]

        )

        db.add(prediction)
        db.commit()

        return result

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@router.post("/predict-batch")
async def predict_batch(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user)
):
    # Only admins can use batch prediction
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admins only"
        )

    try:

        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="File must be a CSV"
            )

        contents = await file.read()

        df = pd.read_csv(
            io.BytesIO(contents)
        )

        result = predict_batch_dataframe(df)

        stream = io.StringIO()

        result.to_csv(
            stream,
            index=False
        )

        stream.seek(0)

        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                "attachment; filename=predictions.csv"
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    

@router.get(
    "/history",
    response_model=list[schemas.PredictionHistory]
)
def prediction_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):

    if current_user.role == "admin":

        predictions = db.query(models.Prediction)\
            .order_by(models.Prediction.created_at.desc())\
            .all()

    else:

        predictions = db.query(models.Prediction)\
            .filter(models.Prediction.user_id == current_user.id)\
            .order_by(models.Prediction.created_at.desc())\
            .all()

    return predictions


@router.get("/dashboard")
def dashboard(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):

    if current_user.role == "admin":

        total_users = db.query(models.User).count()

        total_predictions = db.query(
            models.Prediction
        ).count()

        fraud_predictions = db.query(
            models.Prediction
        ).filter(
            models.Prediction.prediction == 1
        ).count()

        fraud_rate = (
            (fraud_predictions / total_predictions) * 100
            if total_predictions > 0 else 0
        )

        recent_predictions = (
            db.query(models.Prediction)
            .order_by(models.Prediction.created_at.desc())
            .limit(10)
            .all()
        )

        return {
            "total_users": total_users,
            "total_predictions": total_predictions,
            "fraud_predictions": fraud_predictions,
            "fraud_rate": round(fraud_rate, 2),
            "recent_predictions": recent_predictions
        }

    total_predictions = (
        db.query(models.Prediction)
        .filter(models.Prediction.user_id == current_user.id)
        .count()
    )

    fraud_predictions = (
        db.query(models.Prediction)
        .filter(
            models.Prediction.user_id == current_user.id,
            models.Prediction.prediction == 1
        )
        .count()
    )

    safe_predictions = total_predictions - fraud_predictions

    recent_predictions = (
        db.query(models.Prediction)
        .filter(models.Prediction.user_id == current_user.id)
        .order_by(models.Prediction.created_at.desc())
        .limit(10)
        .all()
    )

    return {
        "total_predictions": total_predictions,
        "fraud_predictions": fraud_predictions,
        "safe_predictions": safe_predictions,
        "recent_predictions": recent_predictions
    }