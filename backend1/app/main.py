from fastapi import FastAPI
from app.database import Base, engine
from app.routers import users, prediction
from app.services.model_service import load_artifacts

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Credit Card Fraud Detection")

load_artifacts()

app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(prediction.router, prefix="/prediction", tags=["Prediction"])


@app.get("/")
def home():
    return {"message": "API running"}