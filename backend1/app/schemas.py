from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import List
from datetime import datetime


class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: str
    admin_code: str | None = None

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str

    model_config = ConfigDict(from_attributes=True)



class TransactionRequest(BaseModel):
    trans_date_trans_time: str = Field(
        ...,
        description="YYYY-MM-DD HH:MM:SS"
    )

    amt: float = Field(..., gt=0)

    gender: str = Field(..., pattern="^[MF]$")

    city_pop: float = Field(..., ge=0)

    merchant: str

    job: str

    state: str

    category: str

    dob: str = Field(
        ...,
        description="YYYY-MM-DD"
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud_prediction: int
    risk_level: str
    threshold_used: float


class PredictionHistory(BaseModel):

    id: int

    user_id: int

    trans_date_trans_time: datetime

    amount: float

    gender: str

    city_pop: float

    merchant: str

    job: str

    state: str

    category: str

    dob: datetime

    fraud_probability: float

    risk_level: str

    prediction: int

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )

class RecentPrediction(BaseModel):

    trans_date_trans_time: datetime
    amount: float
    merchant: str
    fraud_probability: float
    risk_level: str
    prediction: int

    model_config = ConfigDict(from_attributes=True)


class UserDashboard(BaseModel):

    total_predictions: int
    fraud_predictions: int
    safe_predictions: int
    recent_predictions: List[RecentPrediction]


class AdminDashboard(BaseModel):

    total_users: int
    total_predictions: int
    fraud_predictions: int
    fraud_rate: float
    recent_predictions: List[RecentPrediction]