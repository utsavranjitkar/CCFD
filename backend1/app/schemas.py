from pydantic import BaseModel, ConfigDict, Field, EmailStr, field_validator
from datetime import datetime
from typing import List
from datetime import datetime
import re

class UserCreate(BaseModel):

    name: str
    email: EmailStr
    password: str
    role: str
    admin_code: str = ""

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):

        if len(value.strip()) < 2:
            raise ValueError(
                "Name must be at least 2 characters."
            )

        if not re.fullmatch(r"[A-Za-z ]+", value):
            raise ValueError(
                "Name can only contain letters and spaces."
            )

        return value.strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, value):

        if len(value) < 8:
            raise ValueError(
                "Password must be at least 8 characters."
            )

        if not any(char.isdigit() for char in value):
            raise ValueError(
                "Password must contain at least one number."
            )

        return value

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


class AdminUserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str

    class Config:
        from_attributes = True