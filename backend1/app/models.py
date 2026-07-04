from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from app.database import Base

from sqlalchemy.orm import relationship
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False)

    predictions = relationship(
        "Prediction",
        back_populates="user"
    )

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    ForeignKey
)



class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )

    trans_date_trans_time = Column(
        DateTime,
        nullable=False
    )

    amount = Column(
        Float,
        nullable=False
    )

    gender = Column(
        String,
        nullable=False
    )

    city_pop = Column(
        Float,
        nullable=False
    )

    merchant = Column(
        String,
        nullable=False
    )

    job = Column(
        String,
        nullable=False
    )

    state = Column(
        String,
        nullable=False
    )

    category = Column(
        String,
        nullable=False
    )

    dob = Column(
        DateTime,
        nullable=False
    )

    fraud_probability = Column(
        Float,
        nullable=False
    )

    risk_level = Column(
        String,
        nullable=False
    )

    prediction = Column(
        Integer,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    user = relationship(
        "User",
        back_populates="predictions"
    )