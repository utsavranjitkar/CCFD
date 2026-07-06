"""
Pytest configuration and shared fixtures for testing.
"""
import sys
import os
from pathlib import Path

# Add backend1 directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Set environment variables before importing app modules
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SECRET_KEY"] = "test_secret_key_for_testing_only"
os.environ["ALGORITHM"] = "HS256"
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "60"
os.environ["ADMIN_SECRET"] = "test_admin_secret"

import pytest # type: ignore
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Mock load_artifacts before importing app modules
with patch('app.services.model_service.load_artifacts'):
    from app.database import Base, get_db
    from app.main import app
    from app import models,schemas # pyright: ignore[reportAttributeAccessIssue]
    from app.auth import hash_password, create_access_token

# Use in-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db():
    """
    Create a fresh database for each test.
    """
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db):
    """
    Create a test client with database dependency override.
    """
    def override_get_db():
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db: Session):
    """
    Create a test user in the database.
    """
    user = models.User(
        name="Test User",
        email="test@example.com",
        password=hash_password("testpassword123"),
        role="user"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_admin(db: Session):
    """
    Create a test admin user in the database.
    """
    admin = models.User(
        name="Admin User",
        email="admin@example.com",
        password=hash_password("adminpassword123"),
        role="admin"
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)
    return admin


@pytest.fixture
def user_token(test_user):
    """
    Create a JWT token for test user.
    """
    return create_access_token(data={"sub": test_user.email})


@pytest.fixture
def admin_token(test_admin):
    """
    Create a JWT token for test admin.
    """
    return create_access_token(data={"sub": test_admin.email})


@pytest.fixture
def test_prediction(db: Session, test_user):
    """
    Create a test prediction in the database.
    """
    from datetime import datetime
    
    prediction = models.Prediction(
        user_id=test_user.id,
        trans_date_trans_time=datetime(2024, 1, 15, 14, 30, 0),
        amount=150.50,
        gender="M",
        city_pop=50000.0,
        merchant="Test Merchant",
        job="Engineer",
        state="CA",
        category="shopping_net",
        dob=datetime(1990, 5, 20),
        fraud_probability=0.75,
        risk_level="High",
        prediction=1
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction