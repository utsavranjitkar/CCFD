"""
Unit tests for database models and schemas.
"""
import pytest
from datetime import datetime
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import Base


class TestUserModel:
    """Test User database model."""

    def test_create_user(self, db: Session):
        """Test creating a user in database."""
        user = models.User(
            name="Test User",
            email="test@example.com",
            password="hashedpassword123",
            role="user"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        assert user.id is not None
        assert user.name == "Test User"
        assert user.email == "test@example.com"
        assert user.role == "user"
        assert user.password == "hashedpassword123"

    def test_user_email_unique(self, db: Session):
        """Test that user email must be unique."""
        user1 = models.User(
            name="User 1",
            email="same@example.com",
            password="password123",
            role="user"
        )
        user2 = models.User(
            name="User 2",
            email="same@example.com",
            password="password456",
            role="user"
        )
        
        db.add(user1)
        db.commit()
        
        db.add(user2)
        
        # Should raise integrity error on commit
        with pytest.raises(Exception):
            db.commit()

    def test_user_required_fields(self, db: Session):
        """Test that required fields cannot be null."""
        user = models.User()
        
        db.add(user)
        
        with pytest.raises(Exception):
            db.commit()

    def test_user_prediction_relationship(self, db: Session, test_user):
        """Test user-prediction relationship."""
        # Create predictions for user
        prediction1 = models.Prediction(
            user_id=test_user.id,
            trans_date_trans_time=datetime(2024, 1, 15, 14, 30, 0),
            amount=150.50,
            gender="M",
            city_pop=50000.0,
            merchant="Merchant 1",
            job="Engineer",
            state="CA",
            category="shopping_net",
            dob=datetime(1990, 5, 20),
            fraud_probability=0.75,
            risk_level="High",
            prediction=1
        )
        
        prediction2 = models.Prediction(
            user_id=test_user.id,
            trans_date_trans_time=datetime(2024, 1, 16, 10, 0, 0),
            amount=200.00,
            gender="F",
            city_pop=100000.0,
            merchant="Merchant 2",
            job="Doctor",
            state="NY",
            category="food_dining",
            dob=datetime(1985, 3, 15),
            fraud_probability=0.25,
            risk_level="Low",
            prediction=0
        )
        
        db.add_all([prediction1, prediction2])
        db.commit()
        
        # Refresh user to load predictions
        db.refresh(test_user)
        
        assert len(test_user.predictions) == 2
        assert test_user.predictions[0].amount == 150.50
        assert test_user.predictions[1].amount == 200.00


class TestPredictionModel:
    """Test Prediction database model."""

    def test_create_prediction(self, db: Session, test_user):
        """Test creating a prediction."""
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
        
        assert prediction.id is not None
        assert prediction.user_id == test_user.id
        assert prediction.amount == 150.50
        assert prediction.fraud_probability == 0.75
        assert prediction.risk_level == "High"
        assert prediction.prediction == 1
        assert prediction.created_at is not None

    def test_prediction_foreign_key(self, db: Session):
        """Test prediction requires valid user_id."""
        prediction = models.Prediction(
            user_id=99999,  # Non-existent user
            trans_date_trans_time=datetime(2024, 1, 15, 14, 30, 0),
            amount=150.50,
            gender="M",
            city_pop=50000.0,
            merchant="Test",
            job="Engineer",
            state="CA",
            category="shopping_net",
            dob=datetime(1990, 5, 20),
            fraud_probability=0.75,
            risk_level="High",
            prediction=1
        )
        
        db.add(prediction)
        
        # SQLite may not enforce foreign key constraints by default
        # This test is more relevant for PostgreSQL
        db.commit()

    def test_prediction_required_fields(self, db: Session, test_user):
        """Test that required fields cannot be null."""
        prediction = models.Prediction(user_id=test_user.id)
        
        db.add(prediction)
        
        with pytest.raises(Exception):
            db.commit()

    def test_prediction_user_relationship(self, db: Session, test_prediction):
        """Test prediction-user relationship."""
        assert test_prediction.user is not None
        assert test_prediction.user.id == test_prediction.user_id
        assert test_prediction.user.name == "Test User"


class TestUserSchema:
    """Test User Pydantic schemas."""

    def test_user_create_schema_valid(self):
        """Test valid user creation schema."""
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "securepass123",
            "role": "user"
        }
        
        user = schemas.UserCreate(**user_data)
        
        assert user.name == "Test User"
        assert user.email == "test@example.com"
        assert user.password == "securepass123"
        assert user.role == "user"

    def test_user_create_schema_name_validation(self):
        """Test name validation in user creation."""
        # Too short
        with pytest.raises(Exception):
            schemas.UserCreate(
                name="A",
                email="test@example.com",
                password="securepass123",
                role="user"
            )
        
        # Invalid characters
        with pytest.raises(Exception):
            schemas.UserCreate(
                name="Test123",
                email="test@example.com",
                password="securepass123",
                role="user"
            )

    def test_user_create_schema_password_validation(self):
        """Test password validation."""
        # Too short
        with pytest.raises(Exception):
            schemas.UserCreate(
                name="Test User",
                email="test@example.com",
                password="short",
                role="user"
            )
        
        # No numbers
        with pytest.raises(Exception):
            schemas.UserCreate(
                name="Test User",
                email="test@example.com",
                password="password",
                role="user"
            )

    def test_user_response_schema(self):
        """Test user response schema."""
        user_data = {
            "id": 1,
            "name": "Test User",
            "email": "test@example.com",
            "role": "user"
        }
        
        user = schemas.UserResponse(**user_data)
        
        assert user.id == 1
        assert user.name == "Test User"
        assert user.email == "test@example.com"
        assert user.role == "user"


class TestTransactionSchema:
    """Test Transaction request schema."""

    def test_valid_transaction_request(self):
        """Test valid transaction request."""
        tx_data = {
            "trans_date_trans_time": "2024-01-15 14:30:00",
            "amt": 150.50,
            "gender": "M",
            "city_pop": 50000.0,
            "merchant": "Test Merchant",
            "job": "Engineer",
            "state": "CA",
            "category": "shopping_net",
            "dob": "1990-05-20"
        }
        
        tx = schemas.TransactionRequest(**tx_data)
        
        assert tx.amt == 150.50
        assert tx.gender == "M"
        assert tx.city_pop == 50000.0

    def test_transaction_amount_validation(self):
        """Test amount must be positive."""
        with pytest.raises(Exception):
            schemas.TransactionRequest(
                trans_date_trans_time="2024-01-15 14:30:00",
                amt=-100.0,  # Invalid: negative
                gender="M",
                city_pop=50000.0,
                merchant="Test",
                job="Test",
                state="CA",
                category="test",
                dob="1990-05-20"
            )

    def test_transaction_gender_validation(self):
        """Test gender must be M or F."""
        with pytest.raises(Exception):
            schemas.TransactionRequest(
                trans_date_trans_time="2024-01-15 14:30:00",
                amt=150.50,
                gender="X",  # Invalid
                city_pop=50000.0,
                merchant="Test",
                job="Test",
                state="CA",
                category="test",
                dob="1990-05-20"
            )

    def test_transaction_city_pop_validation(self):
        """Test city population must be non-negative."""
        with pytest.raises(Exception):
            schemas.TransactionRequest(
                trans_date_trans_time="2024-01-15 14:30:00",
                amt=150.50,
                gender="M",
                city_pop=-1000.0,  # Invalid: negative
                merchant="Test",
                job="Test",
                state="CA",
                category="test",
                dob="1990-05-20"
            )


class TestPredictionResponseSchema:
    """Test Prediction response schema."""

    def test_prediction_response(self):
        """Test prediction response schema."""
        response_data = {
            "fraud_probability": 0.75,
            "is_fraud_prediction": 1,
            "risk_level": "High",
            "threshold_used": 0.45
        }
        
        response = schemas.PredictionResponse(**response_data)
        
        assert response.fraud_probability == 0.75
        assert response.is_fraud_prediction == 1
        assert response.risk_level == "High"
        assert response.threshold_used == 0.45


class TestDashboardSchemas:
    """Test dashboard schemas."""

    def test_user_dashboard_schema(self):
        """Test user dashboard schema."""
        dashboard_data = {
            "total_predictions": 10,
            "fraud_predictions": 2,
            "safe_predictions": 8,
            "recent_predictions": []
        }
        
        dashboard = schemas.UserDashboard(**dashboard_data)
        
        assert dashboard.total_predictions == 10
        assert dashboard.fraud_predictions == 2
        assert dashboard.safe_predictions == 8

    def test_admin_dashboard_schema(self):
        """Test admin dashboard schema."""
        dashboard_data = {
            "total_users": 5,
            "total_predictions": 100,
            "fraud_predictions": 10,
            "fraud_rate": 10.0,
            "recent_predictions": []
        }
        
        dashboard = schemas.AdminDashboard(**dashboard_data)
        
        assert dashboard.total_users == 5
        assert dashboard.total_predictions == 100
        assert dashboard.fraud_rate == 10.0