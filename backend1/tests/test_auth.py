"""
Unit tests for authentication functionality.
"""
import pytest
from datetime import datetime, timedelta
from jose import jwt

from app.auth import hash_password, verify_password, create_access_token, get_current_user
from app.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from app import models, schemas
from fastapi import HTTPException


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_hash_password(self):
        """Test that password is hashed correctly."""
        password = "testpassword123"
        hashed = hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 20
        assert hashed.startswith("$2b$")

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "testpassword123"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        hashed = hash_password(password)
        
        assert verify_password(wrong_password, hashed) is False

    def test_different_passwords_different_hashes(self):
        """Test that different passwords produce different hashes."""
        hash1 = hash_password("password1")
        hash2 = hash_password("password2")
        
        assert hash1 != hash2


class TestJWTTokens:
    """Test JWT token creation and validation."""

    def test_create_access_token(self):
        """Test JWT token creation."""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_with_expiry(self):
        """Test that token contains expiration claim."""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in payload
        assert "sub" in payload
        assert payload["sub"] == "test@example.com"

    def test_token_expiry_time(self):
        """Test that token expires after configured minutes."""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp_time = datetime.utcfromtimestamp(payload["exp"])
        expected_exp = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        # Allow 1 second difference for test execution time
        assert abs((exp_time - expected_exp).total_seconds()) < 1

    def test_decode_valid_token(self):
        """Test decoding a valid token."""
        data = {"sub": "test@example.com"}
        token = create_access_token(data)
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "test@example.com"


class TestGetCurrentUser:
    """Test get_current_user dependency."""

    def test_get_current_user_valid_token(self, client, test_user):
        """Test getting current user with valid token."""
        from app.auth import create_access_token
        
        token = create_access_token(data={"sub": test_user.email})
        
        response = client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["name"] == test_user.name
        assert data["role"] == test_user.role

    def test_get_current_user_invalid_token(self, client):
        """Test getting current user with invalid token."""
        response = client.get(
            "/users/me",
            headers={"Authorization": "Bearer invalidtoken123"}
        )
        
        assert response.status_code == 401
        assert "Could not validate credentials" in response.json()["detail"]

    def test_get_current_user_missing_token(self, client):
        """Test getting current user without token."""
        response = client.get("/users/me")
        
        assert response.status_code == 401

    def test_get_current_user_nonexistent_user(self, client, db):
        """Test getting current user with token for non-existent user."""
        from app.auth import create_access_token
        
        # Create token for email that doesn't exist in database
        token = create_access_token(data={"sub": "nonexistent@example.com"})
        
        response = client.get(
            "/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 401


class TestUserRegistration:
    """Test user registration endpoint."""

    def test_register_user_success(self, client, db):
        """Test successful user registration."""
        response = client.post(
            "/users/register",
            json={
                "name": "New User",
                "email": "newuser@example.com",
                "password": "securepass123",
                "role": "user"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New User"
        assert data["email"] == "newuser@example.com"
        assert data["role"] == "user"
        assert "id" in data

    def test_register_duplicate_email(self, client, test_user):
        """Test registration with duplicate email."""
        response = client.post(
            "/users/register",
            json={
                "name": "Another User",
                "email": test_user.email,
                "password": "securepass123",
                "role": "user"
            }
        )
        
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]

    def test_register_admin_with_invalid_code(self, client):
        """Test admin registration with invalid code."""
        response = client.post(
            "/users/register",
            json={
                "name": "Admin User",
                "email": "admin@example.com",
                "password": "securepass123",
                "role": "admin",
                "admin_code": "wrongcode"
            }
        )
        
        assert response.status_code == 403
        assert "Invalid admin registration code" in response.json()["detail"]

    def test_register_user_with_short_name(self, client):
        """Test registration with name too short."""
        response = client.post(
            "/users/register",
            json={
                "name": "A",
                "email": "test@example.com",
                "password": "securepass123",
                "role": "user"
            }
        )
        
        assert response.status_code == 422  # Validation error

    def test_register_user_with_invalid_name(self, client):
        """Test registration with invalid characters in name."""
        response = client.post(
            "/users/register",
            json={
                "name": "Test123",
                "email": "test@example.com",
                "password": "securepass123",
                "role": "user"
            }
        )
        
        assert response.status_code == 422

    def test_register_user_with_weak_password(self, client):
        """Test registration with weak password."""
        response = client.post(
            "/users/register",
            json={
                "name": "Test User",
                "email": "test@example.com",
                "password": "short",
                "role": "user"
            }
        )
        
        assert response.status_code == 422


class TestUserLogin:
    """Test user login endpoint."""

    def test_login_success(self, client, test_user):
        """Test successful login."""
        response = client.post(
            "/users/login",
            data={
                "username": test_user.email,
                "password": "testpassword123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client, test_user):
        """Test login with wrong password."""
        response = client.post(
            "/users/login",
            data={
                "username": test_user.email,
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]

    def test_login_nonexistent_user(self, client):
        """Test login with non-existent email."""
        response = client.post(
            "/users/login",
            data={
                "username": "nonexistent@example.com",
                "password": "password123"
            }
        )
        
        assert response.status_code == 401

    def test_login_empty_credentials(self, client):
        """Test login with empty credentials."""
        response = client.post(
            "/users/login",
            data={
                "username": "",
                "password": ""
            }
        )
        
        # Empty credentials should return 422 (validation error) or 401
        assert response.status_code in [401, 422]


class TestUserEndpoints:
    """Test user management endpoints."""

    def test_get_all_users_as_admin(self, client, test_admin, admin_token):
        """Test admin can get all users."""
        response = client.get(
            "/users/all",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_get_all_users_as_regular_user(self, client, test_user, user_token):
        """Test regular user cannot get all users."""
        response = client.get(
            "/users/all",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        assert response.status_code == 403
        assert "Only admins" in response.json()["detail"]

    def test_delete_user_as_admin(self, client, test_admin, test_user, admin_token):
        """Test admin can delete a user."""
        response = client.delete(
            f"/users/delete/{test_user.id}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]

    def test_delete_user_as_regular_user(self, client, test_user, user_token):
        """Test regular user cannot delete users."""
        response = client.delete(
            f"/users/delete/{test_user.id}",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        assert response.status_code == 403

    def test_delete_self(self, client, test_admin, admin_token):
        """Test admin cannot delete themselves."""
        response = client.delete(
            f"/users/delete/{test_admin.id}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 400
        assert "cannot delete your own account" in response.json()["detail"]

    def test_delete_another_admin(self, client, db, test_admin, admin_token):
        """Test admin cannot delete another admin."""
        # Create another admin
        admin2 = models.User(
            name="Admin 2",
            email="admin2@example.com",
            password=hash_password("password123"),
            role="admin"
        )
        db.add(admin2)
        db.commit()
        db.refresh(admin2)
        
        response = client.delete(
            f"/users/delete/{admin2.id}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 400
        assert "Admin accounts cannot be deleted" in response.json()["detail"]

    def test_delete_nonexistent_user(self, client, admin_token):
        """Test deleting non-existent user."""
        response = client.delete(
            "/users/delete/99999",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 404
        assert "User not found" in response.json()["detail"]