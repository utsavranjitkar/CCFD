from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from app.config import ADMIN_SECRET

from app.database import get_db
from app import models, schemas

from app.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user
)

router = APIRouter()


@router.get("/")
def test():
    return {
        "message": "Users router is working!"
    }


@router.post(
    "/register",
    response_model=schemas.UserResponse
)
def register_user(
    user: schemas.UserCreate,
    db: Session = Depends(get_db)
):

    db_user = db.query(models.User).filter(
        models.User.email == user.email
    ).first()

    if db_user:
        raise HTTPException(
        status_code=400,
        detail="Email already registered"
    )

    # Verify admin registration code
    if user.role == "admin":
        if user.admin_code != ADMIN_SECRET:
            raise HTTPException(
                status_code=403,
                detail="Invalid admin registration code."
            )

    new_user = models.User(
        name=user.name,
        email=user.email,
        password=hash_password(user.password),
        role=user.role
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


@router.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # Find user by email (username contains the email)
    db_user = db.query(models.User).filter(
        models.User.email == form_data.username
    ).first()

    if not db_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(
        form_data.password,
        db_user.password
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    access_token = create_access_token(
        data={"sub": db_user.email}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.get("/me")
def get_me(
    current_user: models.User = Depends(get_current_user)
):

    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "role": current_user.role
    }


@router.get(
    "/all",
    response_model=list[schemas.AdminUserResponse]
)
def get_all_users(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):

    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only admins can view users."
        )

    users = db.query(models.User).all()

    return users


@router.delete("/delete/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):

    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only admins can delete users."
        )

    user = db.query(models.User).filter(
        models.User.id == user_id
    ).first()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found."
        )

    # Prevent deleting yourself
    if user.id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="You cannot delete your own account."
        )

    # Prevent deleting other admins
    if user.role == "admin":
        raise HTTPException(
            status_code=400,
            detail="Admin accounts cannot be deleted."
        )

    db.delete(user)
    db.commit()

    return {
        "message": "User deleted successfully."
    }