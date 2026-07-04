# Credit Card Fraud Detection System

A full-stack Machine Learning web application that detects fraudulent credit card transactions using a trained Random Forest model. The system provides secure user authentication, real-time fraud prediction, batch prediction, dashboards, prediction history, and model analytics.

---

## Features

### Authentication
- User Registration
- User Login with JWT Authentication
- Role-Based Access Control (User/Admin)

### Fraud Prediction
- Single Transaction Prediction
- Fraud Probability Score
- Risk Level Classification (Low, Medium, High)

### Batch Prediction
- Upload CSV files
- Predict fraud for multiple transactions
- Download prediction results as CSV

### Dashboard

#### User Dashboard
- Total Predictions
- Fraud Predictions
- Safe Predictions
- Prediction Distribution Chart
- Recent Prediction History

#### Admin Dashboard
- Total Users
- Total Predictions
- Fraud Rate
- Fraud Distribution Charts
- Recent Predictions

### Prediction History
- View previous predictions
- Filter by Date
- Filter by Prediction (Safe/Fraud)

### Model Information (Admin)
- Precision
- Recall
- F1 Score
- PR AUC
- Threshold
- Training/Test Dataset Statistics
- Confusion Matrix
- Feature Importance Visualization

---

# Technology Stack

## Frontend
- Streamlit

## Backend
- FastAPI
- Uvicorn

## Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Joblib

## Database
- PostgreSQL
- SQLAlchemy

## Authentication
- JWT (JSON Web Token)
- Passlib (Password Hashing)

---

# Project Structure

```
CreditCardFraudDetection/

├── backend/
│
│   ├── app/
│   │
│   ├── auth.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── main.py
│   │
│   ├── routers/
│   │     ├── users.py
│   │     └── prediction.py
│   │
│   ├── services/
│   │     └── model_service.py
│   │
│   ├── model_artifacts/
│   │     ├── model.joblib
│   │     ├── scaler.joblib
│   │     ├── feature_cols.json
│   │     ├── freq_maps.joblib
│   │     └── metrics.json
│   │
│   └── requirements.txt
│
├── frontend/
│
│   ├── pages/
│   │     ├── dashboard.py
│   │     ├── prediction.py
│   │     ├── batch_prediction.py
│   │     ├── history.py
│   │     └── model_info.py
│   │
│   ├── login.py
│   ├── register.py
│   ├── api.py
│   ├── app.py
│   └── requirements.txt
│
└── README.md
```

---

# Machine Learning Model

- Algorithm: Random Forest Classifier
- Feature Scaling: StandardScaler
- Frequency Encoding for categorical variables
- Probability Threshold for Fraud Detection
- Model saved using Joblib

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>

cd CreditCardFraudDetection
```

---

## 2. Backend Setup

Navigate to the backend directory.

```bash
cd backend
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate the virtual environment.

### Windows

```bash
venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Create a `.env` file.

```env
DATABASE_URL=postgresql://username:password@localhost:5432/database_name

SECRET_KEY=your_secret_key

ALGORITHM=HS256

ACCESS_TOKEN_EXPIRE_MINUTES=60

ADMIN_SECRET_CODE=your_admin_code
```

Run the backend.

```bash
uvicorn app.main:app --reload
```

Backend runs on:

```
http://127.0.0.1:8000
```

---

## 3. Frontend Setup

Navigate to the frontend directory.

```bash
cd frontend
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

```bash
venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run Streamlit.

```bash
python -m streamlit run app.py
```

Frontend runs on:

```
http://localhost:8501
```

---

# Application Workflow

1. Register/Login
2. Authenticate using JWT
3. Perform fraud prediction
4. View prediction result
5. Save prediction to PostgreSQL
6. Review prediction history
7. View dashboard statistics
8. Admin users can perform batch prediction and inspect model performance

---

# API Endpoints

## Authentication

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/users/register` | Register a new user |
| POST | `/users/login` | User login |
| GET | `/users/me` | Get current user |

---

## Prediction

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/prediction/health` | Backend health status |
| GET | `/prediction/form-options` | Retrieve form dropdown values |
| POST | `/prediction/predict` | Single transaction prediction |
| POST | `/prediction/predict-batch` | Batch CSV prediction |
| GET | `/prediction/history` | Prediction history |
| GET | `/prediction/dashboard` | Dashboard statistics |
| GET | `/prediction/model-info` | Model information |

---

# Database

## Users Table

Stores:

- User Name
- Email
- Password Hash
- Role

---

## Predictions Table

Stores:

- User ID
- Transaction Date
- Amount
- Merchant
- Category
- State
- Job
- Gender
- Date of Birth
- Fraud Probability
- Risk Level
- Prediction
- Created At

---

# Future Improvements

- Docker Deployment
- Cloud Deployment
- Prediction Export (PDF/Excel)
- Dark Theme
- Email Notifications
- Model Retraining Pipeline
- Real-Time Fraud Monitoring

---

# Author

Developed as a Full-Stack Machine Learning project using FastAPI, Streamlit, PostgreSQL, and Scikit-learn.