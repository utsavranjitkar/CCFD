import requests

def get_form_options(token):

    response = requests.get(
        f"{BASE_URL}/prediction/form-options",
        headers={
            "Authorization": f"Bearer {token}"
        }
    )

    return response

BASE_URL = "http://127.0.0.1:8000"


# -------------------------
# Authentication
# -------------------------

def login(email, password):

    response = requests.post(
        f"{BASE_URL}/users/login",
        data={
            "username": email,
            "password": password
        }
    )

    return response


def register(name, email, password, role, admin_code=""):

    response = requests.post(
        f"{BASE_URL}/users/register",
        json={
            "name": name,
            "email": email,
            "password": password,
            "role": role,
            "admin_code": admin_code
        }
    )

    return response


def get_current_user(token):

    headers = {
        "Authorization": f"Bearer {token}"
    }

    return requests.get(
        f"{BASE_URL}/users/me",
        headers=headers
    )


# -------------------------
# Prediction
# -------------------------

def predict(token, data):

    headers = {
        "Authorization": f"Bearer {token}"
    }

    return requests.post(
        f"{BASE_URL}/prediction/predict",
        json=data,
        headers=headers
    )


def get_model_info(token):

    headers = {
        "Authorization": f"Bearer {token}"
    }

    return requests.get(
        f"{BASE_URL}/prediction/model-info",
        headers=headers
    )


def health():

    return requests.get(
        f"{BASE_URL}/prediction/health"
    )

def get_prediction_history(token):

    return requests.get(

        f"{BASE_URL}/prediction/history",

        headers={
            "Authorization": f"Bearer {token}"
        }

    )


def get_dashboard(token):

    return requests.get(

        f"{BASE_URL}/prediction/dashboard",

        headers={
            "Authorization": f"Bearer {token}"
        }

    )

def batch_predict(token, uploaded_file):

    headers = {
        "Authorization": f"Bearer {token}"
    }

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "text/csv"
        )
    }

    return requests.post(
        f"{BASE_URL}/prediction/predict-batch",
        headers=headers,
        files=files
    )