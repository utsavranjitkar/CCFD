import streamlit as st

from login import show as login
from register import show as register

from pages.dashboard import show as dashboard
from pages.prediction import show as prediction
from pages.batch_prediction import show as batch_prediction
from pages.model_info import show as model_info
from pages.history import show as history

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# -----------------------------
# Session initialization
# -----------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# -----------------------------
# Header
# -----------------------------
st.title("Credit Card Fraud Detection")

st.markdown("---")

# ==========================================================
# BEFORE LOGIN
# ==========================================================
if "user" not in st.session_state:

    col1, col2, col3 = st.columns(3)

    if col1.button(
        "Home",
        key="nav_home",
        use_container_width=True
    ):
        st.session_state.current_page = "Home"

    if col2.button(
        "Login",
        key="nav_login",
        use_container_width=True
    ):
        st.session_state.current_page = "Login"

    if col3.button(
        "Register",
        key="nav_register",
        use_container_width=True
    ):
        st.session_state.current_page = "Register"

    st.markdown("---")

    if st.session_state.current_page == "Home":

        st.header("Welcome")

        st.write(
            """
            This application detects fraudulent credit card transactions
            using a Machine Learning model.

            Features

            • User Authentication

            • Single Transaction Prediction

            • Batch Prediction (Admin)

            • Model Information (Admin)
            """
        )

    elif st.session_state.current_page == "Login":

        login()

    elif st.session_state.current_page == "Register":

        register()

# ==========================================================
# AFTER LOGIN
# ==========================================================
else:

    role = st.session_state.user["role"]

    st.write(
        f"Signed in as **{st.session_state.user['name']}** ({role})"
    )

    if role == "admin":

        col1, col2 = st.columns([8, 1])

        with col1:

            page = st.segmented_control(
                "",
                options=[
                    "Dashboard",
                    "Prediction",
                    "Batch Prediction",
                    "History",
                    "Model Info"
                ],
                default=st.session_state.current_page,
                key="admin_navigation"
            )

        with col2:

            if st.button(
                "Logout",
                use_container_width=True
            ):
                st.session_state.clear()
                st.rerun()

    else:

        col1, col2 = st.columns([6, 1])

        with col1:

            page = st.segmented_control(
                "",
                options=[
                    "Dashboard",
                    "Prediction",
                    "History"
                ],
                default=st.session_state.current_page,
                key="user_navigation"
            )

        with col2:

            if st.button(
                "Logout",
                key="logout_button",
                use_container_width=True
            ):
                st.session_state.clear()
                st.rerun()

    st.session_state.current_page = page

    st.markdown("---")

    if page == "Dashboard":

        dashboard()

    elif page == "Prediction":

        prediction()

    elif page == "Batch Prediction":

        batch_prediction()

    elif page == "Model Info":

        model_info()

    elif page == "History":
        history()