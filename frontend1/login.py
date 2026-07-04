import streamlit as st
from api import login, get_current_user


def show():

    st.title("Login")

    email = st.text_input("Email")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        key="login_submit",
        use_container_width=True
    ):

        if email == "" or password == "":
            st.warning("Please enter your email and password.")
            return

        with st.spinner("Signing in..."):
            response = login(email, password)

        if response.status_code != 200:

            try:
                st.error(response.json()["detail"])
            except Exception:
                st.error("Invalid email or password.")

            return

        token = response.json()["access_token"]

        st.session_state.token = token

        user_response = get_current_user(token)

        if user_response.status_code != 200:

            st.error("Unable to fetch user details.")
            return

        st.session_state.user = user_response.json()
        st.session_state.current_page = "Dashboard"

        st.rerun()