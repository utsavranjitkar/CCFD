import streamlit as st

from frontend1.pages.prediction import show as prediction_page


def show():

    st.header("User Dashboard")

    st.success(
        f"Welcome {st.session_state.user['name']}"
    )

    st.write(
        f"Role: {st.session_state.user['role']}"
    )

    st.divider()

    prediction_page()