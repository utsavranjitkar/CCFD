import streamlit as st
import pandas as pd

from api import get_all_users, delete_user


def show():

    st.header("Admin Panel")

    response = get_all_users(
        st.session_state.token
    )

    if response.status_code != 200:
        st.error(response.json()["detail"])
        return

    users = response.json()

    if not users:
        st.info("No users found.")
        return

    df = pd.DataFrame(users)

    df = df[df["id"] != st.session_state.user["id"]]

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    selected_user = st.selectbox(
        "Select user to delete",
        df["id"]
    )

    if st.button(
        "Delete User",
        use_container_width=True
    ):

        response = delete_user(
            st.session_state.token,
            selected_user
        )

        if response.status_code == 200:
            st.success("User deleted successfully.")
            st.rerun()

        else:
            st.error(
                response.json()["detail"]
            )