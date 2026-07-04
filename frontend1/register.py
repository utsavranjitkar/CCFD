import streamlit as st
from api import register


def show():

    st.header("Create Account")

    name = st.text_input("Full Name")

    email = st.text_input("Email")

    password = st.text_input(
        "Password",
        type="password"
    )

    role = st.selectbox(
        "Register As",
        [
            "user",
            "admin"
        ]
    )

    # Default value for normal users
    admin_code = ""

    # Only ask for the code if registering as admin
    if role == "admin":
        admin_code = st.text_input(
            "Admin Registration Code",
            type="password"
        )

    if st.button(
        "Create Account",
        key="register_submit",
        use_container_width=True
    ):

        if (
            name == ""
            or email == ""
            or password == ""
        ):
            st.warning("Please fill all fields.")

        elif role == "admin" and admin_code == "":
            st.warning("Please enter the admin registration code.")

        else:
            
            with st.spinner("Creating account..."):
                response = register(
                    name,
                    email,
                    password,
                    role,
                    admin_code
                )

            if response.status_code == 200:

                st.success("Account created successfully!")
                st.info("You can now login.")

            else:

                st.error(response.json()["detail"])