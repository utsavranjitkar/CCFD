import streamlit as st
from datetime import datetime
from datetime import date

from api import predict, get_form_options       

def show():

    options_response = get_form_options(
    st.session_state.token
    )

    if options_response.status_code != 200:
        st.error("Unable to load form options.")
        return

    options = options_response.json()

    st.header("Prediction")

    st.subheader("Transaction Information")

    col1, col2 = st.columns(2)

    with col1:

        amt = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            format="%.2f"
        )

        gender = st.selectbox(
            "Gender",
            ["M", "F"]
        )

        city_pop = st.number_input(
            "City Population",
            min_value=0
        )

    with col2:

        transaction_date = st.date_input(
            "Transaction Date"
        )

        transaction_time = st.time_input(
            "Transaction Time"
        )

        dob = st.date_input(
            "Date of Birth",
            value=date(1990, 1, 1),       # Default selected date
            min_value=date(1900, 1, 1),   # Earliest allowed DOB
            max_value=date.today()        # Latest allowed DOB
        )

    st.divider()

    st.subheader("Merchant Information")

    col3, col4 = st.columns(2)

    with col3:

        merchant = st.selectbox(
            "Merchant",
            options["merchants"]
        )

        state = st.selectbox(
            "State",
            options["states"]
        )

    with col4:

        category = st.selectbox(
            "Category",
            options["categories"]
        )

        job = st.selectbox(
            "Occupation",
            options["jobs"]
        )

    st.divider()

    if st.button(
        "Predict",
        key="predict_button",
        use_container_width=True
    ):

        trans_datetime = datetime.combine(
            transaction_date,
            transaction_time
        )

        payload = {

            "trans_date_trans_time":
                trans_datetime.strftime("%Y-%m-%d %H:%M:%S"),

            "amt": amt,

            "gender": gender,

            "city_pop": city_pop,

            "merchant": merchant,

            "job": job,

            "state": state,

            "category": category,

            "dob": dob.strftime("%Y-%m-%d")
        }

        with st.spinner("Running prediction..."):
            response = predict(
                st.session_state.token,
                payload
            )

        if response.status_code == 200:

            result = response.json()

            probability = result["fraud_probability"]

            st.divider()

            st.subheader("Prediction Result")

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Fraud Probability",
                    f"{probability:.2%}"
                )

            with c2:

                st.metric(
                    "Risk Level",
                    result["risk_level"]
                )

            with c3:

                prediction = (
                    "Fraud"
                    if result["is_fraud_prediction"]
                    else "Safe"
                )

                st.metric(
                    "Prediction",
                    prediction
                )

            st.progress(probability)

            if result["is_fraud_prediction"]:

                st.error(
                    "This transaction is predicted to be fraudulent."
                )

            else:

                st.success(
                    "This transaction appears to be legitimate."
                )

        else:

            st.error(
                response.json()["detail"]
            )