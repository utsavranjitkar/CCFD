import streamlit as st
import pandas as pd

from api import get_prediction_history


def show():

    st.header("Prediction History")

    response = get_prediction_history(
        st.session_state.token
    )

    if response.status_code != 200:

        st.error(
            response.json()["detail"]
        )

        return

    df = pd.DataFrame(response.json())

    if df.empty:

        st.info("No predictions found.")
        return

    # -----------------------------
    # Formatting
    # -----------------------------

    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )

    df["fraud_probability"] = (
        df["fraud_probability"] * 100
    ).round(2)

    df["prediction"] = df["prediction"].map(
        {
            0: "Safe",
            1: "Fraud"
        }
    )

    # -----------------------------
    # Filters
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:

        selected_date = st.date_input(
            "Filter by Date",
            value=None
        )

    with col2:

        prediction_filter = st.selectbox(
            "Prediction",
            [
                "All",
                "Safe",
                "Fraud"
            ]
        )

    # Filter by date
    if selected_date:

        df = df[
            df["trans_date_trans_time"].dt.date == selected_date
        ]

    # Filter by prediction
    if prediction_filter != "All":

        df = df[
            df["prediction"] == prediction_filter
        ]


    # -----------------------------
    # Display
    # -----------------------------

    display_df = df[
        [
            "trans_date_trans_time",
            "merchant",
            "amount",
            "fraud_probability",
            "risk_level",
            "prediction"
        ]
    ].copy()

    display_df.columns = [
        "Transaction Time",
        "Merchant",
        "Amount",
        "Fraud Probability (%)",
        "Risk Level",
        "Prediction"
    ]

    display_df["Transaction Time"] = (
        display_df["Transaction Time"]
        .dt.strftime("%d %b %Y %H:%M")
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )