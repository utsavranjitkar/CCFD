import streamlit as st
import pandas as pd
import plotly.express as px

from api import get_dashboard


def show():

    response = get_dashboard(
        st.session_state.token
    )

    if response.status_code != 200:

        st.error(
            response.json()["detail"]
        )
        return

    data = response.json()

    role = st.session_state.user["role"]

    st.header("Dashboard")

    # ==========================================================
    # Metrics
    # ==========================================================

    if role == "admin":

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Users",
            data["total_users"]
        )

        c2.metric(
            "Predictions",
            data["total_predictions"]
        )

        c3.metric(
            "Fraud",
            data["fraud_predictions"]
        )

        c4.metric(
            "Fraud Rate",
            f"{data['fraud_rate']}%"
        )

        safe_predictions = (
            data["total_predictions"] -
            data["fraud_predictions"]
        )

    else:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predictions",
            data["total_predictions"]
        )

        c2.metric(
            "Fraud",
            data["fraud_predictions"]
        )

        c3.metric(
            "Safe",
            data["safe_predictions"]
        )

        safe_predictions = data["safe_predictions"]

    # ==========================================================
    # Charts
    # ==========================================================

    st.divider()

    left, right = st.columns(2)

    pie_data = pd.DataFrame(
        {
            "Type": ["Fraud", "Safe"],
            "Count": [
                data["fraud_predictions"],
                safe_predictions
            ]
        }
    )

    fig = px.pie(
        pie_data,
        values="Count",
        names="Type",
        title="Prediction Distribution",
        hole=0.45
    )

    left.plotly_chart(
        fig,
        use_container_width=True
    )

    recent = pd.DataFrame(
        data["recent_predictions"]
    )

    if not recent.empty:

        risk_counts = (
            recent["risk_level"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Level",
            "Count"
        ]

        fig2 = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            title="Recent Risk Levels"
        )

        right.plotly_chart(
            fig2,
            use_container_width=True
        )

    # ==========================================================
    # Recent Predictions
    # ==========================================================

    st.divider()

    st.subheader("Recent Predictions")

    if len(data["recent_predictions"]) == 0:

        st.info(
            "No predictions available."
        )
        return

    df = pd.DataFrame(
        data["recent_predictions"]
    )

    df = df[
        [
            "trans_date_trans_time",
            "merchant",
            "amount",
            "fraud_probability",
            "risk_level",
            "prediction"
        ]
    ]

    df.columns = [
        "Transaction Time",
        "Merchant",
        "Amount",
        "Fraud Probability",
        "Risk Level",
        "Prediction"
    ]

    df["Fraud Probability"] = (
        df["Fraud Probability"] * 100
    ).round(2).astype(str) + "%"

    df["Prediction"] = df["Prediction"].map(
        {
            0: "Safe",
            1: "Fraud"
        }
    )

    df["Transaction Time"] = pd.to_datetime(
        df["Transaction Time"]
    ).dt.strftime(
        "%d %b %Y %H:%M"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )