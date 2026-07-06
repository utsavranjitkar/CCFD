import streamlit as st
import pandas as pd
import plotly.express as px

from api import get_model_info


def show():

    st.header("Model Information")

    response = get_model_info(
        st.session_state.token
    )

    if response.status_code != 200:

        st.error(
            response.json()["detail"]
        )
        return

    data = response.json()

    # ---------------------------------
    # Model Performance
    # ---------------------------------

    st.subheader("Model Performance")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Precision",
        f"{data['precision']:.3f}"
    )

    c2.metric(
        "Recall",
        f"{data['recall']:.3f}"
    )

    c3.metric(
        "F1 Score",
        f"{data['f1_score']:.3f}"
    )

    c4.metric(
        "PR AUC",
        f"{data['pr_auc']:.3f}"
    )

    c5.metric(
        "Threshold",
        data["threshold"]
    )

    st.divider()

    # ---------------------------------
    # Dataset Information
    # ---------------------------------

    st.subheader("Dataset Information")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Training Rows",
        data["train_rows"]
    )

    c2.metric(
        "Testing Rows",
        data["test_rows"]
    )

    c3.metric(
        "Test Fraud Rate",
        f"{data['test_fraud_rate']:.2%}"
    )

    st.divider()

    # ---------------------------------
    # Feature Importance
    # ---------------------------------

    st.subheader("Top Feature Importance")

    features = pd.DataFrame(
        {
            "Feature": list(data["top_features"].keys()),
            "Importance": list(data["top_features"].values())
        }
    )

    fig = px.bar(
        features.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Most Important Features"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )