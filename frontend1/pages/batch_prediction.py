import streamlit as st
import pandas as pd

from api import batch_predict


def show():

    st.header("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    if st.button(
        "Run Batch Prediction",
        use_container_width=True
    ):
        with st.spinner("Running batch prediction..."):
            response = batch_predict(
                st.session_state.token,
                uploaded_file
            )

        if response.status_code != 200:

            st.error(
                response.json()["detail"]
            )

            return

        st.success(
            "Prediction completed successfully."
        )

        st.download_button(
            "Download Predictions",
            response.content,
            file_name="predictions.csv",
            mime="text/csv"
        )