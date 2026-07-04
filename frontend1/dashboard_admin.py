import streamlit as st

from frontend1.pages.prediction import show as prediction_page
from api import predict_batch, get_model_info


def show():

    st.header("🛡 Admin Dashboard")

    st.success(
        f"Welcome {st.session_state.user['name']}"
    )

    st.write(
        f"Role: {st.session_state.user['role']}"
    )

    st.divider()

    # -----------------------
    # Single Prediction
    # -----------------------

    prediction_page()

    st.divider()

    # -----------------------
    # Batch Prediction
    # -----------------------

    st.subheader("📄 Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        if st.button("Predict CSV"):

            response = predict_batch(
                st.session_state.token,
                uploaded_file
            )

            if response.status_code == 200:

                st.success("Prediction completed!")

                st.download_button(
                    "Download Results",
                    response.content,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            else:

                st.error(
                    response.json()["detail"]
                )

    st.divider()

    # -----------------------
    # Model Information
    # -----------------------

    st.subheader("📊 Model Information")

    if st.button("Load Model Metrics"):

        response = get_model_info(
            st.session_state.token
        )

        if response.status_code == 200:

            info = response.json()

            st.write("### Model Type")
            st.write(info["model_type"])

            st.write("### Threshold")
            st.write(info["threshold"])

            st.write("### Precision")
            st.write(info["precision"])

            st.write("### Recall")
            st.write(info["recall"])

            st.write("### F1 Score")
            st.write(info["f1_score"])

            st.write("### PR AUC")
            st.write(info["pr_auc"])

            st.write("### Top Features")

            st.json(
                info["top_features"]
            )

        else:

            st.error(
                response.json()["detail"]
            )