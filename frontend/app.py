"""
Streamlit frontend for the Credit Card Fraud Detection API.
Run with: streamlit run frontend/app.py
"""
import os
import io
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, date

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# ── User accounts (username → password) ───────────────────────────────────────
USERS = {
    "admin": "admin123",
    "analyst": "fraud2024",
    "demo": "demo123",
}

# ── Session state init ─────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "login_error" not in st.session_state:
    st.session_state.login_error = ""


def do_login(username, password):
    if username in USERS and USERS[username] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.login_error = ""
    else:
        st.session_state.login_error = "Invalid username or password."


def do_logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.login_error = ""


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 💳 Credit Card Fraud Detection")
        st.markdown("#### Please log in to continue")
        st.markdown("---")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("🔐 Login", use_container_width=True)
            if submitted:
                do_login(username, password)
                st.rerun()

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

        st.markdown("---")
        st.caption("Demo accounts: `admin / admin123` · `analyst / fraud2024` · `demo / demo123`")

    st.stop()  # Block everything below until logged in


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP (only reachable after login)
# ══════════════════════════════════════════════════════════════════════════════
def get_model_info():
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=5)
        if r.ok:
            return r.json()
    except requests.exceptions.RequestException:
        return None
    return None


def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.ok and r.json().get("model_loaded", False)
    except requests.exceptions.RequestException:
        return False


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"👤 Logged in as **{st.session_state.username}**")
    if st.button("🚪 Logout", use_container_width=True):
        do_logout()
        st.rerun()

    st.markdown("---")
    st.header("⚙️ System Status")
    healthy = get_health()
    if healthy:
        st.success("API connected, model loaded")
    else:
        st.error(f"Cannot reach API at {API_URL}")

    info = get_model_info() if healthy else None
    if info:
        st.subheader("Model Info")
        st.metric("Precision", f"{info['precision']:.2%}")
        st.metric("Recall",    f"{info['recall']:.2%}")
        st.metric("F1-Score",  f"{info['f1_score']:.2%}")
        st.metric("PR-AUC",   f"{info['pr_auc']:.2%}")
        st.write(f"Decision threshold: **{info['threshold']:.2f}**")
        with st.expander("Confusion Matrix"):
            cm = info["confusion_matrix"]
            cm_df = pd.DataFrame(
                [[cm["true_negatives"], cm["false_positives"]],
                 [cm["false_negatives"], cm["true_positives"]]],
                index=["Actual: Legit", "Actual: Fraud"],
                columns=["Pred: Legit", "Pred: Fraud"],
            )
            st.dataframe(cm_df)
        with st.expander("Top Features"):
            feat_df = pd.DataFrame(
                info["top_features"].items(), columns=["Feature", "Importance"]
            )
            st.bar_chart(feat_df.set_index("Feature"))

# ── Main content ───────────────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection")
st.caption("Random Forest powered real-time and batch fraud scoring")

tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]

# ── Dropdown options ────────────────────────────────────────────────────────────
# Top 10 highest fraud-rate values from fraudTest.csv (min transaction-count
# thresholds applied to avoid picking up low-sample-size flukes).
TOP_FRAUD_MERCHANTS = [
    "fraud_Romaguera, Cruickshank and Greenholt",
    "fraud_Lemke-Gutmann",
    "fraud_Mosciski, Ziemann and Farrell",
    "fraud_Heathcote, Yost and Kertzmann",
    "fraud_Rodriguez, Yost and Jenkins",
    "fraud_Medhurst PLC",
    "fraud_Bashirian Group",
    "fraud_Kris-Weimann",
    "fraud_Heathcote LLC",
    "fraud_Bednar Group",
]

TOP_FRAUD_JOBS = [
    "Horticultural consultant",
    "Accountant, chartered certified",
    "Television camera operator",
    "Designer, television/film set",
    "TEFL teacher",
    "Tour manager",
    "Surveyor, hydrographic",
    "Hydrogeologist",
    "Commissioning editor",
    "Conservator, furniture",
]

TOP_FRAUD_STATES = [
    "AK", "CT", "ID", "HI", "MT", "DC", "IN", "OR", "MS", "VA",
]

with tab1:
    st.subheader("Score a single transaction")
    col1, col2 = st.columns(2)

    with col1:
        amt = st.number_input(
            "Transaction amount ($)",
            min_value=0.0, max_value=50000.0, value=50.0, step=1.0,
        )
        gender = st.selectbox("Cardholder gender", ["M", "F"])
        city_pop = st.number_input("City population", min_value=0, value=50000, step=1000)
        category = st.selectbox("Merchant category", CATEGORIES)
        merchant = st.selectbox("Merchant name", TOP_FRAUD_MERCHANTS)

    with col2:
        job = st.selectbox("Cardholder job", TOP_FRAUD_JOBS)
        state = st.selectbox("State", TOP_FRAUD_STATES)
        trans_date = st.date_input("Transaction date", value=date.today())
        if "trans_time_str" not in st.session_state:
            st.session_state.trans_time_str = datetime.now().strftime("%H:%M:%S")
        trans_time_str = st.text_input(
            "Transaction time (HH:MM:SS, 24-hour)",
            key="trans_time_str",
        )
        dob = st.date_input(
            "Cardholder date of birth", value=date(1980, 1, 1),
            min_value=date(1920, 1, 1), max_value=date.today(),
        )

    if st.button("🔍 Predict Fraud Risk", type="primary"):
        payload = {
            "trans_date_trans_time": f"{trans_date} {trans_time_str}",
            "amt": amt,
            "gender": gender,
            "city_pop": city_pop,
            "merchant": merchant,
            "job": job,
            "state": state.upper(),
            "category": category,
            "dob": str(dob),
        }
        try:
            with st.spinner("Scoring transaction..."):
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if r.ok:
                result = r.json()
                prob = result["fraud_probability"]
                risk = result["risk_level"]
                badge = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[risk]

                c1, c2, c3 = st.columns(3)
                c1.metric("Fraud Probability", f"{prob:.2%}")
                c2.metric("Risk Level", f"{badge} {risk}")
                c3.metric(
                    "Prediction",
                    "🚨 FRAUD" if result["is_fraud_prediction"] else "✅ Legitimate",
                )
                st.progress(min(prob, 1.0))
            else:
                st.error(f"API error: {r.json().get('detail', r.text)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

with tab2:
    st.subheader("Batch prediction via CSV upload")
    st.write(
        "Upload a CSV with columns: `trans_date_trans_time, amt, gender, city_pop, "
        "merchant, job, state, category, dob`"
    )
    uploaded = st.file_uploader("Choose CSV file", type="csv")

    if uploaded is not None:
        preview = pd.read_csv(uploaded)
        st.write(f"Loaded {len(preview)} rows")
        st.dataframe(preview.head())

        if st.button("🚀 Run Batch Prediction", type="primary"):
            uploaded.seek(0)
            try:
                with st.spinner(f"Scoring {len(preview)} transactions..."):
                    files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                    r = requests.post(f"{API_URL}/predict-batch", files=files, timeout=120)
                if r.ok:
                    result_df = pd.read_csv(io.StringIO(r.text))
                    n_fraud = result_df["is_fraud_prediction"].sum()
                    st.success(f"Done. Flagged {n_fraud} of {len(result_df)} as fraud.")
                    st.dataframe(result_df)
                    st.download_button(
                        "⬇️ Download predictions CSV",
                        data=r.content,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"API error: {r.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")