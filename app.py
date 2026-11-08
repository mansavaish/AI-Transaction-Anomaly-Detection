
import os
import json
import sqlite3
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from datetime import datetime

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Transaction Guard",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
[data-testid="stMetric"] {
    background-color: #172033;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263449;
}
h1, h2, h3 {
    color: #f8fafc;
}
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 9px 18px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# FIND PROJECT FILES
# Works with files in the repository root OR folders
# --------------------------------------------------

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

SEARCH_PATHS = [
    BASE_PATH,
    os.path.join(BASE_PATH, "models"),
    os.path.join(BASE_PATH, "database"),
    os.path.join(BASE_PATH, "app"),
    os.path.dirname(BASE_PATH),
    os.path.join(os.path.dirname(BASE_PATH), "models"),
    os.path.join(os.path.dirname(BASE_PATH), "database")
]


def find_file(filename):
    for folder in SEARCH_PATHS:
        candidate = os.path.join(folder, filename)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Required file not found: {filename}"
    )


RF_PATH = find_file("random_forest_final.pkl")
ISO_PATH = find_file("isolation_forest_final.pkl")
SCALER_PATH = find_file("scaler.pkl")
FEATURE_PATH = find_file("feature_names.pkl")
THRESHOLD_PATH = find_file("thresholds.json")

# Use existing database if present; otherwise create it
try:
    DATABASE_PATH = find_file("transactions.db")
except FileNotFoundError:
    DATABASE_PATH = os.path.join(BASE_PATH, "transactions.db")


# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    rf_model = joblib.load(RF_PATH)
    iso_model = joblib.load(ISO_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    with open(THRESHOLD_PATH, "r") as f:
        thresholds = json.load(f)

    return (
        rf_model,
        iso_model,
        scaler,
        feature_names,
        thresholds
    )


try:
    (
        rf_model,
        iso_model,
        scaler,
        feature_names,
        thresholds
    ) = load_models()

except Exception as error:
    st.error("Unable to load the trained machine-learning models.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def create_table():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_time REAL,
            amount REAL,
            prediction TEXT,
            fraud_probability REAL,
            risk_score REAL,
            risk_level TEXT,
            anomaly_score REAL,
            anomaly_detected INTEGER,
            features_json TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_transaction(transaction, result):
    conn = get_connection()

    conn.execute("""
        INSERT INTO transactions (
            transaction_time,
            amount,
            prediction,
            fraud_probability,
            risk_score,
            risk_level,
            anomaly_score,
            anomaly_detected,
            features_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        float(transaction["Time"]),
        float(transaction["Amount"]),
        result["transaction_status"],
        result["fraud_probability"],
        result["fraud_probability"],
        result["risk_level"],
        result["anomaly_score"],
        int(result["anomaly_detected"]),
        json.dumps({
            key: float(value)
            for key, value in transaction.items()
        }),
        datetime.now().isoformat(timespec="seconds")
    ))

    conn.commit()
    conn.close()


def get_history():
    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM transactions ORDER BY id DESC",
        conn
    )

    conn.close()
    return df


create_table()


# --------------------------------------------------
# PREDICTION ENGINE
# --------------------------------------------------

def predict_transaction(transaction):
    transaction_df = pd.DataFrame(
        [transaction],
        columns=feature_names
    )

    transaction_df[["Time", "Amount"]] = scaler.transform(
        transaction_df[["Time", "Amount"]]
    )

    probability = float(
        rf_model.predict_proba(transaction_df)[0][1]
    )

    rf_threshold = thresholds["random_forest_threshold"]

    prediction = int(probability >= rf_threshold)

    anomaly_score = float(
        -iso_model.decision_function(transaction_df)[0]
    )

    iso_threshold = thresholds["isolation_forest_threshold"]

    anomaly_detected = bool(
        anomaly_score >= iso_threshold
    )

    risk_score = round(probability * 100, 2)

    if risk_score >= 75:
        risk_level = "High"
    elif risk_score >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "transaction_status": (
            "Suspicious" if prediction == 1 else "Legitimate"
        ),
        "fraud_probability": risk_score,
        "risk_level": risk_level,
        "anomaly_score": round(anomaly_score, 6),
        "anomaly_detected": anomaly_detected
    }


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🛡️ Transaction Guard")
st.sidebar.caption("AI-Based Transaction Anomaly Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Analyse Transactions",
        "Transaction History",
        "Fraud Alerts",
        "Reports"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Academic project prototype")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.title("🛡️ AI Transaction Guard")
    st.write(
        "Machine-learning-based online transaction "
        "fraud and anomaly detection dashboard."
    )

    history = get_history()

    total = len(history)

    suspicious = (
        int((history["prediction"] == "Suspicious").sum())
        if total else 0
    )

    legitimate = total - suspicious

    anomalies = (
        int(history["anomaly_detected"].sum())
        if total else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", total)
    col2.metric("Suspicious Transactions", suspicious)
    col3.metric("Legitimate Transactions", legitimate)
    col4.metric("Anomalies Detected", anomalies)

    st.divider()

    if total:
        left, right = st.columns(2)

        with left:
            st.subheader("Transaction Classification")

            chart_data = pd.DataFrame({
                "Category": ["Legitimate", "Suspicious"],
                "Count": [legitimate, suspicious]
            })

            fig = px.pie(
                chart_data,
                names="Category",
                values="Count",
                hole=0.55
            )

            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Risk Level Distribution")

            risk_data = (
                history["risk_level"]
                .value_counts()
                .reset_index()
            )

            risk_data.columns = ["Risk Level", "Count"]

            fig = px.bar(
                risk_data,
                x="Risk Level",
                y="Count",
                color="Risk Level"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Transactions")

        st.dataframe(
            history.head(10),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No transactions analysed yet. "
            "Open Analyse Transactions to upload a compatible CSV."
        )


# --------------------------------------------------
# ANALYSE TRANSACTIONS
# --------------------------------------------------

elif page == "Analyse Transactions":

    st.title("Analyse Transactions")

    st.write(
        "Upload a CSV containing the 30 model input features "
        "to analyse transactions."
    )

    st.warning(
        "This academic prototype uses the Credit Card Fraud "
        "dataset's anonymized PCA features V1–V28. "
        "It cannot directly analyse ordinary bank transaction "
        "exports without a compatible feature transformation."
    )

    required_columns = list(feature_names)

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            input_df = pd.read_csv(uploaded_file)

            # Class is an optional actual label, never a model input
            missing_columns = [
                col for col in required_columns
                if col not in input_df.columns
            ]

            if missing_columns:
                st.error(
                    "Your CSV is missing required model columns: "
                    + ", ".join(missing_columns)
                )

            else:
                transaction_df = input_df[required_columns].copy()

                transaction_df = transaction_df.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if transaction_df.isnull().any().any():
                    st.error(
                        "The model input columns contain missing "
                        "or non-numeric values. Please correct the CSV."
                    )

                elif len(transaction_df) == 0:
                    st.error("The uploaded CSV contains no transactions.")

                elif len(transaction_df) > 10000:
                    st.error(
                        "Please upload at most 10,000 transactions at a time."
                    )

                else:
                    st.success(
                        f"{len(transaction_df):,} compatible transactions loaded."
                    )

                    st.dataframe(
                        transaction_df.head(10),
                        use_container_width=True
                    )

                    if st.button(
                        "Analyse Transactions",
                        type="primary"
                    ):

                        results = []

                        progress = st.progress(0)

                        for index, row in transaction_df.iterrows():

                            transaction = {
                                col: float(row[col])
                                for col in required_columns
                            }

                            result = predict_transaction(transaction)

                            save_transaction(
                                transaction,
                                result
                            )

                            results.append({
                                **transaction,
                                **result
                            })

                            progress.progress(
                                (index + 1) / len(transaction_df)
                            )

                        result_df = pd.DataFrame(results)

                        st.session_state["latest_results"] = result_df

                        st.success(
                            "Transaction analysis completed!"
                        )

        except Exception as error:
            st.error("Unable to analyse this CSV file.")
            st.exception(error)

    if "latest_results" in st.session_state:

        result_df = st.session_state["latest_results"]

        st.divider()
        st.subheader("Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Analysed", len(result_df))

        col2.metric(
            "Suspicious",
            int(
                (result_df["transaction_status"] == "Suspicious").sum()
            )
        )

        col3.metric(
            "Anomalies",
            int(result_df["anomaly_detected"].sum())
        )

        display_columns = [
            "transaction_status",
            "fraud_probability",
            "risk_level",
            "anomaly_score",
            "anomaly_detected"
        ]

        st.dataframe(
            result_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Analysis Results",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="transaction_analysis_results.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# TRANSACTION HISTORY
# --------------------------------------------------

elif page == "Transaction History":

    st.title("Transaction History")

    history = get_history()

    if history.empty:
        st.info("No transaction history is available yet.")

    else:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Transaction History",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="transaction_history.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# FRAUD ALERTS
# --------------------------------------------------

elif page == "Fraud Alerts":

    st.title("🚨 Fraud Alerts")

    history = get_history()

    if history.empty:
        st.info("No alerts available yet.")

    else:
        suspicious = history[
            (history["prediction"] == "Suspicious")
            | (history["anomaly_detected"] == 1)
        ]

        if suspicious.empty:
            st.success("No suspicious transactions found in saved history.")

        else:
            st.warning(
                f"{len(suspicious)} suspicious or anomalous "
                "transactions found."
            )

            st.dataframe(
                suspicious,
                use_container_width=True,
                hide_index=True
            )


# --------------------------------------------------
# REPORTS
# --------------------------------------------------

elif page == "Reports":

    st.title("Transaction Reports")

    history = get_history()

    if history.empty:
        st.info("No data available to generate reports.")

    else:
        col1, col2 = st.columns(2)

        col1.metric(
            "Total Transactions",
            len(history)
        )

        col2.metric(
            "Average Fraud Probability",
            f"{history['fraud_probability'].mean():.2f}%"
        )

        st.subheader("Risk Level Summary")

        st.dataframe(
            history["risk_level"].value_counts().rename_axis(
                "Risk Level"
            ).reset_index(name="Transactions"),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Full Report",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="transaction_report.csv",
            mime="text/csv"
        )

st.sidebar.divider()
st.sidebar.caption("AI Transaction Guard | Project Prototype")
