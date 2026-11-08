
import pandas as pd
import numpy as np

def preprocess_transaction(transaction, feature_names, scaler):

    transaction_df = pd.DataFrame(
        [transaction],
        columns=feature_names
    )

    transaction_df[["Time", "Amount"]] = scaler.transform(
        transaction_df[["Time", "Amount"]]
    )

    return transaction_df


def predict_transaction(
    transaction,
    rf_model,
    iso_model,
    scaler,
    feature_names,
    thresholds
):

    processed_data = preprocess_transaction(
        transaction,
        feature_names,
        scaler
    )

    probability = float(
        rf_model.predict_proba(processed_data)[0][1]
    )

    rf_threshold = thresholds["random_forest_threshold"]

    prediction = int(probability >= rf_threshold)

    anomaly_score = float(
        -iso_model.decision_function(processed_data)[0]
    )

    iso_threshold = thresholds["isolation_forest_threshold"]

    anomaly_detected = anomaly_score >= iso_threshold

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
        "anomaly_detected": bool(anomaly_detected)
    }
