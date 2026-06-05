import json
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("data/raw/ph_motorcycle_loans.csv")
MODEL_PATH = Path("training/artifacts/model.pkl")
PREDICTIONS_LOG_PATH = Path("data/processed/predictions_log.csv")
BACKUP_LOG_PATH = Path("data/processed/predictions_log_backup_before_drift.csv")
DRIFT_ONLY_PATH = Path("data/processed/drift_simulated_predictions.csv")

RANDOM_SEED = 42
N_DRIFTED_APPLICATIONS = 80
DRIFT_ORIGINATION_DATE = date(2026, 6, 4)
DECISION_THRESHOLD = 0.35


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run python training/train.py first."
        )

    return joblib.load(MODEL_PATH)


def load_raw_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")

    return pd.read_csv(RAW_DATA_PATH)


def make_decision(probability, threshold):
    if probability < 0.20:
        return "APPROVED"

    if probability < threshold:
        return "NEEDS_MANUAL_REVIEW"

    return "HIGH_RISK_MANUAL_REVIEW"


def get_risk_factors(row):
    factors = []

    loan_to_income_ratio = row["loan_asset_cost"] / max(row["total_income"], 1)
    down_payment_ratio = row["down_payment_amount"] / max(row["loan_asset_cost"], 1)

    if row["bureau_score"] == -1:
        factors.append("No bureau history")
    elif row["bureau_score"] < 600:
        factors.append("Low bureau score")

    if row["total_income"] < 12000:
        factors.append("Low declared monthly income")

    if row["number_of_dependents"] >= 4:
        factors.append("High number of dependents")

    if row["has_existing_loan"] == "Yes":
        factors.append("Has existing loan obligation")

    if loan_to_income_ratio > 6:
        factors.append("High loan-to-income ratio")

    if row["res_resident_status"] == "Rented":
        factors.append("Rented residence")

    if row["age"] < 22:
        factors.append("Young borrower profile")

    if down_payment_ratio < 0.12:
        factors.append("Low down payment ratio")

    if not factors:
        factors.append("No major rule-based risk factor detected")

    return factors


def get_top_reasons(probability, threshold, risk_factors):
    if probability < 0.20:
        return [
            "Low predicted default probability",
            "Application is within acceptable risk range",
            "Proceed with normal document verification",
        ]

    if probability < threshold:
        return [
            "Moderate predicted default probability",
            "Manual review recommended before approval",
            risk_factors[0],
        ]

    return [
        "High predicted default probability",
        "Manual review required before decision",
        risk_factors[0],
    ]


def create_drifted_applications(raw_df, model_features):
    rng = np.random.default_rng(RANDOM_SEED)

    pending_mask = raw_df["label_is_mature"].astype(str).str.lower() == "false"
    base_df = raw_df[pending_mask].copy()

    if base_df.empty:
        base_df = raw_df.copy()

    sampled_df = base_df.sample(
        n=N_DRIFTED_APPLICATIONS,
        replace=True,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    drifted_df = sampled_df.copy()

    # Simulate a risky economic period:
    # lower income, lower bureau score, higher asset cost, more dependents,
    # more existing loans, more rented residences, more online applications.
    drifted_df["total_income"] = (
        drifted_df["total_income"] * rng.uniform(0.55, 0.80, size=len(drifted_df))
    ).round(2)
    drifted_df["total_income"] = drifted_df["total_income"].clip(lower=5000, upper=200000)

    drifted_df["loan_asset_cost"] = (
        drifted_df["loan_asset_cost"] * rng.uniform(1.10, 1.30, size=len(drifted_df))
    ).round(2)
    drifted_df["loan_asset_cost"] = drifted_df["loan_asset_cost"].clip(lower=30000, upper=200000)

    drifted_df["down_payment_amount"] = (
        drifted_df["down_payment_amount"] * rng.uniform(0.45, 0.75, size=len(drifted_df))
    ).round(2)

    minimum_down_payment = drifted_df["loan_asset_cost"] * 0.05
    drifted_df["down_payment_amount"] = np.maximum(
        drifted_df["down_payment_amount"],
        minimum_down_payment,
    ).round(2)

    drifted_df["bureau_score"] = drifted_df["bureau_score"].apply(
        lambda score: -1 if score == -1 else max(300, int(score - rng.integers(80, 180)))
    )

    drifted_df["number_of_dependents"] = (
        drifted_df["number_of_dependents"] + rng.integers(1, 4, size=len(drifted_df))
    ).clip(upper=10)

    drifted_df["interest_rate"] = (
        drifted_df["interest_rate"] + rng.uniform(1.0, 4.0, size=len(drifted_df))
    ).round(2)
    drifted_df["interest_rate"] = drifted_df["interest_rate"].clip(lower=5, upper=30)

    drifted_df["has_existing_loan"] = rng.choice(
        ["Yes", "No"],
        size=len(drifted_df),
        p=[0.75, 0.25],
    )

    drifted_df["res_resident_status"] = rng.choice(
        ["Rented", "Living with Parents", "Owned"],
        size=len(drifted_df),
        p=[0.60, 0.25, 0.15],
    )

    drifted_df["channel"] = rng.choice(
        ["ONLINE", "DEALER", "DIRECT"],
        size=len(drifted_df),
        p=[0.65, 0.25, 0.10],
    )

    drifted_df["manufacturer"] = rng.choice(
        ["Rusi", "TMX", "Honda", "Yamaha", "Suzuki", "Kawasaki"],
        size=len(drifted_df),
        p=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05],
    )

    drifted_df["loan_to_income_ratio"] = (
        drifted_df["loan_asset_cost"] / drifted_df["total_income"].replace(0, np.nan)
    )

    missing_features = [
        feature for feature in model_features if feature not in drifted_df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing model features in drifted data: {missing_features}")

    return drifted_df


def create_prediction_log_rows(drifted_df, artifact):
    pipeline = artifact["pipeline"]
    model_features = artifact["model_features"]
    threshold = DECISION_THRESHOLD
    model_name = artifact.get("model_name", "unknown_model")

    X_score = drifted_df[model_features].copy()
    probabilities = pipeline.predict_proba(X_score)[:, 1]

    rows = []

    for index, (_, row) in enumerate(drifted_df.iterrows()):
        probability = float(probabilities[index])
        decision = make_decision(probability, threshold)

        risk_factors = get_risk_factors(row)
        top_reasons = get_top_reasons(probability, threshold, risk_factors)

        expected_label_date = DRIFT_ORIGINATION_DATE + timedelta(days=90)
        prediction_timestamp = datetime.now().isoformat(timespec="seconds")

        input_payload = {
            "applicant_name": f"Drift Applicant {index + 1}",
            "bureau_score": int(row["bureau_score"]),
            "down_payment_amount": float(row["down_payment_amount"]),
            "res_years_at_current_city": int(row["res_years_at_current_city"]),
            "res_years_at_current_address": int(row["res_years_at_current_address"]),
            "interest_rate": float(row["interest_rate"]),
            "loan_asset_cost": float(row["loan_asset_cost"]),
            "age": int(row["age"]),
            "number_of_installments": int(row["number_of_installments"]),
            "total_income": float(row["total_income"]),
            "years_in_occupation": int(row["years_in_occupation"]),
            "number_of_dependents": int(row["number_of_dependents"]),
            "gender": str(row["gender"]),
            "channel": str(row["channel"]),
            "marital_status": str(row["marital_status"]),
            "manufacturer": str(row["manufacturer"]),
            "occupation_type": str(row["occupation_type"]),
            "res_resident_status": str(row["res_resident_status"]),
            "has_existing_loan": str(row["has_existing_loan"]),
            "loan_origination_date": DRIFT_ORIGINATION_DATE.isoformat(),
        }

        log_row = {
            "prediction_id": str(uuid4()),
            "prediction_timestamp": prediction_timestamp,
            "applicant_name": f"Drift Applicant {index + 1}",
            "model_version": f"{model_name}_v1",
            "decision": decision,
            "default_probability": round(probability, 4),
            "default_probability_percent": f"{probability * 100:.2f}%",
            "approval_confidence": round(1.0 - probability, 4),
            "top_reasons": " | ".join(top_reasons),
            "risk_factors": " | ".join(risk_factors),
            "label_status": "PENDING, outcome observable after 90+ days",
            "expected_label_date": expected_label_date.isoformat(),
            "label_is_mature_at_prediction": False,
            "actual_TARGET": "",
            "actual_days_past_due": "",
            "actual_label_arrival_date": "",
            "loan_origination_date": DRIFT_ORIGINATION_DATE.isoformat(),
            "bureau_score": int(row["bureau_score"]),
            "down_payment_amount": float(row["down_payment_amount"]),
            "res_years_at_current_city": int(row["res_years_at_current_city"]),
            "res_years_at_current_address": int(row["res_years_at_current_address"]),
            "interest_rate": float(row["interest_rate"]),
            "loan_asset_cost": float(row["loan_asset_cost"]),
            "age": int(row["age"]),
            "number_of_installments": int(row["number_of_installments"]),
            "total_income": float(row["total_income"]),
            "years_in_occupation": int(row["years_in_occupation"]),
            "number_of_dependents": int(row["number_of_dependents"]),
            "loan_to_income_ratio": float(row["loan_to_income_ratio"]),
            "gender": str(row["gender"]),
            "channel": str(row["channel"]),
            "marital_status": str(row["marital_status"]),
            "manufacturer": str(row["manufacturer"]),
            "occupation_type": str(row["occupation_type"]),
            "res_resident_status": str(row["res_resident_status"]),
            "has_existing_loan": str(row["has_existing_loan"]),
            "input_payload_json": json.dumps(input_payload, ensure_ascii=False),
        }

        rows.append(log_row)

    return pd.DataFrame(rows)


def main():
    artifact = load_model_artifact()
    raw_df = load_raw_data()

    model_features = artifact["model_features"]

    drifted_applications = create_drifted_applications(raw_df, model_features)
    drift_predictions = create_prediction_log_rows(drifted_applications, artifact)

    if PREDICTIONS_LOG_PATH.exists():
        existing_log = pd.read_csv(PREDICTIONS_LOG_PATH)
        existing_log.to_csv(BACKUP_LOG_PATH, index=False)
        combined_log = pd.concat([existing_log, drift_predictions], ignore_index=True)
    else:
        existing_log = pd.DataFrame()
        combined_log = drift_predictions

    drift_predictions.to_csv(DRIFT_ONLY_PATH, index=False)
    combined_log.to_csv(PREDICTIONS_LOG_PATH, index=False)

    print("Drift simulation complete.")
    print(f"Existing rows before drift: {len(existing_log)}")
    print(f"New drifted prediction rows: {len(drift_predictions)}")
    print(f"Total rows after drift: {len(combined_log)}")
    print(f"Updated prediction log: {PREDICTIONS_LOG_PATH}")
    print(f"Drift-only file: {DRIFT_ONLY_PATH}")

    if BACKUP_LOG_PATH.exists():
        print(f"Backup before drift: {BACKUP_LOG_PATH}")

    print()
    print("Decision counts after drift:")
    print(combined_log["decision"].value_counts().to_string())

    print()
    print("Average default probability after drift:")
    print(round(combined_log["default_probability"].mean(), 4))


if __name__ == "__main__":
    main()
