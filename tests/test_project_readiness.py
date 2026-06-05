from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


RAW_DATA_PATH = Path("data/raw/ph_motorcycle_loans.csv")
MODEL_PATH = Path("training/artifacts/model.pkl")
PREDICTIONS_LOG_PATH = Path("data/processed/predictions_log.csv")

REQUIRED_COLUMNS = [
    "loan_account_no",
    "product_description",
    "loan_origination_date",
    "bureau_score",
    "down_payment_amount",
    "branch_name",
    "res_years_at_current_city",
    "res_years_at_current_address",
    "interest_rate",
    "loan_asset_cost",
    "gender",
    "channel",
    "age",
    "number_of_installments",
    "res_city",
    "res_province",
    "total_income",
    "asset_model",
    "industry",
    "nature_of_business",
    "marital_status",
    "manufacturer",
    "occupation_type",
    "res_resident_status",
    "years_in_occupation",
    "has_existing_loan",
    "number_of_dependents",
    "days_past_due",
    "label_determination_date",
    "label_is_mature",
    "TARGET",
]

LEAKAGE_COLUMNS = {
    "days_past_due",
    "label_determination_date",
    "loan_origination_date",
    "TARGET",
    "label_is_mature",
}


VALID_PAYLOAD = {
    "applicant_name": "Test Borrower",
    "bureau_score": 650,
    "down_payment_amount": 12000,
    "res_years_at_current_city": 10,
    "res_years_at_current_address": 5,
    "interest_rate": 18.5,
    "loan_asset_cost": 85000,
    "age": 38,
    "number_of_installments": 24,
    "total_income": 18000,
    "years_in_occupation": 8,
    "number_of_dependents": 3,
    "gender": "Male",
    "channel": "DEALER",
    "marital_status": "Married",
    "manufacturer": "Honda",
    "occupation_type": "Tricycle/Jeepney Driver",
    "res_resident_status": "Owned",
    "has_existing_loan": "No",
    "loan_origination_date": "2026-06-04",
}


def test_raw_dataset_exists_and_has_required_columns():
    assert RAW_DATA_PATH.exists(), f"Missing dataset: {RAW_DATA_PATH}"

    df = pd.read_csv(RAW_DATA_PATH)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    assert df.shape[0] == 5000
    assert not missing_columns, f"Missing columns: {missing_columns}"


def test_target_matches_d3_default_rule():
    df = pd.read_csv(RAW_DATA_PATH)

    expected_target = (df["days_past_due"] >= 90).astype(int)
    actual_target = df["TARGET"].astype(int)

    assert (expected_target == actual_target).all()


def test_model_artifact_exists_and_has_expected_keys():
    assert MODEL_PATH.exists(), f"Missing model artifact: {MODEL_PATH}"

    artifact = joblib.load(MODEL_PATH)

    assert "pipeline" in artifact
    assert "model_features" in artifact
    assert "model_name" in artifact
    assert "threshold" in artifact


def test_model_features_do_not_contain_leakage_columns():
    artifact = joblib.load(MODEL_PATH)
    model_features = set(artifact["model_features"])

    leakage_overlap = model_features & LEAKAGE_COLUMNS

    assert leakage_overlap == set(), f"Leakage columns found: {leakage_overlap}"


def test_fastapi_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 19
    assert data["application_validation"] == "enabled"
    assert data["prediction_logging"] == "enabled"


def test_fastapi_predict_endpoint_returns_expected_fields():
    client = TestClient(app)

    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()

    assert response.status_code == 200
    assert data["applicant_name"] == "Test Borrower"
    assert "decision" in data
    assert "default_probability" in data
    assert "default_probability_percent" in data
    assert "approval_confidence" in data
    assert "risk_factors" in data
    assert "top_reasons" in data
    assert data["label_status"] == "PENDING, outcome observable after 90+ days"
    assert data["expected_label_date"] == "2026-09-02"


def test_fastapi_rejects_invalid_application():
    client = TestClient(app)

    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["age"] = 16
    invalid_payload["bureau_score"] = 100
    invalid_payload["gender"] = "Unknown"

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_prediction_log_exists_after_prediction():
    client = TestClient(app)

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert PREDICTIONS_LOG_PATH.exists()

    log_df = pd.read_csv(PREDICTIONS_LOG_PATH)

    assert len(log_df) >= 1
    assert "prediction_id" in log_df.columns
    assert "label_status" in log_df.columns
    assert "actual_TARGET" in log_df.columns
    assert log_df["label_status"].astype(str).str.contains("PENDING").any()
