import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator


MODEL_PATH = Path("training/artifacts/model.pkl")
PREDICTIONS_LOG_PATH = Path("data/processed/predictions_log.csv")
DECISION_THRESHOLD = 0.35

VALID_GENDERS = {
    "male": "Male",
    "female": "Female",
}

VALID_CHANNELS = {
    "dealer": "DEALER",
    "direct": "DIRECT",
    "online": "ONLINE",
}

VALID_MARITAL_STATUS = {
    "single": "Single",
    "married": "Married",
    "live-in": "Live-in",
    "widowed/separated": "Widowed/Separated",
}

VALID_MANUFACTURERS = {
    "honda": "Honda",
    "yamaha": "Yamaha",
    "suzuki": "Suzuki",
    "kawasaki": "Kawasaki",
    "rusi": "Rusi",
    "tmx": "TMX",
}

VALID_OCCUPATIONS = {
    "tricycle/jeepney driver": "Tricycle/Jeepney Driver",
    "sari-sari store owner": "Sari-sari Store Owner",
    "construction worker": "Construction Worker",
    "employed": "Employed",
    "ofw dependent": "OFW Dependent",
    "other self employed": "Other Self Employed",
}

VALID_RESIDENT_STATUS = {
    "owned": "Owned",
    "rented": "Rented",
    "living with parents": "Living with Parents",
}

VALID_EXISTING_LOAN = {
    "yes": "Yes",
    "no": "No",
}


app = FastAPI(
    title="MOTO² API",
    description=(
        "FastAPI service for Philippine motorcycle loan default prediction "
        "with label latency awareness, application validation, and prediction logging."
    ),
    version="1.2.0",
)


def normalize_category(value: str, valid_map: dict, field_name: str) -> str:
    normalized = value.strip().lower()

    if normalized not in valid_map:
        allowed_values = list(valid_map.values())
        raise ValueError(
            f"Invalid {field_name}: {value}. Allowed values: {allowed_values}"
        )

    return valid_map[normalized]


class LoanApplication(BaseModel):
    applicant_name: Optional[str] = Field(
        default="Unknown Applicant",
        description="Name of applicant for display only. Not used by the model.",
    )

    bureau_score: int = Field(
        ...,
        description="Credit bureau score. Use -1 if borrower has no bureau history.",
    )

    down_payment_amount: float = Field(
        ...,
        ge=0,
        description="Borrower's down payment amount in PHP.",
    )

    res_years_at_current_city: int = Field(
        ...,
        ge=0,
        le=70,
        description="Years living in current city.",
    )

    res_years_at_current_address: int = Field(
        ...,
        ge=0,
        le=70,
        description="Years living at current address.",
    )

    interest_rate: float = Field(
        ...,
        ge=5,
        le=30,
        description="Annual interest rate percentage.",
    )

    loan_asset_cost: float = Field(
        ...,
        ge=30000,
        le=200000,
        description="Motorcycle asset cost in PHP.",
    )

    age: int = Field(
        ...,
        ge=18,
        le=70,
        description="Borrower age.",
    )

    number_of_installments: int = Field(
        ...,
        description="Loan term in months. Valid values: 12, 18, 24, 36.",
    )

    total_income: float = Field(
        ...,
        ge=5000,
        le=200000,
        description="Declared monthly income in PHP.",
    )

    years_in_occupation: int = Field(
        ...,
        ge=0,
        le=60,
        description="Years in current occupation.",
    )

    number_of_dependents: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of dependents.",
    )

    gender: str
    channel: str
    marital_status: str
    manufacturer: str
    occupation_type: str
    res_resident_status: str
    has_existing_loan: str

    loan_origination_date: Optional[date] = Field(
        default=None,
        description=(
            "Date of loan application/origination. Used only to estimate "
            "when the label becomes observable."
        ),
    )

    @field_validator("bureau_score")
    @classmethod
    def validate_bureau_score(cls, value: int) -> int:
        if value == -1:
            return value

        if 300 <= value <= 900:
            return value

        raise ValueError("bureau_score must be -1 or between 300 and 900.")

    @field_validator("number_of_installments")
    @classmethod
    def validate_installments(cls, value: int) -> int:
        valid_terms = {12, 18, 24, 36}

        if value not in valid_terms:
            raise ValueError("number_of_installments must be one of: 12, 18, 24, 36.")

        return value

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        return normalize_category(value, VALID_GENDERS, "gender")

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, value: str) -> str:
        return normalize_category(value, VALID_CHANNELS, "channel")

    @field_validator("marital_status")
    @classmethod
    def validate_marital_status(cls, value: str) -> str:
        return normalize_category(value, VALID_MARITAL_STATUS, "marital_status")

    @field_validator("manufacturer")
    @classmethod
    def validate_manufacturer(cls, value: str) -> str:
        return normalize_category(value, VALID_MANUFACTURERS, "manufacturer")

    @field_validator("occupation_type")
    @classmethod
    def validate_occupation_type(cls, value: str) -> str:
        return normalize_category(value, VALID_OCCUPATIONS, "occupation_type")

    @field_validator("res_resident_status")
    @classmethod
    def validate_resident_status(cls, value: str) -> str:
        return normalize_category(value, VALID_RESIDENT_STATUS, "res_resident_status")

    @field_validator("has_existing_loan")
    @classmethod
    def validate_existing_loan(cls, value: str) -> str:
        return normalize_category(value, VALID_EXISTING_LOAN, "has_existing_loan")

    @model_validator(mode="after")
    def validate_cross_field_rules(self):
        if self.res_years_at_current_address > self.res_years_at_current_city:
            raise ValueError(
                "res_years_at_current_address cannot be greater than "
                "res_years_at_current_city."
            )

        max_possible_work_years = self.age - 18
        if self.years_in_occupation > max_possible_work_years:
            raise ValueError(
                "years_in_occupation cannot be greater than age - 18."
            )

        if self.down_payment_amount >= self.loan_asset_cost:
            raise ValueError(
                "down_payment_amount must be lower than loan_asset_cost."
            )

        down_payment_ratio = self.down_payment_amount / self.loan_asset_cost
        if down_payment_ratio < 0.05:
            raise ValueError(
                "down_payment_amount must be at least 5% of loan_asset_cost."
            )

        if down_payment_ratio > 0.60:
            raise ValueError(
                "down_payment_amount is unusually high. It must not exceed "
                "60% of loan_asset_cost for this demo."
            )

        if self.loan_origination_date is not None:
            if self.loan_origination_date > date.today():
                raise ValueError("loan_origination_date cannot be in the future.")

        return self


class PredictionResponse(BaseModel):
    applicant_name: str
    decision: str
    default_probability: float
    default_probability_percent: str
    approval_confidence: float
    top_reasons: List[str]
    risk_factors: List[str]
    model_version: str
    label_status: str
    expected_label_date: str
    timestamp: str


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run python training/train.py first."
        )

    artifact = joblib.load(MODEL_PATH)
    return artifact


model_artifact = load_model_artifact()
pipeline = model_artifact["pipeline"]
model_features = model_artifact["model_features"]
threshold = DECISION_THRESHOLD
model_name = model_artifact.get("model_name", "unknown_model")


def build_model_input(application: LoanApplication) -> pd.DataFrame:
    raw_data = application.model_dump()

    raw_data.pop("applicant_name", None)
    raw_data.pop("loan_origination_date", None)

    raw_data["loan_to_income_ratio"] = (
        raw_data["loan_asset_cost"] / raw_data["total_income"]
    )

    input_df = pd.DataFrame([raw_data])

    missing_features = [
        feature for feature in model_features if feature not in input_df.columns
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing model features: {missing_features}",
        )

    return input_df[model_features]


def get_decision(default_probability: float) -> str:
    if default_probability < 0.20:
        return "APPROVED"

    if default_probability < threshold:
        return "NEEDS_MANUAL_REVIEW"

    return "HIGH_RISK_MANUAL_REVIEW"


def get_approval_confidence(default_probability: float) -> float:
    confidence = 1.0 - default_probability
    return round(float(confidence), 4)


def get_risk_factors(application: LoanApplication) -> List[str]:
    factors = []

    loan_to_income_ratio = application.loan_asset_cost / max(application.total_income, 1)
    down_payment_ratio = application.down_payment_amount / max(application.loan_asset_cost, 1)

    if application.bureau_score == -1:
        factors.append("No bureau history")
    elif application.bureau_score < 600:
        factors.append("Low bureau score")

    if application.total_income < 12000:
        factors.append("Low declared monthly income")

    if application.number_of_dependents >= 4:
        factors.append("High number of dependents")

    if application.has_existing_loan == "Yes":
        factors.append("Has existing loan obligation")

    if loan_to_income_ratio > 6:
        factors.append("High loan-to-income ratio")

    if application.res_resident_status == "Rented":
        factors.append("Rented residence")

    if application.age < 22:
        factors.append("Young borrower profile")

    if down_payment_ratio < 0.12:
        factors.append("Low down payment ratio")

    if not factors:
        factors.append("No major rule-based risk factor detected")

    return factors


def get_top_reasons(default_probability: float, risk_factors: List[str]) -> List[str]:
    if default_probability < 0.20:
        return [
            "Low predicted default probability",
            "Application is within acceptable risk range",
            "Proceed with normal document verification",
        ]

    if default_probability < threshold:
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


def get_expected_label_date(application: LoanApplication) -> str:
    if application.loan_origination_date is None:
        base_date = date.today()
    else:
        base_date = application.loan_origination_date

    expected_date = base_date + timedelta(days=90)
    return expected_date.isoformat()


def append_prediction_log(
    application: LoanApplication,
    response: PredictionResponse,
    input_df: pd.DataFrame,
) -> None:
    """
    Save every model prediction.

    This log is intentionally label-pending at prediction time.
    Actual TARGET fields are blank first and can be filled later when
    the loan outcome becomes mature.
    """
    PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    input_payload = application.model_dump(mode="json")
    loan_to_income_ratio = float(input_df["loan_to_income_ratio"].iloc[0])

    record = {
        "prediction_id": str(uuid4()),
        "prediction_timestamp": response.timestamp,
        "applicant_name": response.applicant_name,
        "model_version": response.model_version,
        "decision": response.decision,
        "default_probability": response.default_probability,
        "default_probability_percent": response.default_probability_percent,
        "approval_confidence": response.approval_confidence,
        "top_reasons": " | ".join(response.top_reasons),
        "risk_factors": " | ".join(response.risk_factors),
        "label_status": response.label_status,
        "expected_label_date": response.expected_label_date,
        "label_is_mature_at_prediction": False,
        "actual_TARGET": "",
        "actual_days_past_due": "",
        "actual_label_arrival_date": "",
        "loan_origination_date": (
            application.loan_origination_date.isoformat()
            if application.loan_origination_date
            else ""
        ),
        "bureau_score": application.bureau_score,
        "down_payment_amount": application.down_payment_amount,
        "res_years_at_current_city": application.res_years_at_current_city,
        "res_years_at_current_address": application.res_years_at_current_address,
        "interest_rate": application.interest_rate,
        "loan_asset_cost": application.loan_asset_cost,
        "age": application.age,
        "number_of_installments": application.number_of_installments,
        "total_income": application.total_income,
        "years_in_occupation": application.years_in_occupation,
        "number_of_dependents": application.number_of_dependents,
        "loan_to_income_ratio": loan_to_income_ratio,
        "gender": application.gender,
        "channel": application.channel,
        "marital_status": application.marital_status,
        "manufacturer": application.manufacturer,
        "occupation_type": application.occupation_type,
        "res_resident_status": application.res_resident_status,
        "has_existing_loan": application.has_existing_loan,
        "input_payload_json": json.dumps(input_payload, ensure_ascii=False),
    }

    file_exists = PREDICTIONS_LOG_PATH.exists()
    needs_header = (not file_exists) or PREDICTIONS_LOG_PATH.stat().st_size == 0

    with open(PREDICTIONS_LOG_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(record.keys()))

        if needs_header:
            writer.writeheader()

        writer.writerow(record)


@app.get("/")
def root():
    return {
        "message": "MOTO² API is running.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "model_name": model_name,
        "model_path": str(MODEL_PATH),
        "feature_count": len(model_features),
        "threshold": threshold,
        "application_validation": "enabled",
        "prediction_logging": "enabled",
        "prediction_log_path": str(PREDICTIONS_LOG_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    try:
        input_df = build_model_input(application)
        default_probability = float(pipeline.predict_proba(input_df)[0, 1])

        decision = get_decision(default_probability)
        approval_confidence = get_approval_confidence(default_probability)
        risk_factors = get_risk_factors(application)
        top_reasons = get_top_reasons(default_probability, risk_factors)

        response = PredictionResponse(
            applicant_name=application.applicant_name or "Unknown Applicant",
            decision=decision,
            default_probability=round(default_probability, 4),
            default_probability_percent=f"{default_probability * 100:.2f}%",
            approval_confidence=approval_confidence,
            top_reasons=top_reasons,
            risk_factors=risk_factors,
            model_version=f"{model_name}_v1",
            label_status="PENDING, outcome observable after 90+ days",
            expected_label_date=get_expected_label_date(application),
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

        append_prediction_log(
            application=application,
            response=response,
            input_df=input_df,
        )

        return response

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )
