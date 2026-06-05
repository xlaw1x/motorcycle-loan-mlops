from pathlib import Path

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report


REFERENCE_DATA_PATH = Path("data/raw/ph_motorcycle_loans.csv")
PREDICTIONS_LOG_PATH = Path("data/processed/predictions_log.csv")
REPORT_DIR = Path("monitoring/reports")
EVIDENTLY_REPORT_PATH = REPORT_DIR / "evidently_report.html"

NUMERIC_FEATURES = [
    "bureau_score",
    "down_payment_amount",
    "res_years_at_current_city",
    "res_years_at_current_address",
    "interest_rate",
    "loan_asset_cost",
    "age",
    "number_of_installments",
    "total_income",
    "years_in_occupation",
    "number_of_dependents",
    "loan_to_income_ratio",
]

CATEGORICAL_FEATURES = [
    "gender",
    "channel",
    "marital_status",
    "manufacturer",
    "occupation_type",
    "res_resident_status",
    "has_existing_loan",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_reference_data() -> pd.DataFrame:
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference data not found: {REFERENCE_DATA_PATH}")

    df = pd.read_csv(REFERENCE_DATA_PATH)
    mature_mask = df["label_is_mature"].astype(str).str.lower() == "true"
    reference_df = df[mature_mask].copy()
    reference_df["loan_to_income_ratio"] = (
        reference_df["loan_asset_cost"] / reference_df["total_income"].replace(0, np.nan)
    )
    reference_df["loan_to_income_ratio"] = reference_df["loan_to_income_ratio"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return reference_df[FEATURES].copy()


def load_current_data() -> pd.DataFrame:
    if not PREDICTIONS_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {PREDICTIONS_LOG_PATH}. "
            "Run API predictions or monitoring/simulate_drift.py first."
        )

    df = pd.read_csv(PREDICTIONS_LOG_PATH)
    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Prediction log missing features: {missing_features}")

    return df[FEATURES].copy()


def normalize_for_evidently(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()

    for column in NUMERIC_FEATURES:
        normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        normalized_df[column] = normalized_df[column].fillna("MISSING").astype(str)

    return normalized_df


def write_missing_data_report(message: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENTLY_REPORT_PATH.write_text(
        f"""
        <html>
        <head><title>MOTO² Evidently Report</title></head>
        <body style="font-family: Arial, sans-serif; margin: 32px;">
            <h1>MOTO² Evidently Report</h1>
            <p>{message}</p>
        </body>
        </html>
        """,
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        reference_df = normalize_for_evidently(load_reference_data())
        current_df = normalize_for_evidently(load_current_data())
    except (FileNotFoundError, ValueError) as exc:
        write_missing_data_report(str(exc))
        print(f"Evidently placeholder report saved to: {EVIDENTLY_REPORT_PATH}")
        return

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(str(EVIDENTLY_REPORT_PATH))

    print(f"Evidently report saved to: {EVIDENTLY_REPORT_PATH}")


if __name__ == "__main__":
    main()
