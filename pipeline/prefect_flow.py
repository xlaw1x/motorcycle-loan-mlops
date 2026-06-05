from datetime import datetime
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task


DATA_PATH = Path("data/raw/ph_motorcycle_loans.csv")
MODEL_PATH = Path("training/artifacts/model.pkl")
OUTPUT_DIR = Path("data/processed")

LATEST_OUTPUT_PATH = OUTPUT_DIR / "batch_scoring_output.csv"

REFERENCE_DATE = "2026-06-04"
DECISION_THRESHOLD = 0.35

LEAKAGE_COLUMNS = [
    "days_past_due",
    "label_determination_date",
    "loan_origination_date",
    "TARGET",
    "label_is_mature",
]


@task
def load_model_artifact() -> dict:
    logger = get_run_logger()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run python training/train.py first."
        )

    artifact = joblib.load(MODEL_PATH)

    logger.info("Loaded model artifact from %s", MODEL_PATH)
    logger.info("Model name: %s", artifact.get("model_name"))
    logger.info("Trained on: %s", artifact.get("trained_on"))

    return artifact


@task
def load_applications() -> pd.DataFrame:
    logger = get_run_logger()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    logger.info("Loaded applications from %s", DATA_PATH)
    logger.info("Loaded shape: %s", df.shape)

    return df


@task
def select_label_pending_applications(
    df: pd.DataFrame,
    max_rows: int = 100,
) -> pd.DataFrame:
    """
    For batch scoring, we simulate scoring applications whose labels are still pending.

    In real life, these are the applications where the model prediction is needed now,
    but the true repayment outcome is not yet observable.
    """
    logger = get_run_logger()

    pending_mask = df["label_is_mature"].astype(str).str.lower() == "false"
    pending_df = df[pending_mask].copy()

    if pending_df.empty:
        logger.warning("No label-pending rows found. Falling back to latest rows.")
        pending_df = df.sort_values("loan_origination_date", ascending=False).head(max_rows)
    else:
        pending_df = pending_df.head(max_rows)

    logger.info("Selected %s applications for batch scoring.", len(pending_df))

    return pending_df


@task
def prepare_features(
    applications_df: pd.DataFrame,
    model_features: list,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_run_logger()

    scoring_df = applications_df.copy()

    scoring_df["loan_to_income_ratio"] = (
        scoring_df["loan_asset_cost"] / scoring_df["total_income"].replace(0, np.nan)
    )
    scoring_df["loan_to_income_ratio"] = scoring_df["loan_to_income_ratio"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    scoring_df["loan_to_income_ratio"] = scoring_df["loan_to_income_ratio"].fillna(
        scoring_df["loan_to_income_ratio"].median()
    )

    missing_features = [
        feature for feature in model_features if feature not in scoring_df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing model features for scoring: {missing_features}")

    leakage_overlap = set(model_features) & set(LEAKAGE_COLUMNS)

    if leakage_overlap:
        raise ValueError(f"Leakage columns found in model features: {leakage_overlap}")

    X_score = scoring_df[model_features].copy()

    metadata_columns = [
        "loan_account_no",
        "loan_origination_date",
        "label_determination_date",
        "label_is_mature",
    ]

    available_metadata = [
        column for column in metadata_columns if column in scoring_df.columns
    ]

    metadata_df = scoring_df[available_metadata].copy()

    logger.info("Prepared scoring feature shape: %s", X_score.shape)
    logger.info("Leakage guard passed for batch scoring.")

    return X_score, metadata_df


@task
def score_applications(
    artifact: dict,
    X_score: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    logger = get_run_logger()

    pipeline = artifact["pipeline"]
    model_name = artifact.get("model_name", "unknown_model")
    threshold = DECISION_THRESHOLD

    probabilities = pipeline.predict_proba(X_score)[:, 1]

    output_df = metadata_df.copy()
    output_df["default_probability"] = probabilities.round(4)
    output_df["default_probability_percent"] = [
        f"{prob * 100:.2f}%" for prob in probabilities
    ]

    def make_decision(probability: float) -> str:
        if probability < 0.20:
            return "APPROVED"
        if probability < threshold:
            return "NEEDS_MANUAL_REVIEW"
        return "HIGH_RISK_MANUAL_REVIEW"

    output_df["decision"] = [make_decision(prob) for prob in probabilities]
    output_df["model_version"] = f"{model_name}_v1"
    output_df["threshold"] = threshold
    output_df["label_status"] = "PENDING, outcome observable after 90+ days"

    output_df["expected_label_date"] = (
        pd.to_datetime(output_df["loan_origination_date"]) + pd.Timedelta(days=90)
    ).dt.strftime("%Y-%m-%d")

    output_df["scored_at"] = datetime.now().isoformat(timespec="seconds")

    logger.info("Scored %s applications.", len(output_df))
    logger.info("Decision counts: %s", output_df["decision"].value_counts().to_dict())

    return output_df


@task
def save_scoring_output(scored_df: pd.DataFrame) -> str:
    logger = get_run_logger()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_path = OUTPUT_DIR / f"batch_scoring_output_{timestamp}.csv"

    scored_df.to_csv(LATEST_OUTPUT_PATH, index=False)
    scored_df.to_csv(timestamped_output_path, index=False)

    logger.info("Saved latest output to %s", LATEST_OUTPUT_PATH)
    logger.info("Saved timestamped output to %s", timestamped_output_path)

    return str(timestamped_output_path)


@flow(
    name="moto-loan-batch-scoring-flow",
    log_prints=True,
)
def batch_scoring_flow(max_rows: int = 100) -> str:
    artifact = load_model_artifact()
    df = load_applications()

    model_features = artifact["model_features"]

    applications_df = select_label_pending_applications(df, max_rows=max_rows)
    X_score, metadata_df = prepare_features(applications_df, model_features)

    scored_df = score_applications(artifact, X_score, metadata_df)
    output_path = save_scoring_output(scored_df)

    print("Batch scoring flow complete.")
    print(f"Output saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    batch_scoring_flow(max_rows=100)
