from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REFERENCE_DATA_PATH = Path("data/raw/ph_motorcycle_loans.csv")
PREDICTIONS_LOG_PATH = Path("data/processed/predictions_log.csv")
REPORT_DIR = Path("monitoring/reports")

REFERENCE_DATE = "2026-06-04"
MIN_CURRENT_ROWS_FOR_DRIFT = 30

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

    return reference_df


def load_prediction_log() -> pd.DataFrame:
    if not PREDICTIONS_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {PREDICTIONS_LOG_PATH}. "
            "Run at least one API prediction first."
        )

    df = pd.read_csv(PREDICTIONS_LOG_PATH)

    if df.empty:
        raise ValueError("Prediction log exists, but it has no rows.")

    return df


def safe_psi(reference_values: pd.Series, current_values: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index for numeric features.

    Interpretation for demo:
    PSI < 0.10: stable
    0.10 to 0.25: moderate shift
    > 0.25: significant shift
    """
    reference_values = pd.to_numeric(reference_values, errors="coerce").dropna()
    current_values = pd.to_numeric(current_values, errors="coerce").dropna()

    if reference_values.empty or current_values.empty:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(reference_values.quantile(quantiles).to_numpy())

    if len(bin_edges) < 3:
        return np.nan

    reference_counts, _ = np.histogram(reference_values, bins=bin_edges)
    current_counts, _ = np.histogram(current_values, bins=bin_edges)

    reference_pct = reference_counts / max(reference_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)

    epsilon = 0.0001
    reference_pct = np.where(reference_pct == 0, epsilon, reference_pct)
    current_pct = np.where(current_pct == 0, epsilon, current_pct)

    psi = np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))
    return float(psi)


def categorical_psi(reference_values: pd.Series, current_values: pd.Series) -> float:
    reference_values = reference_values.fillna("MISSING").astype(str)
    current_values = current_values.fillna("MISSING").astype(str)

    categories = sorted(set(reference_values.unique()) | set(current_values.unique()))

    if not categories:
        return np.nan

    reference_dist = reference_values.value_counts(normalize=True).reindex(
        categories,
        fill_value=0,
    )
    current_dist = current_values.value_counts(normalize=True).reindex(
        categories,
        fill_value=0,
    )

    epsilon = 0.0001
    reference_pct = reference_dist.replace(0, epsilon)
    current_pct = current_dist.replace(0, epsilon)

    psi = ((current_pct - reference_pct) * np.log(current_pct / reference_pct)).sum()
    return float(psi)


def psi_status(psi: float, current_rows: int) -> str:
    if pd.isna(psi):
        return "INSUFFICIENT_DATA"

    if current_rows < MIN_CURRENT_ROWS_FOR_DRIFT:
        return "TOO_FEW_ROWS_FOR_RELIABLE_DRIFT"

    if psi < 0.10:
        return "STABLE"

    if psi < 0.25:
        return "MODERATE_SHIFT"

    return "SIGNIFICANT_SHIFT"


def create_numeric_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict] = []

    current_rows = len(current_df)

    for feature in NUMERIC_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            rows.append(
                {
                    "feature": feature,
                    "status": "MISSING_FEATURE",
                    "reference_mean": np.nan,
                    "current_mean": np.nan,
                    "mean_difference": np.nan,
                    "psi": np.nan,
                }
            )
            continue

        reference_series = pd.to_numeric(reference_df[feature], errors="coerce")
        current_series = pd.to_numeric(current_df[feature], errors="coerce")

        reference_mean = reference_series.mean()
        current_mean = current_series.mean()
        mean_difference = current_mean - reference_mean
        psi = safe_psi(reference_series, current_series)

        rows.append(
            {
                "feature": feature,
                "status": psi_status(psi, current_rows),
                "reference_mean": round(reference_mean, 4),
                "current_mean": round(current_mean, 4),
                "mean_difference": round(mean_difference, 4),
                "reference_p50": round(reference_series.median(), 4),
                "current_p50": round(current_series.median(), 4),
                "reference_p95": round(reference_series.quantile(0.95), 4),
                "current_p95": round(current_series.quantile(0.95), 4),
                "psi": round(psi, 4) if not pd.isna(psi) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def create_categorical_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict] = []
    current_rows = len(current_df)

    for feature in CATEGORICAL_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            rows.append(
                {
                    "feature": feature,
                    "status": "MISSING_FEATURE",
                    "reference_top_category": "",
                    "current_top_category": "",
                    "unseen_categories_in_current": "",
                    "psi": np.nan,
                }
            )
            continue

        reference_series = reference_df[feature].fillna("MISSING").astype(str)
        current_series = current_df[feature].fillna("MISSING").astype(str)

        reference_top = reference_series.value_counts().idxmax()
        current_top = current_series.value_counts().idxmax()

        unseen_categories = sorted(set(current_series.unique()) - set(reference_series.unique()))
        psi = categorical_psi(reference_series, current_series)

        rows.append(
            {
                "feature": feature,
                "status": psi_status(psi, current_rows),
                "reference_top_category": reference_top,
                "current_top_category": current_top,
                "unseen_categories_in_current": ", ".join(unseen_categories),
                "psi": round(psi, 4) if not pd.isna(psi) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def create_decision_monitoring(current_df: pd.DataFrame) -> pd.DataFrame:
    decision_counts = current_df["decision"].value_counts(dropna=False).reset_index()
    decision_counts.columns = ["decision", "count"]
    decision_counts["share"] = decision_counts["count"] / len(current_df)
    decision_counts["share"] = decision_counts["share"].round(4)

    return decision_counts


def create_summary(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_report: pd.DataFrame,
    categorical_report: pd.DataFrame,
) -> pd.DataFrame:
    high_numeric_alerts = numeric_report[
        numeric_report["status"].isin(["MODERATE_SHIFT", "SIGNIFICANT_SHIFT"])
    ]

    high_categorical_alerts = categorical_report[
        categorical_report["status"].isin(["MODERATE_SHIFT", "SIGNIFICANT_SHIFT"])
    ]

    pending_label_share = (
        current_df["label_status"]
        .astype(str)
        .str.contains("PENDING", case=False, na=False)
        .mean()
    )

    actual_target_blank_share = current_df["actual_TARGET"].isna().mean()

    summary = {
        "report_generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_rows_matured_training": len(reference_df),
        "current_logged_predictions": len(current_df),
        "average_default_probability": round(current_df["default_probability"].mean(), 4),
        "max_default_probability": round(current_df["default_probability"].max(), 4),
        "pending_label_share": round(pending_label_share, 4),
        "actual_target_blank_share": round(actual_target_blank_share, 4),
        "numeric_drift_alert_count": len(high_numeric_alerts),
        "categorical_drift_alert_count": len(high_categorical_alerts),
        "monitoring_note": (
            "Current prediction log has too few rows for reliable drift detection."
            if len(current_df) < MIN_CURRENT_ROWS_FOR_DRIFT
            else "Current prediction log has enough rows for basic drift checks."
        ),
    }

    return pd.DataFrame([summary])


def write_html_report(
    summary_df: pd.DataFrame,
    numeric_report: pd.DataFrame,
    categorical_report: pd.DataFrame,
    decision_report: pd.DataFrame,
    output_path: Path,
) -> None:
    html = f"""
    <html>
    <head>
        <title>MOTO² Monitoring Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 32px;
                line-height: 1.5;
            }}
            h1, h2 {{
                color: #263238;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 24px;
                font-size: 14px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .note {{
                background: #fff8e1;
                padding: 12px;
                border: 1px solid #ffecb3;
                margin-bottom: 24px;
            }}
        </style>
    </head>
    <body>
        <h1>MOTO² Monitoring Report</h1>

        <div class="note">
            <strong>Label latency note:</strong>
            Most production predictions are label-pending at scoring time.
            This report monitors prediction behavior now, while delayed actual labels
            are filled later for performance evaluation and retraining.
        </div>

        <h2>Summary</h2>
        {summary_df.to_html(index=False)}

        <h2>Decision Monitoring</h2>
        {decision_report.to_html(index=False)}

        <h2>Numeric Feature Drift</h2>
        {numeric_report.to_html(index=False)}

        <h2>Categorical Feature Drift</h2>
        {categorical_report.to_html(index=False)}
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reference_df = load_reference_data()
    current_df = load_prediction_log()

    numeric_report = create_numeric_drift_report(reference_df, current_df)
    categorical_report = create_categorical_drift_report(reference_df, current_df)
    decision_report = create_decision_monitoring(current_df)
    summary_df = create_summary(
        reference_df,
        current_df,
        numeric_report,
        categorical_report,
    )

    summary_path = REPORT_DIR / "monitoring_summary.csv"
    numeric_path = REPORT_DIR / "numeric_drift_report.csv"
    categorical_path = REPORT_DIR / "categorical_drift_report.csv"
    decision_path = REPORT_DIR / "decision_monitoring.csv"
    html_path = REPORT_DIR / "monitoring_report.html"

    summary_df.to_csv(summary_path, index=False)
    numeric_report.to_csv(numeric_path, index=False)
    categorical_report.to_csv(categorical_path, index=False)
    decision_report.to_csv(decision_path, index=False)

    write_html_report(
        summary_df=summary_df,
        numeric_report=numeric_report,
        categorical_report=categorical_report,
        decision_report=decision_report,
        output_path=html_path,
    )

    print("Monitoring complete.")
    print(f"Summary saved to: {summary_path}")
    print(f"Numeric drift report saved to: {numeric_path}")
    print(f"Categorical drift report saved to: {categorical_path}")
    print(f"Decision monitoring saved to: {decision_path}")
    print(f"HTML report saved to: {html_path}")
    print()
    print("Summary:")
    print(summary_df.to_string(index=False))
    print()
    print("Decision counts:")
    print(decision_report.to_string(index=False))


if __name__ == "__main__":
    main()
