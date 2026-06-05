from pathlib import Path
import json
import joblib
import pandas as pd


CHECKS = []

RAW_DATA_PATH = Path("data/raw/ph_motorcycle_loans.csv")
MODEL_PATH = Path("training/artifacts/model.pkl")
METRICS_PATH = Path("training/artifacts/metrics.json")
PREDICTIONS_LOG_PATH = Path("data/processed/predictions_log.csv")
MONITORING_REPORT_PATH = Path("monitoring/reports/monitoring_report.html")
DELAYED_REPORT_PATH = Path("retrain/delayed_performance_report.csv")
RETRAIN_DECISION_PATH = Path("retrain/retraining_decision.txt")

LEAKAGE_COLUMNS = {
    "days_past_due",
    "label_determination_date",
    "loan_origination_date",
    "TARGET",
    "label_is_mature",
}


def pass_check(name, detail=""):
    CHECKS.append(("PASS", name, detail))


def warn_check(name, detail=""):
    CHECKS.append(("WARN", name, detail))


def fail_check(name, detail=""):
    CHECKS.append(("FAIL", name, detail))


def check_file_exists(path, name):
    if path.exists():
        pass_check(name, str(path))
    else:
        fail_check(name, f"Missing: {path}")


def check_dataset():
    if not RAW_DATA_PATH.exists():
        fail_check("Raw dataset", f"Missing: {RAW_DATA_PATH}")
        return

    df = pd.read_csv(RAW_DATA_PATH)

    if df.shape[0] == 5000:
        pass_check("Dataset row count", f"{df.shape[0]} rows")
    else:
        warn_check("Dataset row count", f"Expected 5000, got {df.shape[0]}")

    if df.shape[1] == 31:
        pass_check("Dataset column count", f"{df.shape[1]} columns")
    else:
        warn_check("Dataset column count", f"Expected 31, got {df.shape[1]}")

    expected_target = (df["days_past_due"] >= 90).astype(int)
    actual_target = df["TARGET"].astype(int)

    if (expected_target == actual_target).all():
        pass_check("D3 default rule", "TARGET matches days_past_due >= 90")
    else:
        fail_check("D3 default rule", "TARGET does not match days_past_due >= 90")

    mature_share = df["label_is_mature"].astype(str).str.lower().eq("true").mean()
    pass_check("Label maturity present", f"Mature label share: {mature_share:.2%}")


def check_model():
    if not MODEL_PATH.exists():
        fail_check("Model artifact", f"Missing: {MODEL_PATH}")
        return

    artifact = joblib.load(MODEL_PATH)

    required_keys = ["pipeline", "model_features", "model_name", "threshold"]
    missing_keys = [key for key in required_keys if key not in artifact]

    if not missing_keys:
        pass_check("Model artifact keys", "pipeline, model_features, model_name, threshold found")
    else:
        fail_check("Model artifact keys", f"Missing keys: {missing_keys}")

    model_features = set(artifact.get("model_features", []))
    leakage_overlap = model_features & LEAKAGE_COLUMNS

    if leakage_overlap == set():
        pass_check("Leakage guard", "No leakage columns in model features")
    else:
        fail_check("Leakage guard", f"Leakage columns found: {leakage_overlap}")

    pass_check("Model feature count", f"{len(model_features)} features")


def check_metrics():
    if not METRICS_PATH.exists():
        fail_check("Training metrics", f"Missing: {METRICS_PATH}")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    required_metrics = ["auc_roc", "precision", "recall", "f1", "accuracy"]
    missing_metrics = [metric for metric in required_metrics if metric not in metrics]

    if not missing_metrics:
        pass_check(
            "Training metrics",
            f"AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}",
        )
    else:
        fail_check("Training metrics", f"Missing metrics: {missing_metrics}")


def check_prediction_log():
    if not PREDICTIONS_LOG_PATH.exists():
        fail_check("Prediction log", f"Missing: {PREDICTIONS_LOG_PATH}")
        return

    df = pd.read_csv(PREDICTIONS_LOG_PATH)

    if len(df) >= 1:
        pass_check("Prediction log rows", f"{len(df)} rows")
    else:
        fail_check("Prediction log rows", "No logged predictions")

    required_columns = [
        "prediction_id",
        "applicant_name",
        "decision",
        "default_probability",
        "label_status",
        "expected_label_date",
        "actual_TARGET",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]

    if not missing_columns:
        pass_check("Prediction log columns", "Required columns found")
    else:
        fail_check("Prediction log columns", f"Missing columns: {missing_columns}")

    if "label_status" in df.columns:
        pending_share = df["label_status"].astype(str).str.contains("PENDING", case=False, na=False).mean()
        pass_check("Pending label tracking", f"Pending label share: {pending_share:.2%}")

    if "actual_TARGET" in df.columns:
        blank_share = df["actual_TARGET"].isna().mean()
        pass_check("Actual target latency", f"Blank actual_TARGET share: {blank_share:.2%}")


def check_monitoring():
    check_file_exists(MONITORING_REPORT_PATH, "Monitoring HTML report")

    summary_path = Path("monitoring/reports/monitoring_summary.csv")
    numeric_path = Path("monitoring/reports/numeric_drift_report.csv")
    categorical_path = Path("monitoring/reports/categorical_drift_report.csv")
    evidently_path = Path("monitoring/reports/evidently_report.html")

    check_file_exists(summary_path, "Monitoring summary CSV")
    check_file_exists(numeric_path, "Numeric drift report")
    check_file_exists(categorical_path, "Categorical drift report")
    check_file_exists(evidently_path, "Evidently HTML report")


def check_delayed_evaluation():
    check_file_exists(DELAYED_REPORT_PATH, "Delayed performance report")
    check_file_exists(RETRAIN_DECISION_PATH, "Retraining decision report")


def check_docs_and_docker():
    required_files = [
        Path("README.md"),
        Path("DEMO_SCRIPT.md"),
        Path("FINAL_DEMO_CHECKLIST.md"),
        Path("Dockerfile.api"),
        Path("Dockerfile.frontend"),
        Path("Dockerfile.mlflow"),
        Path("Dockerfile.evidently"),
        Path("Dockerfile.prefect"),
        Path("docker-compose.yml"),
        Path("requirements-mlflow.txt"),
        Path("requirements-monitoring.txt"),
        Path("requirements-prefect.txt"),
        Path("api/main.py"),
        Path("frontend/app.py"),
        Path("monitoring/evidently_report.py"),
        Path("pipeline/prefect_flow.py"),
        Path("tests/test_project_readiness.py"),
    ]

    for path in required_files:
        check_file_exists(path, f"Required file: {path}")


def print_report():
    print("=" * 80)
    print("MOTO² FINAL QA REPORT")
    print("=" * 80)

    for status, name, detail in CHECKS:
        print(f"[{status}] {name}")
        if detail:
            print(f"       {detail}")

    print("=" * 80)

    fail_count = sum(1 for status, _, _ in CHECKS if status == "FAIL")
    warn_count = sum(1 for status, _, _ in CHECKS if status == "WARN")
    pass_count = sum(1 for status, _, _ in CHECKS if status == "PASS")

    print(f"PASS: {pass_count}")
    print(f"WARN: {warn_count}")
    print(f"FAIL: {fail_count}")

    if fail_count == 0:
        print("FINAL STATUS: DEMO READY")
    else:
        print("FINAL STATUS: NEEDS FIXES BEFORE DEMO")

    print("=" * 80)


def main():
    check_dataset()
    check_model()
    check_metrics()
    check_prediction_log()
    check_monitoring()
    check_delayed_evaluation()
    check_docs_and_docker()
    print_report()


if __name__ == "__main__":
    main()
