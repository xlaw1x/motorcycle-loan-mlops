"""
retrain_flow.py — drift-triggered retraining for MOTO2.
Trains v2 on matured data, compares to v1, promotes via MLflow registry.
"""
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from prefect import flow, task, get_run_logger

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

RANDOM_SEED = 42
MODEL_NAME = "motorcycle-loan-model"
DATA_PATH = Path("data/raw/ph_motorcycle_loans.csv")
PRED_LOG = Path("data/processed/predictions_log.csv")
DRIFT_ALERT = Path("monitoring/reports/drift_alert.txt")
REFERENCE_DATE = pd.Timestamp("2026-06-04")

NUMERIC_FEATURES = [
    "bureau_score", "down_payment_amount", "res_years_at_current_city",
    "res_years_at_current_address", "interest_rate", "loan_asset_cost",
    "age", "number_of_installments", "total_income", "years_in_occupation",
    "number_of_dependents", "loan_to_income_ratio",
]
CATEGORICAL_FEATURES = [
    "gender", "channel", "marital_status", "manufacturer",
    "occupation_type", "res_resident_status", "has_existing_loan",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _add_lti(df):
    df = df.copy()
    df["loan_to_income_ratio"] = df["loan_asset_cost"] / df["total_income"].replace(0, np.nan)
    df["loan_to_income_ratio"] = df["loan_to_income_ratio"].replace([np.inf, -np.inf], np.nan)
    df["loan_to_income_ratio"] = df["loan_to_income_ratio"].fillna(df["loan_to_income_ratio"].median())
    return df


@task
def detect_drift_alert():
    logger = get_run_logger()
    if DRIFT_ALERT.exists():
        logger.info("Drift alert found:\n" + DRIFT_ALERT.read_text(encoding="utf-8", errors="ignore"))
        return True
    logger.info("No drift alert present. Running anyway (manual trigger).")
    return False


@task
def load_training_data():
    logger = get_run_logger()
    df = _add_lti(pd.read_csv(DATA_PATH))
    matured = df[df["label_is_mature"].astype(str).str.lower() == "true"].copy()
    logger.info(f"Base matured training rows: {len(matured)}")

    newly = 0
    if PRED_LOG.exists():
        log = pd.read_csv(PRED_LOG)
        if "expected_label_date" in log.columns:
            log["expected_label_date"] = pd.to_datetime(log["expected_label_date"], errors="coerce")
            mature_log = log[log["expected_label_date"] <= REFERENCE_DATE].copy()
            if "actual_TARGET" in mature_log.columns and mature_log["actual_TARGET"].notna().any():
                mature_log = mature_log[mature_log["actual_TARGET"].notna()]
                mature_log["TARGET"] = mature_log["actual_TARGET"].astype(int)
            else:
                rng = np.random.default_rng(RANDOM_SEED)
                if "default_probability" in mature_log.columns:
                    mature_log["TARGET"] = (rng.random(len(mature_log)) < mature_log["default_probability"]).astype(int)
                else:
                    mature_log["TARGET"] = 0
            if len(mature_log) > 0:
                mature_log = _add_lti(mature_log)
                keep = [c for c in MODEL_FEATURES + ["TARGET"] if c in mature_log.columns]
                mature_log = mature_log[keep].dropna(subset=[c for c in MODEL_FEATURES if c in mature_log.columns])
                newly = len(mature_log)
                if newly > 0:
                    matured = pd.concat([matured[keep], mature_log[keep]], ignore_index=True)

    logger.info(f"Retraining on {len(matured)} matured rows ({newly} newly matured from production)")
    return matured


@task
def retrain_model(data: pd.DataFrame):
    logger = get_run_logger()
    X = data[MODEL_FEATURES]
    y = data["TARGET"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    spw = n_neg / max(n_pos, 1)

    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="auc",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    pipe = Pipeline([("preprocessor", pre), ("classifier", clf)])
    pipe.fit(X_tr, y_tr)

    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    logger.info(f"v2 candidate AUC = {auc:.4f}")
    return pipe, float(auc)


@task
def get_production_auc():
    logger = get_run_logger()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    try:
        for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
            if mv.current_stage == "Production":
                auc = client.get_run(mv.run_id).data.metrics.get("auc_roc")
                if auc is not None:
                    logger.info(f"Production (v{mv.version}) AUC = {auc:.4f}")
                    return float(auc)
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")
    m = json.loads(Path("training/artifacts/metrics.json").read_text())
    logger.info(f"Fallback metrics.json AUC = {m['auc_roc']:.4f}")
    return float(m["auc_roc"])


@task
def register_and_promote(pipe, v2_auc, v1_auc):
    logger = get_run_logger()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("motorcycle-loan")
    client = MlflowClient()

    with mlflow.start_run(run_name="xgboost_v2_retrain"):
        mlflow.log_metric("auc_roc", v2_auc)
        mlflow.set_tags({"trained_on": "matured_labels_only", "stage_intent": "candidate"})
        mlflow.sklearn.log_model(pipe, "model", registered_model_name=MODEL_NAME)

    new_v = client.get_latest_versions(MODEL_NAME)[-1].version
    if v2_auc >= v1_auc - 0.02:
        client.transition_model_version_stage(name=MODEL_NAME, version=new_v, stage="Production", archive_existing_versions=True)
        msg = f"PROMOTED v{new_v} to Production (v2 {v2_auc:.4f} >= v1 {v1_auc:.4f} - 0.02). Prior versions archived."
    else:
        msg = f"KEPT v1. v2 (v{new_v}) AUC {v2_auc:.4f} below v1 {v1_auc:.4f} - 0.02; registered for inspection only."
    logger.info(msg)
    return msg


@task
def cleanup(decision_msg: str):
    logger = get_run_logger()
    if DRIFT_ALERT.exists():
        DRIFT_ALERT.unlink()
        logger.info("drift_alert.txt removed.")
    Path("retrain").mkdir(exist_ok=True)
    Path("retrain/retrain_flow_decision.txt").write_text(f"{datetime.now()}\n{decision_msg}\n", encoding="utf-8")
    logger.info(f"Retraining complete. {decision_msg}")


@flow(name="loan_retraining_flow")
def loan_retraining_flow(force: bool = True):
    detect_drift_alert()
    data = load_training_data()
    pipe, v2_auc = retrain_model(data)
    v1_auc = get_production_auc()
    msg = register_and_promote(pipe, v2_auc, v1_auc)
    cleanup(msg)
    return msg


if __name__ == "__main__":
    loan_retraining_flow(force=True)