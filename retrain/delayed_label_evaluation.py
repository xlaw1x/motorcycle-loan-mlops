from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PREDICTIONS_LOG_PATH = Path("data/processed/predictions_log.csv")
MATURED_OUTPUT_PATH = Path("data/processed/predictions_log_with_matured_labels.csv")

RETRAIN_DIR = Path("retrain")
PERFORMANCE_REPORT_PATH = RETRAIN_DIR / "delayed_performance_report.csv"
DECISION_REPORT_PATH = RETRAIN_DIR / "retraining_decision.txt"
EVAL_ROWS_PATH = RETRAIN_DIR / "matured_prediction_eval.csv"

SIMULATED_FUTURE_DATE = date(2026, 10, 15)
RANDOM_SEED = 42

AUC_RETRAIN_THRESHOLD = 0.65
F1_RETRAIN_THRESHOLD = 0.45
RECALL_RETRAIN_THRESHOLD = 0.50


def load_prediction_log():
    if not PREDICTIONS_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {PREDICTIONS_LOG_PATH}. "
            "Run Step 10 and Step 12 first."
        )

    df = pd.read_csv(PREDICTIONS_LOG_PATH)

    if df.empty:
        raise ValueError("Prediction log is empty.")

    return df


def simulate_matured_labels(df):
    """
    This simulates a future date where some previously pending loan labels
    have now matured.

    In real life, these values would come from the loan servicing system:
    days_past_due and actual default outcome.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    matured_df = df.copy()
    matured_df["expected_label_date"] = pd.to_datetime(
        matured_df["expected_label_date"],
        errors="coerce",
    )

    future_timestamp = pd.Timestamp(SIMULATED_FUTURE_DATE)
    is_mature_now = matured_df["expected_label_date"] <= future_timestamp

    matured_df["actual_label_arrival_date"] = ""
    matured_df.loc[is_mature_now, "actual_label_arrival_date"] = SIMULATED_FUTURE_DATE.isoformat()

    probabilities = matured_df["default_probability"].astype(float).clip(0.01, 0.99)

    actual_targets = []
    actual_dpd = []

    for probability, mature in zip(probabilities, is_mature_now):
        if not mature:
            actual_targets.append(np.nan)
            actual_dpd.append(np.nan)
            continue

        adjusted_probability = min(max(probability * 1.10, 0.01), 0.95)
        target = rng.binomial(1, adjusted_probability)

        if target == 1:
            dpd = int(rng.integers(90, 181))
        else:
            dpd = int(rng.integers(0, 60))

        actual_targets.append(target)
        actual_dpd.append(dpd)

    matured_df["actual_TARGET"] = actual_targets
    matured_df["actual_days_past_due"] = actual_dpd
    matured_df["label_is_mature_at_evaluation"] = is_mature_now

    matured_df.to_csv(MATURED_OUTPUT_PATH, index=False)

    return matured_df


def evaluate_matured_predictions(matured_df):
    eval_df = matured_df[matured_df["label_is_mature_at_evaluation"] == True].copy()

    if eval_df.empty:
        raise ValueError("No matured labels available for evaluation.")

    eval_df = eval_df.dropna(subset=["actual_TARGET", "default_probability"]).copy()

    y_true = eval_df["actual_TARGET"].astype(int)
    y_proba = eval_df["default_probability"].astype(float)
    y_pred = (y_proba >= 0.35).astype(int)

    auc = roc_auc_score(y_true, y_proba) if y_true.nunique() > 1 else np.nan
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    default_rate = y_true.mean()

    report = {
        "evaluation_date": SIMULATED_FUTURE_DATE.isoformat(),
        "matured_rows_evaluated": len(eval_df),
        "actual_default_rate": round(default_rate, 4),
        "auc_roc": round(float(auc), 4) if not pd.isna(auc) else np.nan,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "true_negative": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        "false_positive": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        "false_negative": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        "true_positive": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
    }

    return eval_df, pd.DataFrame([report])


def make_retraining_decision(report_df):
    row = report_df.iloc[0]

    reasons = []
    retrain_needed = False

    auc = row["auc_roc"]
    f1 = row["f1"]
    recall = row["recall"]

    if not pd.isna(auc) and auc < AUC_RETRAIN_THRESHOLD:
        retrain_needed = True
        reasons.append(f"AUC dropped below threshold: {auc} < {AUC_RETRAIN_THRESHOLD}")

    if f1 < F1_RETRAIN_THRESHOLD:
        retrain_needed = True
        reasons.append(f"F1 dropped below threshold: {f1} < {F1_RETRAIN_THRESHOLD}")

    if recall < RECALL_RETRAIN_THRESHOLD:
        retrain_needed = True
        reasons.append(f"Recall dropped below threshold: {recall} < {RECALL_RETRAIN_THRESHOLD}")

    if row["matured_rows_evaluated"] < 30:
        reasons.append(
            "Warning: fewer than 30 matured rows, decision should be treated as preliminary."
        )

    if retrain_needed:
        decision = "RETRAINING RECOMMENDED"
    else:
        decision = "NO RETRAINING REQUIRED YET"

    return decision, reasons


def save_reports(eval_df, report_df, decision, reasons):
    RETRAIN_DIR.mkdir(parents=True, exist_ok=True)

    eval_df.to_csv(EVAL_ROWS_PATH, index=False)
    report_df.to_csv(PERFORMANCE_REPORT_PATH, index=False)

    with open(DECISION_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("MOTO² Retraining Decision Report\n")
        file.write("======================================\n\n")
        file.write(f"Evaluation date: {SIMULATED_FUTURE_DATE.isoformat()}\n")
        file.write(f"Decision: {decision}\n\n")

        file.write("Performance summary:\n")
        for column, value in report_df.iloc[0].items():
            file.write(f"- {column}: {value}\n")

        file.write("\nReasons:\n")
        if reasons:
            for reason in reasons:
                file.write(f"- {reason}\n")
        else:
            file.write("- Model performance is still within acceptable thresholds.\n")

        file.write("\nLabel latency explanation:\n")
        file.write(
            "The model could not be fully evaluated at prediction time because actual loan "
            "outcomes were still pending. After labels matured, delayed performance was "
            "computed and used for the retraining decision.\n"
        )


def main():
    prediction_log = load_prediction_log()
    matured_df = simulate_matured_labels(prediction_log)
    eval_df, report_df = evaluate_matured_predictions(matured_df)

    decision, reasons = make_retraining_decision(report_df)
    save_reports(eval_df, report_df, decision, reasons)

    print("Delayed label evaluation complete.")
    print(f"Matured prediction log saved to: {MATURED_OUTPUT_PATH}")
    print(f"Evaluation rows saved to: {EVAL_ROWS_PATH}")
    print(f"Performance report saved to: {PERFORMANCE_REPORT_PATH}")
    print(f"Retraining decision saved to: {DECISION_REPORT_PATH}")
    print()
    print("Performance report:")
    print(report_df.to_string(index=False))
    print()
    print("Retraining decision:")
    print(decision)

    if reasons:
        print()
        print("Reasons:")
        for reason in reasons:
            print(f"- {reason}")


if __name__ == "__main__":
    main()
