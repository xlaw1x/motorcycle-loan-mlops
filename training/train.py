import json
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


RANDOM_SEED = 42

DATA_PATH = Path("data/raw/ph_motorcycle_loans.csv")
ARTIFACT_DIR = Path("training/artifacts")

MODEL_PATH = ARTIFACT_DIR / "model.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURE_LIST_PATH = ARTIFACT_DIR / "feature_list.json"
CONFUSION_MATRIX_PATH = ARTIFACT_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.png"
TEST_PREDICTIONS_PATH = ARTIFACT_DIR / "test_predictions.csv"

REGISTERED_MODEL_NAME = "motorcycle-loan-model"

TARGET_COLUMN = "TARGET"

LEAKAGE_COLUMNS = [
    "days_past_due",
    "label_determination_date",
    "loan_origination_date",
    "TARGET",
    "label_is_mature",
]

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

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {DATA_PATH}")
    print(f"Original shape: {df.shape}")

    return df


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Label-latency rule:
    We train only on rows where the true outcome is already observable.
    """
    if "label_is_mature" not in df.columns:
        raise ValueError("Missing required column: label_is_mature")

    mature_mask = df["label_is_mature"].astype(str).str.lower() == "true"
    mature_df = df[mature_mask].copy()
    dropped_rows = len(df) - len(mature_df)

    print(f"Mature rows used for training: {len(mature_df):,}")
    print(f"Rows dropped as label-pending: {dropped_rows:,}")

    if len(mature_df) == 0:
        raise ValueError("No matured rows available for training.")

    mature_df["loan_to_income_ratio"] = (
        mature_df["loan_asset_cost"] / mature_df["total_income"].replace(0, np.nan)
    )
    mature_df["loan_to_income_ratio"] = mature_df["loan_to_income_ratio"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    mature_df["loan_to_income_ratio"] = mature_df["loan_to_income_ratio"].fillna(
        mature_df["loan_to_income_ratio"].median()
    )

    missing_features = [col for col in MODEL_FEATURES if col not in mature_df.columns]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    leaked_features = [col for col in MODEL_FEATURES if col in LEAKAGE_COLUMNS]
    if leaked_features:
        raise ValueError(f"Leakage columns found in model features: {leaked_features}")

    X = mature_df[MODEL_FEATURES].copy()
    y = mature_df[TARGET_COLUMN].astype(int)

    print(f"Training feature shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True).rename('rate')}")
    print("Leakage guard passed. No forbidden columns are in MODEL_FEATURES.")

    return X, y


def build_model(y_train: pd.Series) -> Tuple[str, Pipeline, float]:
    """
    Try XGBoost first.
    If unavailable, use RandomForest as a local fallback.
    Returns model_name, pipeline, and the scale_pos_weight used.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ]
    )

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    try:
        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model_name = "xgboost"

    except Exception as exc:
        print("XGBoost could not be used. Falling back to RandomForest.")
        print(f"Reason: {exc}")

        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model_name = "random_forest"

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return model_name, pipeline, scale_pos_weight


def get_transformed_feature_names(model: Pipeline) -> List[str]:
    preprocessor = model.named_steps["preprocessor"]

    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return MODEL_FEATURES


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.35).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "test_rows": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
    }

    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm, y_proba


def save_confusion_matrix_plot(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-default", "Default"])
    ax.set_yticklabels(["Non-default", "Default"])

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, cm[row, col], ha="center", va="center")

    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)


def save_feature_importance_plot(model: Pipeline) -> None:
    classifier = model.named_steps["classifier"]
    feature_names = get_transformed_feature_names(model)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    else:
        print("Classifier does not expose feature_importances_. Skipping plot.")
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    top_n = importance_df.head(20).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_n["feature"], top_n["importance"])
    ax.set_title("Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close(fig)


def save_artifacts(
    model: Pipeline,
    model_name: str,
    metrics: dict,
    cm: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_proba: np.ndarray,
) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_name": model_name,
        "pipeline": model,
        "model_features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "leakage_columns": LEAKAGE_COLUMNS,
        "threshold": 0.35,
        "trained_on": "matured_labels_only",
        "random_seed": RANDOM_SEED,
    }

    joblib.dump(artifact, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "model_features": MODEL_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "leakage_columns": LEAKAGE_COLUMNS,
            },
            file,
            indent=2,
        )

    save_confusion_matrix_plot(cm)
    save_feature_importance_plot(model)

    predictions_df = X_test.copy()
    predictions_df["actual_TARGET"] = y_test.values
    predictions_df["default_probability"] = y_proba
    predictions_df["predicted_TARGET"] = (y_proba >= 0.35).astype(int)
    predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    return artifact


def register_to_mlflow(
    artifact: dict,
    metrics: dict,
    scale_pos_weight: float,
) -> None:
    """
    Log params/metrics/tags, register the pipeline to the MLflow Model
    Registry, and promote the new version to Production.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("motorcycle-loan")

    with mlflow.start_run(run_name="xgboost_v1_baseline"):
        mlflow.log_params(
            {
                "model_type": artifact["model_name"],
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "scale_pos_weight": round(scale_pos_weight, 3),
                "threshold": 0.35,
            }
        )
        mlflow.log_metrics(
            {
                "auc_roc": metrics["auc_roc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
            }
        )
        mlflow.set_tags(
            {
                "trained_on": "matured_labels_only",
                "dataset": "ph_synthetic_v1",
            }
        )
        mlflow.sklearn.log_model(
            sk_model=artifact["pipeline"],
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    client = MlflowClient()
    latest = client.get_latest_versions(REGISTERED_MODEL_NAME)[-1]
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=False,
    )
    print(
        f"Registered & promoted {REGISTERED_MODEL_NAME} "
        f"v{latest.version} to Production"
    )


def main() -> None:
    df = load_data()
    X, y = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")

    model_name, model, scale_pos_weight = build_model(y_train)
    print(f"Training model: {model_name}")

    model.fit(X_train, y_train)

    metrics, cm, y_proba = evaluate_model(model, X_test, y_test)

    print("\nModel metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")

    print("\nConfusion matrix:")
    print(cm)

    artifact = save_artifacts(
        model=model,
        model_name=model_name,
        metrics=metrics,
        cm=cm,
        X_test=X_test,
        y_test=y_test,
        y_proba=y_proba,
    )

    print("\nArtifacts saved:")
    print(f"- {MODEL_PATH}")
    print(f"- {METRICS_PATH}")
    print(f"- {FEATURE_LIST_PATH}")
    print(f"- {CONFUSION_MATRIX_PATH}")
    print(f"- {FEATURE_IMPORTANCE_PATH}")
    print(f"- {TEST_PREDICTIONS_PATH}")

    register_to_mlflow(
        artifact=artifact,
        metrics=metrics,
        scale_pos_weight=scale_pos_weight,
    )

    print("\nStep 4 complete. Model trained using matured labels only.")


if __name__ == "__main__":
    main()