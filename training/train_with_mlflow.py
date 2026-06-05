from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from train import (
    ARTIFACT_DIR,
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    build_model,
    evaluate_model,
    load_data,
    prepare_training_data,
    save_artifacts,
)


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "moto-loan-ph"
RUN_NAME = "xgboost_matured_labels_v1"

TEST_SIZE = 0.20
THRESHOLD = 0.35


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_data()

    mature_mask = df["label_is_mature"].astype(str).str.lower() == "true"
    mature_rows = int(mature_mask.sum())
    pending_rows = int((~mature_mask).sum())

    X, y = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model_name, model = build_model(y_train)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        print(f"MLflow run started: {run.info.run_id}")
        print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"Experiment: {MLFLOW_EXPERIMENT_NAME}")

        mlflow.set_tags(
            {
                "project": "MOTO²",
                "problem_type": "motorcycle_loan_default_prediction",
                "label_definition": "D3 default, 90+ days past due",
                "training_rule": "matured_labels_only",
                "leakage_guard": "enabled",
            }
        )

        mlflow.log_params(
            {
                "model_name": model_name,
                "random_seed": RANDOM_SEED,
                "test_size": TEST_SIZE,
                "threshold": THRESHOLD,
                "mature_rows": mature_rows,
                "pending_rows_excluded": pending_rows,
                "feature_count": len(MODEL_FEATURES),
                "numeric_feature_count": len(NUMERIC_FEATURES),
                "categorical_feature_count": len(CATEGORICAL_FEATURES),
                "leakage_columns_blocked": ", ".join(LEAKAGE_COLUMNS),
            }
        )

        print(f"Training model: {model_name}")
        model.fit(X_train, y_train)

        metrics, cm, y_proba = evaluate_model(model, X_test, y_test)

        mlflow.log_metrics(metrics)
        mlflow.log_metric("train_rows", len(X_train))
        mlflow.log_metric("test_rows", len(X_test))
        mlflow.log_metric("mature_rows", mature_rows)
        mlflow.log_metric("pending_rows_excluded", pending_rows)

        save_artifacts(
            model=model,
            model_name=model_name,
            metrics=metrics,
            cm=cm,
            X_test=X_test,
            y_test=y_test,
            y_proba=y_proba,
        )

        mlflow.log_artifacts(str(ARTIFACT_DIR), artifact_path="training_artifacts")

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
            )
        except Exception as exc:
            print(f"Warning: MLflow model logging failed, but artifacts were logged: {exc}")

        print("\nModel metrics:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                print(f"{metric_name}: {metric_value:.4f}")
            else:
                print(f"{metric_name}: {metric_value}")

        print("\nArtifacts logged to MLflow.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Artifact URI: {run.info.artifact_uri}")

    print("\nStep 5 complete. Training run tracked with MLflow.")


if __name__ == "__main__":
    main()
