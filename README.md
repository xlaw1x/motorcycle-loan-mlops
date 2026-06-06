# MOTO²

MOTO² is an end-to-end MLOps project for Philippine motorcycle loan default
prediction. It demonstrates how a model can be trained, validated, deployed,
monitored, and evaluated when true loan outcomes arrive only after a delay.

The project includes:

- a machine learning training pipeline
- a FastAPI prediction service
- a customer-facing Streamlit application
- Great Expectations data validation
- MLflow experiment tracking
- Prefect batch scoring
- prediction logging
- custom drift monitoring
- Evidently data drift and data quality reporting
- delayed-label performance evaluation
- Docker Compose services for demo deployment
- pytest and final QA checks

## Business Problem

Motorcycle lenders need to decide whether a loan application is likely to
default. In this project, default follows a D3 rule:

```text
TARGET = 1 when days_past_due >= 90
TARGET = 0 when days_past_due < 90
```

The important MLOps challenge is label latency. At prediction time, the true
default label is not yet known. The borrower must first go through the repayment
window before the lender can observe whether the loan becomes 90+ days past due.

Because of this, MOTO² separates:

- application-time features, which are safe to use for scoring
- future outcome fields, which must never be used as model inputs
- pending production predictions, which can only be evaluated later

## Dataset Source

The dataset used in this project comes from Kaggle:

```text
PH Motorcycle Loan Default Dataset (Synthetic)
https://www.kaggle.com/datasets/patpascual/ph-motorcycle-loan-default-dataset-synthetic/data
```

The dataset is described as synthetic two-wheeler loan origination data based on
a real Southeast Asian bank. It is used here for educational MLOps workflow
demonstration, including validation, model training, scoring, monitoring, and
delayed-label evaluation.

## Architecture

```text
Raw loan data
    |
    v
Great Expectations validation
    |
    v
Matured-label training
    |
    v
Model artifact + metrics + MLflow tracking
    |
    +----------------------------+
    |                            |
    v                            v
FastAPI real-time scoring        Prefect batch scoring
    |
    v
Streamlit application
    |
    v
Prediction log with pending labels
    |
    +----------------------------+
    |                            |
    v                            v
Custom drift reports             Evidently report
    |
    v
Delayed-label evaluation and retraining decision
```

## Tooling Summary

| Area | Tooling | Purpose |
| --- | --- | --- |
| API | FastAPI | Real-time prediction endpoint |
| Frontend | Streamlit | Customer-facing scoring UI |
| Model | scikit-learn, XGBoost | Default prediction model |
| Tracking | MLflow | Experiment tracking and model artifacts |
| Validation | Great Expectations | Dataset schema and quality checks |
| Workflow | Prefect | Batch scoring flow |
| Monitoring | Custom PSI reports | Numeric and categorical drift checks |
| Monitoring | Evidently | Data drift and data quality report |
| Testing | pytest | Project readiness tests |
| Orchestration | Prefect | Batch scoring flow and workflow UI |
| Deployment | Docker Compose | API, frontend, MLflow, Evidently, Prefect services |

## Project Structure

```text
motorcycle-loan-mlops/
|-- api/
|   |-- main.py                         FastAPI prediction service
|   `-- requirements.txt
|-- frontend/
|   |-- app.py                          Streamlit application
|   `-- requirements.txt
|-- training/
|   |-- train.py                        Main model training script
|   |-- train_with_mlflow.py             MLflow-tracked training script
|   |-- requirements.txt
|   `-- artifacts/                       Saved model, metrics, and plots
|-- validation/
|   |-- expectations.py                  Great Expectations validation runner
|   `-- latest_validation_report.txt
|-- pipeline/
|   `-- prefect_flow.py                  Batch scoring flow
|-- monitoring/
|   |-- simulate_drift.py                Creates drifted prediction examples
|   |-- monitor_predictions.py           Custom PSI monitoring report
|   |-- evidently_report.py              Evidently HTML report generator
|   `-- reports/
|-- retrain/
|   |-- delayed_label_evaluation.py      Delayed-label evaluation script
|   |-- delayed_performance_report.csv
|   |-- matured_prediction_eval.csv
|   `-- retraining_decision.txt
|-- qa/
|   `-- final_qa_check.py                Final demo readiness check
|-- tests/
|   `-- test_project_readiness.py
|-- data/
|   |-- raw/ph_motorcycle_loans.csv
|   `-- processed/
|-- Dockerfile.api
|-- Dockerfile.frontend
|-- Dockerfile.mlflow
|-- Dockerfile.evidently
|-- docker-compose.yml
|-- requirements-dev.txt
|-- requirements-api.txt
|-- requirements-frontend.txt
|-- requirements-mlflow.txt
|-- requirements-monitoring.txt
`-- README.md
```

## Prerequisites

Use Python 3.10 for local development.

Recommended local tools:

- Python 3.10
- Docker Desktop
- PowerShell
- Git

The commands below assume you are inside:

```text
motorcycle-loan-mlops/
```

## Local Environment Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Check the environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Use `.\.venv\Scripts\python.exe -m pytest -q` instead of bare `pytest -q` so
the tests run against the project virtual environment.

## Data and Pipeline Versioning with DVC

DVC is configured to make the main project workflow reproducible. The project
includes:

```text
.dvc/config
.dvcignore
dvc.yaml
```

The DVC pipeline defines these stages:

| Stage | Command | Purpose |
| --- | --- | --- |
| `validate` | `python validation/expectations.py` | Run Great Expectations validation |
| `train` | `python training/train.py` | Train the model and save artifacts |
| `batch_score` | `python pipeline/prefect_flow.py` | Run batch scoring |
| `monitor` | `python monitoring/monitor_predictions.py` | Generate custom monitoring reports |
| `delayed_evaluation` | `python retrain/delayed_label_evaluation.py` | Evaluate delayed labels and retraining decision |

Run the full DVC pipeline:

```powershell
.\.venv\Scripts\python.exe -m dvc repro
```

Run one stage:

```powershell
.\.venv\Scripts\python.exe -m dvc repro train
```

Check pipeline status:

```powershell
.\.venv\Scripts\python.exe -m dvc status
```

This setup uses DVC for pipeline reproducibility. If you want full external
data versioning later, add a DVC remote and run `dvc add` for large datasets or
artifacts that should be stored outside Git.

## Data Validation with Great Expectations

Run dataset validation:

```powershell
.\.venv\Scripts\python.exe validation/expectations.py
```

This validates `data/raw/ph_motorcycle_loans.csv` with a Great Expectations
suite and writes:

```text
validation/latest_validation_report.txt
```

The validation checks include:

- exact required column order
- row count
- non-null required fields
- unique `loan_account_no`
- numeric ranges for age, income, asset cost, interest rate, dependents, and DPD
- allowed categorical values
- valid bureau score logic, including `-1` for no bureau history
- D3 target rule consistency
- date order consistency
- label maturity consistency
- leakage column reminder

The CI workflow also runs this validation step.

## Model Training

Run standard training:

```powershell
.\.venv\Scripts\python.exe training/train.py
```

The training script:

- loads `data/raw/ph_motorcycle_loans.csv`
- keeps only rows where `label_is_mature == true`
- excludes future-looking leakage fields
- creates `loan_to_income_ratio`
- trains XGBoost if available
- falls back to RandomForest if XGBoost is unavailable
- saves metrics, plots, predictions, and the model artifact

Generated artifacts:

```text
training/artifacts/model.pkl
training/artifacts/metrics.json
training/artifacts/feature_list.json
training/artifacts/confusion_matrix.png
training/artifacts/feature_importance.png
training/artifacts/test_predictions.csv
```

The API requires `training/artifacts/model.pkl` to exist before startup.

## MLflow Tracking

Run MLflow-tracked training:

```powershell
.\.venv\Scripts\python.exe training/train_with_mlflow.py
```

This logs:

- project tags
- model parameters
- training and test row counts
- feature counts
- leakage guard metadata
- model metrics
- training artifacts
- a registered model named `moto2-motorcycle-loan-default`

The local tracking backend is:

```text
sqlite:///mlflow.db
```

To view MLflow in Docker:

```powershell
docker compose up --build mlflow
```

Open:

```text
http://localhost:5000
```

The Docker MLflow service mounts:

```text
./mlflow.db  -> /mlflow/mlflow.db
./mlruns     -> /mlflow/mlruns
```

If the MLflow UI shows the Registry page but no registered models, run:

```powershell
.\.venv\Scripts\python.exe training/train_with_mlflow.py
```

Then refresh:

```text
http://localhost:5000/#/models
```

The registry is not populated by `training/train.py`; it is populated by
`training/train_with_mlflow.py`.

## FastAPI Prediction Service

Run the API locally:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

Main endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Basic service metadata |
| `GET /health` | Model and feature health check |
| `POST /predict` | Validate and score one loan application |

The prediction endpoint returns:

- applicant name
- decision
- default probability
- approval confidence
- top reasons
- risk factors
- model version
- label status
- expected label date
- timestamp

Every successful prediction is appended to:

```text
data/processed/predictions_log.csv
```

The log intentionally leaves actual label fields blank at prediction time.

## Streamlit Application

Run the customer-facing app locally:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit app is named `MOTO²`. It provides:

- borrower profile inputs
- employment and income inputs
- loan and motorcycle details
- backend health status
- default probability
- decision output
- top reasons and risk factors
- label latency explanation
- raw API response expander

By default, the app calls:

```text
http://127.0.0.1:8000
```

Override the backend URL:

```powershell
$env:API_URL="http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

## Batch Scoring with Prefect

Run the batch scoring flow:

```powershell
.\.venv\Scripts\python.exe pipeline/prefect_flow.py
```

The flow:

- loads the trained model artifact
- selects label-pending applications
- prepares scoring features
- scores default probability
- assigns a decision
- writes latest and timestamped output files

Outputs:

```text
data/processed/batch_scoring_output.csv
data/processed/batch_scoring_output_<timestamp>.csv
```

## Prediction Logging

Real-time predictions are saved to:

```text
data/processed/predictions_log.csv
```

Important columns include:

- `prediction_id`
- `prediction_timestamp`
- `applicant_name`
- `model_version`
- `decision`
- `default_probability`
- `approval_confidence`
- `risk_factors`
- `label_status`
- `expected_label_date`
- `actual_TARGET`
- `actual_days_past_due`
- `actual_label_arrival_date`

The `actual_*` fields start blank because labels are not observable at scoring
time.

## Drift Simulation

Run drift simulation:

```powershell
.\.venv\Scripts\python.exe monitoring/simulate_drift.py
```

This creates riskier synthetic production applications by shifting features such
as:

- lower income
- lower bureau score
- higher asset cost
- more dependents
- more existing loans
- more rented residences
- more online applications

Outputs include:

```text
data/processed/drift_simulated_predictions.csv
data/processed/predictions_log_backup_before_drift.csv
data/processed/predictions_log.csv
```

## Custom Monitoring Reports

Run custom monitoring:

```powershell
.\.venv\Scripts\python.exe monitoring/monitor_predictions.py
```

This compares matured training reference data against logged predictions.

Outputs:

```text
monitoring/reports/monitoring_summary.csv
monitoring/reports/numeric_drift_report.csv
monitoring/reports/categorical_drift_report.csv
monitoring/reports/decision_monitoring.csv
monitoring/reports/monitoring_report.html
```

The custom monitoring report includes:

- average and max default probability
- pending-label share
- numeric PSI values
- categorical PSI values
- decision distribution
- drift status labels

## Evidently Monitoring

Generate and serve the Evidently report with Docker:

```powershell
docker compose up --build evidently
```

Open:

```text
http://localhost:8080/evidently_report.html
```

The Evidently service:

- loads mature reference rows from the raw dataset
- loads current rows from the prediction log
- compares numeric and categorical feature distributions
- writes `monitoring/reports/evidently_report.html`
- serves the report from port `8080`

If the prediction log is missing, the service writes a placeholder HTML report
explaining what data is missing.

## Delayed-Label Evaluation

Run delayed-label evaluation:

```powershell
.\.venv\Scripts\python.exe retrain/delayed_label_evaluation.py
```

This simulates a future date where some pending labels have matured. It fills in
actual outcomes, evaluates model performance, and writes a retraining decision.

Outputs:

```text
data/processed/predictions_log_with_matured_labels.csv
retrain/matured_prediction_eval.csv
retrain/delayed_performance_report.csv
retrain/retraining_decision.txt
```

Evaluation metrics include:

- AUC ROC
- precision
- recall
- F1
- accuracy
- confusion matrix counts
- actual default rate

Retraining is recommended if delayed performance drops below configured
thresholds.

## Docker Compose Services

Run everything:

```powershell
docker compose up --build
```

Services:

| Service | Port | URL | Description |
| --- | --- | --- | --- |
| `api` | `8000` | `http://localhost:8000` | FastAPI scoring service |
| `frontend` | `8501` | `http://localhost:8501` | Streamlit MOTO² app |
| `mlflow` | `5000` | `http://localhost:5000` | MLflow tracking UI |
| `evidently` | `8080` | `http://localhost:8080/evidently_report.html` | Evidently report server |
| `prefect` | `4200` | `http://localhost:4200` | Prefect orchestration UI |

Stop all services:

```powershell
docker compose down
```

Run only MLflow and Evidently:

```powershell
docker compose up --build mlflow evidently
```

Run only the Prefect server:

```powershell
docker compose up --build prefect
```

Open:

```text
http://localhost:4200
```

Run the batch scoring flow and record it in the Docker Prefect UI:

```powershell
docker compose run --rm prefect-flow
```

## Tests

Run pytest:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests check:

- raw dataset exists and has expected columns
- target follows the D3 rule
- model artifact exists and has expected keys
- model features do not include leakage columns
- FastAPI health endpoint works
- FastAPI prediction endpoint returns expected fields
- invalid applications are rejected
- prediction logging works

## Final QA

Run final readiness checks:

```powershell
.\.venv\Scripts\python.exe qa/final_qa_check.py
```

The QA script checks:

- dataset shape and target rule
- label maturity presence
- model artifact keys
- leakage guard
- training metrics
- prediction log
- monitoring outputs
- Evidently HTML report
- delayed-label reports
- required Docker files
- required demo files
- tests file presence

Expected final status:

```text
FINAL STATUS: DEMO READY
```

## CI

The GitHub Actions workflow is in:

```text
.github/workflows/ci.yml
```

It installs `requirements-dev.txt`, runs pytest, and runs the Great Expectations
validation script.

## Suggested Demo Flow

1. Show the business problem and label-latency constraint.
2. Run Great Expectations validation.
3. Run model training.
4. Run MLflow-tracked training and open MLflow.
5. Start Docker services.
6. Open Streamlit and score a sample application.
7. Show the FastAPI docs.
8. Show the prediction log.
9. Run drift simulation.
10. Generate custom monitoring reports.
11. Open the Evidently report.
12. Run delayed-label evaluation.
13. Run pytest.
14. Run final QA and show `DEMO READY`.

## Common Commands

```powershell
# Validate data
.\.venv\Scripts\python.exe validation/expectations.py

# Train model
.\.venv\Scripts\python.exe training/train.py

# Train with MLflow
.\.venv\Scripts\python.exe training/train_with_mlflow.py

# Run API
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Run Streamlit
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py

# Run batch scoring
.\.venv\Scripts\python.exe pipeline/prefect_flow.py

# Simulate drift
.\.venv\Scripts\python.exe monitoring/simulate_drift.py

# Generate custom monitoring report
.\.venv\Scripts\python.exe monitoring/monitor_predictions.py

# Run delayed-label evaluation
.\.venv\Scripts\python.exe retrain/delayed_label_evaluation.py

# Run tests
.\.venv\Scripts\python.exe -m pytest -q

# Run final QA
.\.venv\Scripts\python.exe qa/final_qa_check.py

# Run all Docker services
docker compose up --build
```

## Important Notes

- Do not train on `TARGET`, `days_past_due`, `label_determination_date`,
  `loan_origination_date`, or `label_is_mature`.
- The API must be restarted after replacing `training/artifacts/model.pkl`.
- Streamlit requires the FastAPI backend to be reachable.
- Evidently requires `data/processed/predictions_log.csv` for a real report.
- MLflow Docker reads the host `mlflow.db` and `mlruns/` directories.
- Bare `pytest -q` may use the wrong Python environment. Prefer:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
## MLflow Artifact Troubleshooting

If a run appears in MLflow but artifacts such as `confusion_matrix.png` do not
show, the run was probably logged with a local file artifact URI instead of the
Docker MLflow server URI.

Use this sequence for demo runs:

```powershell
docker compose up -d mlflow
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:MLFLOW_EXPERIMENT_NAME="moto2-loan-ph"
.\.venv\Scripts\python.exe training\train_with_mlflow.py
docker compose restart mlflow
```

Then open:

```text
http://localhost:5000
```

Click the latest run and check:

```text
Artifacts -> training_artifacts -> confusion_matrix.png
```

The local-only fallback still works:

```powershell
.\.venv\Scripts\python.exe training\train_with_mlflow.py
```

But local-only runs may not show artifacts correctly in the Docker MLflow UI if
their artifact URI points to a Windows path.
