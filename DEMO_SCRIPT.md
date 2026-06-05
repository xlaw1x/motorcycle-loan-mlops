# MOTO² Demo Script

Use this script to walk through the project in a short final demo.

## 1. Project Overview

MOTO² is an MLOps workflow for Philippine motorcycle loan default prediction.
It demonstrates real-time scoring, label latency, prediction logging, drift
monitoring, delayed-label evaluation, and retraining decision support.

Key point: the model scores applications now, while the actual default label is
only observable after the repayment window.

## 2. Validate the Dataset with Great Expectations

```powershell
.\.venv\Scripts\python.exe validation/expectations.py
```

Show:

- required columns are present
- target follows the D3 rule: default means 90+ days past due
- leakage columns are identified and excluded from model features
- label maturity is tracked

## 3. Train the Model

```powershell
.\.venv\Scripts\python.exe training/train.py
```

Show artifacts in `training/artifacts/`:

- `model.pkl`
- `metrics.json`
- `feature_list.json`
- `confusion_matrix.png`
- `feature_importance.png`

Explain that training uses matured labels only.

## 4. Start the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

Show:

- `GET /health`
- `POST /predict`

## 5. Start the Streamlit App

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

In MOTO²:

- fill or use the default borrower profile
- click `Score Application`
- explain the decision, default probability, risk factors, and label-pending message

## 6. Confirm Prediction Logging

Show:

```text
data/processed/predictions_log.csv
```

Explain that each prediction is stored with:

- prediction ID
- model version
- decision
- probability
- input features
- pending actual label fields
- expected label date

## 7. Simulate Drift and Generate Monitoring

```powershell
.\.venv\Scripts\python.exe monitoring/simulate_drift.py
.\.venv\Scripts\python.exe monitoring/monitor_predictions.py
```

Show:

```text
monitoring/reports/monitoring_report.html
```

Explain numeric drift, categorical drift, and decision monitoring.

If running the Docker demo, also open:

```text
http://localhost:8080/evidently_report.html
```

Explain that Evidently compares the mature training reference data against the
logged production predictions.

## 8. Delayed-Label Evaluation

```powershell
.\.venv\Scripts\python.exe retrain/delayed_label_evaluation.py
```

Show:

- `retrain/delayed_performance_report.csv`
- `retrain/retraining_decision.txt`

Explain that performance is evaluated after labels mature.

## 9. Final Readiness Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe qa/final_qa_check.py
```

Close by showing all checks pass.

## 10. Docker Services

```powershell
docker compose up --build
```

Open:

- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`
- Evidently report: `http://localhost:8080/evidently_report.html`
- Prefect UI: `http://localhost:4200`

To show a Prefect flow run in the UI:

```powershell
docker compose run --rm prefect-flow
```
