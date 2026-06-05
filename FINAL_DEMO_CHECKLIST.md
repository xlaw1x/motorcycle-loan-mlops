# MOTO² Final Demo Checklist

## Environment

- [ ] Virtual environment is activated.
- [ ] Dependencies are installed from `requirements-dev.txt`.
- [ ] Commands are run from the project root.
- [ ] `training/artifacts/model.pkl` exists.
- [ ] `data/raw/ph_motorcycle_loans.csv` exists.

## Validation and Training

- [ ] Run Great Expectations dataset validation:

```powershell
.\.venv\Scripts\python.exe validation/expectations.py
```

- [ ] Confirm `validation/latest_validation_report.txt` is generated.
- [ ] Run model training if artifacts need to be refreshed:

```powershell
.\.venv\Scripts\python.exe training/train.py
```

- [ ] Confirm model metrics and artifacts exist in `training/artifacts/`.

## API

- [ ] Start FastAPI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

- [ ] Open `http://127.0.0.1:8000/docs`.
- [ ] Confirm `GET /health` returns `status: ok`.
- [ ] Confirm the API name is `MOTO² API`.

## Streamlit

- [ ] Start Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

- [ ] Open `http://localhost:8501`.
- [ ] Confirm the app title is `MOTO²`.
- [ ] Submit a sample application.
- [ ] Confirm the prediction result appears.
- [ ] Confirm label latency is shown as pending.

## Logging

- [ ] Confirm `data/processed/predictions_log.csv` exists.
- [ ] Confirm the latest prediction row includes pending actual label fields.
- [ ] Confirm `expected_label_date` is populated.

## Monitoring

- [ ] Simulate drift:

```powershell
.\.venv\Scripts\python.exe monitoring/simulate_drift.py
```

- [ ] Generate monitoring reports:

```powershell
.\.venv\Scripts\python.exe monitoring/monitor_predictions.py
```

- [ ] Confirm `monitoring/reports/monitoring_report.html` exists.
- [ ] Confirm numeric and categorical drift reports exist.
- [ ] If Docker is running, open `http://localhost:8080/evidently_report.html`.

## Delayed Labels and Retraining Decision

- [ ] Run delayed-label evaluation:

```powershell
.\.venv\Scripts\python.exe retrain/delayed_label_evaluation.py
```

- [ ] Confirm `retrain/delayed_performance_report.csv` exists.
- [ ] Confirm `retrain/retraining_decision.txt` exists.

## Final Checks

- [ ] Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

- [ ] Run final QA:

```powershell
.\.venv\Scripts\python.exe qa/final_qa_check.py
```

- [ ] Confirm final QA status is `DEMO READY`.

## Docker Services

- [ ] Run Docker services:

```powershell
docker compose up --build
```

- [ ] Confirm FastAPI opens at `http://localhost:8000`.
- [ ] Confirm Streamlit opens at `http://localhost:8501`.
- [ ] Confirm MLflow opens at `http://localhost:5000`.
- [ ] Confirm Evidently report opens at `http://localhost:8080/evidently_report.html`.
- [ ] Confirm Prefect opens at `http://localhost:4200`.
- [ ] Run `docker compose run --rm prefect-flow` and confirm a flow run appears in Prefect.
