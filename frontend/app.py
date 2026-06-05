from datetime import date
from typing import Any, Dict, Optional

import pandas as pd
import requests
import os
import streamlit as st


DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="MOTO²",
    layout="wide",
)


def call_health(api_url: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_predict(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{api_url}/predict",
        json=payload,
        timeout=30,
    )

    if response.status_code == 200:
        return response.json()

    try:
        error_detail = response.json()
    except Exception:
        error_detail = response.text

    raise RuntimeError(error_detail)


def show_decision_box(result: Dict[str, Any]) -> None:
    decision = result["decision"]

    if decision == "APPROVED":
        st.success("Decision: APPROVED")
    elif decision == "NEEDS_MANUAL_REVIEW":
        st.warning("Decision: NEEDS MANUAL REVIEW")
    else:
        st.error("Decision: HIGH RISK MANUAL REVIEW")


def show_prediction_result(result: Dict[str, Any]) -> None:
    show_decision_box(result)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="Default Probability",
        value=result["default_probability_percent"],
    )
    col2.metric(
        label="Approval Confidence",
        value=f"{result['approval_confidence'] * 100:.2f}%",
    )
    col3.metric(
        label="Model Version",
        value=result["model_version"],
    )

    st.subheader("Top Reasons")
    for reason in result["top_reasons"]:
        st.write(f"- {reason}")

    st.subheader("Risk Factors")
    for factor in result["risk_factors"]:
        st.write(f"- {factor}")

    st.subheader("Label Latency")
    st.info(result["label_status"])
    st.write(f"Expected label date: **{result['expected_label_date']}**")
    st.write(f"Prediction timestamp: `{result['timestamp']}`")

    with st.expander("Raw API response"):
        st.json(result)


def build_payload(
    applicant_name: str,
    bureau_score: int,
    down_payment_amount: float,
    res_years_at_current_city: int,
    res_years_at_current_address: int,
    interest_rate: float,
    loan_asset_cost: float,
    age: int,
    number_of_installments: int,
    total_income: float,
    years_in_occupation: int,
    number_of_dependents: int,
    gender: str,
    channel: str,
    marital_status: str,
    manufacturer: str,
    occupation_type: str,
    res_resident_status: str,
    has_existing_loan: str,
    loan_origination_date: date,
) -> Dict[str, Any]:
    return {
        "applicant_name": applicant_name,
        "bureau_score": bureau_score,
        "down_payment_amount": down_payment_amount,
        "res_years_at_current_city": res_years_at_current_city,
        "res_years_at_current_address": res_years_at_current_address,
        "interest_rate": interest_rate,
        "loan_asset_cost": loan_asset_cost,
        "age": age,
        "number_of_installments": number_of_installments,
        "total_income": total_income,
        "years_in_occupation": years_in_occupation,
        "number_of_dependents": number_of_dependents,
        "gender": gender,
        "channel": channel,
        "marital_status": marital_status,
        "manufacturer": manufacturer,
        "occupation_type": occupation_type,
        "res_resident_status": res_resident_status,
        "has_existing_loan": has_existing_loan,
        "loan_origination_date": loan_origination_date.isoformat(),
    }


st.title("MOTO²")
st.caption(
    "Motorcycle loan default prediction with label latency awareness. "
    "This frontend sends applications to the FastAPI scoring service."
)

with st.sidebar:
    st.header("Backend Settings")
    api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL)

    health = call_health(api_url)

    if health is None:
        st.error("FastAPI is not reachable.")
        st.write("Start the API first:")
        st.code("uvicorn api.main:app --reload --host 127.0.0.1 --port 8000")
    else:
        st.success("FastAPI connected.")
        st.write("Model:", health.get("model_name"))
        st.write("Feature count:", health.get("feature_count"))
        st.write("Validation:", health.get("application_validation", "not shown"))
        st.write("Threshold:", health.get("threshold"))

    st.divider()
    st.header("Demo Guide")
    st.write("1. Start FastAPI.")
    st.write("2. Start Streamlit.")
    st.write("3. Fill out the borrower profile.")
    st.write("4. Click Score Application.")
    st.write("5. Explain the label-pending message.")

st.subheader("Loan Application Form")

with st.form("loan_application_form"):
    st.markdown("### Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        applicant_name = st.text_input("Applicant Name", value="Mang Tony")
        age = st.number_input("Age", min_value=18, max_value=70, value=38)
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Live-in", "Widowed/Separated"],
            index=1,
        )
        number_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=3,
        )
        res_resident_status = st.selectbox(
            "Resident Status",
            ["Owned", "Rented", "Living with Parents"],
        )

    with col3:
        res_years_at_current_city = st.number_input(
            "Years at Current City",
            min_value=0,
            max_value=70,
            value=10,
        )
        res_years_at_current_address = st.number_input(
            "Years at Current Address",
            min_value=0,
            max_value=70,
            value=5,
        )
        loan_origination_date = st.date_input(
            "Loan Origination Date",
            value=date(2026, 6, 4),
        )

    st.markdown("### Employment and Income")

    col4, col5, col6 = st.columns(3)

    with col4:
        occupation_type = st.selectbox(
            "Occupation Type",
            [
                "Tricycle/Jeepney Driver",
                "Sari-sari Store Owner",
                "Construction Worker",
                "Employed",
                "OFW Dependent",
                "Other Self Employed",
            ],
        )
        years_in_occupation = st.number_input(
            "Years in Occupation",
            min_value=0,
            max_value=60,
            value=8,
        )

    with col5:
        total_income = st.number_input(
            "Monthly Income",
            min_value=5000.0,
            max_value=200000.0,
            value=18000.0,
            step=500.0,
        )
        has_existing_loan = st.selectbox("Has Existing Loan?", ["No", "Yes"])

    with col6:
        bureau_score = st.number_input(
            "Bureau Score, use -1 if no history",
            min_value=-1,
            max_value=900,
            value=650,
        )

    st.markdown("### Loan and Motorcycle Details")

    col7, col8, col9 = st.columns(3)

    with col7:
        manufacturer = st.selectbox(
            "Manufacturer",
            ["Honda", "Yamaha", "Suzuki", "Kawasaki", "Rusi", "TMX"],
        )
        loan_asset_cost = st.number_input(
            "Motorcycle Asset Cost",
            min_value=30000.0,
            max_value=200000.0,
            value=85000.0,
            step=500.0,
        )

    with col8:
        down_payment_amount = st.number_input(
            "Down Payment Amount",
            min_value=0.0,
            max_value=200000.0,
            value=12000.0,
            step=500.0,
        )
        number_of_installments = st.selectbox(
            "Number of Installments",
            [12, 18, 24, 36],
            index=2,
        )

    with col9:
        interest_rate = st.number_input(
            "Interest Rate",
            min_value=5.0,
            max_value=30.0,
            value=18.5,
            step=0.1,
        )
        channel = st.selectbox("Application Channel", ["DEALER", "DIRECT", "ONLINE"])

    submitted = st.form_submit_button("Score Application")

if submitted:
    payload = build_payload(
        applicant_name=applicant_name,
        bureau_score=int(bureau_score),
        down_payment_amount=float(down_payment_amount),
        res_years_at_current_city=int(res_years_at_current_city),
        res_years_at_current_address=int(res_years_at_current_address),
        interest_rate=float(interest_rate),
        loan_asset_cost=float(loan_asset_cost),
        age=int(age),
        number_of_installments=int(number_of_installments),
        total_income=float(total_income),
        years_in_occupation=int(years_in_occupation),
        number_of_dependents=int(number_of_dependents),
        gender=gender,
        channel=channel,
        marital_status=marital_status,
        manufacturer=manufacturer,
        occupation_type=occupation_type,
        res_resident_status=res_resident_status,
        has_existing_loan=has_existing_loan,
        loan_origination_date=loan_origination_date,
    )

    try:
        result = call_predict(api_url, payload)
        st.session_state["last_prediction"] = result
        st.session_state["last_payload"] = payload

    except RuntimeError as exc:
        st.error("Prediction failed.")
        st.write(exc)

if "last_prediction" in st.session_state:
    st.divider()
    st.subheader("Prediction Result")
    show_prediction_result(st.session_state["last_prediction"])

    with st.expander("Input sent to API"):
        st.json(st.session_state["last_payload"])

st.divider()
st.subheader("What this frontend demonstrates")

st.write(
    """
    This frontend demonstrates a real-time model scoring workflow:
    a loan officer enters an application, the FastAPI service validates it,
    the XGBoost model returns a default probability, and the system explains
    that the true label is still pending because default is only observable
    after the repayment window.
    """
)
