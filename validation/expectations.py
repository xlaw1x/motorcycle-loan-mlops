from pathlib import Path
from typing import List, Tuple

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
from great_expectations.expectations.row_conditions import Column


DATA_PATH = Path("data/raw/ph_motorcycle_loans.csv")
REPORT_PATH = Path("validation/latest_validation_report.txt")

REFERENCE_DATE = pd.Timestamp("2026-06-04")

REQUIRED_COLUMNS = [
    "loan_account_no",
    "product_description",
    "loan_origination_date",
    "bureau_score",
    "down_payment_amount",
    "branch_name",
    "res_years_at_current_city",
    "res_years_at_current_address",
    "interest_rate",
    "loan_asset_cost",
    "gender",
    "channel",
    "age",
    "number_of_installments",
    "res_city",
    "res_province",
    "total_income",
    "asset_model",
    "industry",
    "nature_of_business",
    "marital_status",
    "manufacturer",
    "occupation_type",
    "res_resident_status",
    "years_in_occupation",
    "has_existing_loan",
    "number_of_dependents",
    "days_past_due",
    "label_determination_date",
    "label_is_mature",
    "TARGET",
]

LEAKAGE_COLUMNS = [
    "days_past_due",
    "label_determination_date",
    "loan_origination_date",
    "TARGET",
    "label_is_mature",
]

VALID_GENDERS = ["Male", "Female"]
VALID_CHANNELS = ["DEALER", "DIRECT", "ONLINE"]
VALID_INSTALLMENTS = [12, 18, 24, 36]
VALID_MARITAL_STATUS = ["Single", "Married", "Live-in", "Widowed/Separated"]
VALID_RESIDENT_STATUS = ["Owned", "Rented", "Living with Parents"]
VALID_EXISTING_LOAN = ["Yes", "No"]
VALID_MANUFACTURERS = ["Honda", "Yamaha", "Suzuki", "Kawasaki", "Rusi", "TMX"]
VALID_OCCUPATIONS = [
    "Tricycle/Jeepney Driver",
    "Sari-sari Store Owner",
    "Construction Worker",
    "Employed",
    "OFW Dependent",
    "Other Self Employed",
]

ValidationRow = Tuple[str, str, str]


def make_batch(df: pd.DataFrame):
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = {
        "globally": False,
        "metric_calculations": False,
    }
    data_source = context.data_sources.add_pandas("moto2_pandas")
    data_asset = data_source.add_dataframe_asset(name="ph_motorcycle_loans")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_dataset")
    return batch_definition.get_batch(batch_parameters={"dataframe": df})


def make_expectation_suite():
    suite = gx.ExpectationSuite(name="moto2_raw_dataset_suite")

    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchOrderedList(column_list=REQUIRED_COLUMNS)
    )
    suite.add_expectation(gxe.ExpectTableRowCountToEqual(value=5000))

    for column in REQUIRED_COLUMNS:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=column))

    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="loan_account_no"))

    numeric_ranges = {
        "age": (18, 70),
        "total_income": (1, None),
        "loan_asset_cost": (30000, 200000),
        "down_payment_amount": (0, None),
        "interest_rate": (5, 30),
        "number_of_dependents": (0, 10),
        "days_past_due": (0, None),
        "res_years_at_current_city": (0, None),
        "res_years_at_current_address": (0, None),
    }

    for column, (min_value, max_value) in numeric_ranges.items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=min_value,
                max_value=max_value,
            )
        )

    suite.add_expectation(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="age",
            column_B="years_in_occupation",
            or_equal=True,
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="res_years_at_current_city",
            column_B="res_years_at_current_address",
            or_equal=True,
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="label_determination_date",
            column_B="loan_origination_date",
            or_equal=True,
        )
    )

    categorical_sets = {
        "gender": VALID_GENDERS,
        "channel": VALID_CHANNELS,
        "number_of_installments": VALID_INSTALLMENTS,
        "marital_status": VALID_MARITAL_STATUS,
        "res_resident_status": VALID_RESIDENT_STATUS,
        "has_existing_loan": VALID_EXISTING_LOAN,
        "manufacturer": VALID_MANUFACTURERS,
        "occupation_type": VALID_OCCUPATIONS,
        "TARGET": [0, 1],
        "label_is_mature": [True, False],
    }

    for column, value_set in categorical_sets.items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(column=column, value_set=value_set)
        )

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="bureau_score",
            min_value=300,
            max_value=900,
            row_condition=(Column("bureau_score") != -1),
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="TARGET",
            value_set=[1],
            row_condition=(Column("days_past_due") >= 90),
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="TARGET",
            value_set=[0],
            row_condition=(Column("days_past_due") < 90),
        )
    )

    return suite


def gx_result_rows(validation_result) -> List[ValidationRow]:
    rows: List[ValidationRow] = []

    for result in validation_result["results"]:
        config = result["expectation_config"]
        expectation_type = config["type"]
        kwargs = config.get("kwargs", {})
        column = kwargs.get("column")
        check_name = f"{expectation_type}:{column}" if column else expectation_type

        if result["success"]:
            rows.append((check_name, "PASS", "Great Expectations check passed."))
            continue

        details = result.get("result", {})
        message_parts = []
        for key in ("unexpected_count", "missing_count", "observed_value"):
            if details.get(key) is not None:
                message_parts.append(f"{key}={details[key]}")

        rows.append(
            (
                check_name,
                "FAIL",
                "; ".join(message_parts) or "Great Expectations check failed.",
            )
        )

    return rows


def add_business_rule_rows(df: pd.DataFrame, rows: List[ValidationRow]) -> None:
    df = df.copy()
    df["label_determination_date"] = pd.to_datetime(
        df["label_determination_date"],
        errors="coerce",
    )

    computed_maturity = df["label_determination_date"] <= REFERENCE_DATE
    stored_maturity = df["label_is_mature"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    mismatch_count = int((computed_maturity != stored_maturity).sum())

    if mismatch_count == 0:
        rows.append(
            (
                "label_maturity_consistent",
                "PASS",
                "label_is_mature matches label_determination_date <= 2026-06-04.",
            )
        )
    else:
        rows.append(
            (
                "label_maturity_consistent",
                "FAIL",
                f"{mismatch_count} rows have inconsistent label_is_mature values.",
            )
        )

    mature_share = df["label_is_mature"].astype(str).str.lower().eq("true").mean()
    if 0.75 <= mature_share <= 0.90:
        rows.append(
            ("mature_label_share", "PASS", f"Mature label share is {mature_share:.2%}.")
        )
    else:
        rows.append(
            ("mature_label_share", "WARN", f"Mature label share is {mature_share:.2%}.")
        )

    rows.append(
        (
            "leakage_guard_reminder",
            "INFO",
            (
                "These columns must never be model features: "
                f"{LEAKAGE_COLUMNS}. They are allowed in the raw dataset, "
                "but training must exclude them."
            ),
        )
    )


def validate_training_dataset(data_path: Path = DATA_PATH) -> List[ValidationRow]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    batch = make_batch(df)
    suite = make_expectation_suite()
    validation_result = batch.validate(suite)

    rows = gx_result_rows(validation_result)
    add_business_rule_rows(df, rows)

    return rows


def print_report(results: List[ValidationRow]) -> None:
    lines = [
        "=" * 80,
        "MOTO² Dataset Validation Report",
        "Validation engine: Great Expectations",
        "=" * 80,
    ]

    for check_name, status, message in results:
        lines.append(f"[{status}] {check_name}: {message}")

    fail_count = sum(1 for _, status, _ in results if status == "FAIL")
    warn_count = sum(1 for _, status, _ in results if status == "WARN")

    lines.append("=" * 80)
    if fail_count == 0:
        lines.append("FINAL RESULT: PASS, no blocking validation failures.")
    else:
        lines.append(f"FINAL RESULT: FAIL, {fail_count} blocking issue(s) found.")

    if warn_count > 0:
        lines.append(f"Warnings: {warn_count}. Review them before final demo.")

    report_text = "\n".join(lines)
    print(report_text)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    validation_results = validate_training_dataset()
    print_report(validation_results)
