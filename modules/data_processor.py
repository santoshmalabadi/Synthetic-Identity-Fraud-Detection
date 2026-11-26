"""
Data Processor – cleans, normalises, and enriches applicant DataFrames.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime


REQUIRED_COLS = [
    "applicant_id", "full_name", "dob", "email", "phone",
    "address", "city", "state", "pincode", "id_ref", "device_id",
    "application_date", "employer", "employment_type", "annual_income",
]


def normalise_phone(phone: str) -> str:
    """Strip non-digit characters, keep last 10 digits."""
    if not isinstance(phone, str):
        return ""
    digits = re.sub(r"\D", "", phone)
    return digits[-10:] if len(digits) >= 10 else digits


def normalise_email(email: str) -> str:
    return str(email).strip().lower() if isinstance(email, str) else ""


def normalise_address(addr: str) -> str:
    if not isinstance(addr, str):
        return ""
    return re.sub(r"\s+", " ", addr.strip().lower())


def calculate_age(dob_str: str) -> int | None:
    try:
        dob = datetime.strptime(str(dob_str), "%Y-%m-%d")
        return (datetime.now() - dob).days // 365
    except Exception:
        return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise and add derived columns.
    Returns a copy of the DataFrame.
    """
    df = df.copy()

    # Ensure required columns exist
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""

    # Normalise key attributes
    df["phone_norm"] = df["phone"].apply(normalise_phone)
    df["email_norm"] = df["email"].apply(normalise_email)
    df["address_norm"] = df["address"].apply(normalise_address)

    # Derived fields
    df["age"] = df["dob"].apply(calculate_age)

    # Ensure numeric income
    df["annual_income"] = pd.to_numeric(df["annual_income"], errors="coerce").fillna(0)

    # Ensure risk_score is numeric
    if "risk_score" not in df.columns:
        df["risk_score"] = 0.0
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)

    # Default list columns
    if "risk_flags" not in df.columns:
        df["risk_flags"] = [[] for _ in range(len(df))]

    # Default string columns
    for col in ("risk_category", "risk_explanation", "analyst_note",
                "application_status", "ground_truth_label", "applicant_type"):
        if col not in df.columns:
            df[col] = ""

    return df


def validate_row(row: dict) -> dict:
    """
    Validate a single applicant dict.
    Returns {"valid": bool, "errors": [str], "warnings": [str]}.
    """
    errors = []
    warnings = []

    if not row.get("full_name", "").strip():
        errors.append("Full name is required.")

    email = row.get("email", "")
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", str(email)):
        errors.append("Invalid email format.")

    phone_digits = re.sub(r"\D", "", str(row.get("phone", "")))
    if len(phone_digits) < 10:
        errors.append("Phone number must have at least 10 digits.")

    dob_str = str(row.get("dob", ""))
    age = calculate_age(dob_str)
    if age is None:
        errors.append("Date of birth must be in YYYY-MM-DD format.")
    elif age < 18:
        errors.append("Applicant must be at least 18 years old.")
    elif age > 100:
        warnings.append("Date of birth indicates age > 100; please verify.")

    income = row.get("annual_income", 0)
    try:
        income = float(income)
        if income < 0:
            errors.append("Annual income cannot be negative.")
    except (ValueError, TypeError):
        errors.append("Annual income must be a number.")

    if not row.get("address", "").strip():
        warnings.append("Address is missing.")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def compute_attribute_frequencies(df: pd.DataFrame) -> dict:
    """
    Count how many applicants share each normalised attribute value.
    Returns dicts keyed by normalised attribute value.
    """
    result = {}
    for attr, col in [
        ("phone", "phone_norm"),
        ("email", "email_norm"),
        ("device_id", "device_id"),
        ("address", "address_norm"),
    ]:
        actual_col = col if col in df.columns else attr
        counts = df[actual_col].value_counts().to_dict()
        result[attr] = counts
    return result
