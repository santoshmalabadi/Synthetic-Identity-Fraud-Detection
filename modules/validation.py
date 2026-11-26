"""
Validation module – field-level validation helpers used by multiple pages.
"""

import re
import pandas as pd
from modules.data_processor import calculate_age


def validate_applicant_dict(data: dict) -> tuple[bool, list[str], list[str]]:
    """
    Validate a dict of applicant fields.
    Returns (is_valid, errors, warnings).
    """
    errors   = []
    warnings = []

    # Required text fields
    if not str(data.get("full_name", "")).strip():
        errors.append("Full name is required.")

    email = str(data.get("email", ""))
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        errors.append(f"Invalid email: {email}")

    phone_digits = re.sub(r"\D", "", str(data.get("phone", "")))
    if len(phone_digits) < 10:
        errors.append("Phone number must have at least 10 digits.")

    dob = str(data.get("dob", ""))
    age = calculate_age(dob)
    if age is None:
        errors.append("Date of birth must be in YYYY-MM-DD format.")
    elif age < 18:
        errors.append(f"Applicant is underage ({age} years).")
    elif age > 100:
        warnings.append(f"Applicant age is {age} – please verify.")

    try:
        income = float(data.get("annual_income", 0))
        if income < 0:
            errors.append("Annual income cannot be negative.")
    except (TypeError, ValueError):
        errors.append("Annual income must be a valid number.")

    if not str(data.get("address", "")).strip():
        warnings.append("Address is missing.")

    if not str(data.get("city", "")).strip():
        warnings.append("City is missing.")

    return len(errors) == 0, errors, warnings


def validate_csv_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate uploaded CSV. Returns (cleaned_df, issues_list).
    """
    issues = []
    REQUIRED_COLS = ["full_name", "dob", "email", "phone", "address", "annual_income"]
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {', '.join(missing_cols)}")
        return df, issues

    # Check for empty required fields
    for col in REQUIRED_COLS:
        n_empty = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        if n_empty > 0:
            issues.append(f"Column '{col}' has {n_empty} empty value(s).")

    return df, issues
