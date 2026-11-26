"""
Identity Analyzer – detects inconsistencies within individual applicant records.
"""

import re
import pandas as pd
from datetime import datetime
from modules.data_processor import calculate_age


# ── Single-record checks ──────────────────────────────────────────────────

def _check_age_income(row: pd.Series) -> list[str]:
    flags = []
    age = calculate_age(str(row.get("dob", "")))
    income = float(row.get("annual_income", 0) or 0)
    if age and age < 25 and income > 3_000_000:
        flags.append(f"Very high income ({income:,.0f}) for young applicant (age {age})")
    if age and age > 60 and income > 8_000_000:
        flags.append(f"Unusually high income ({income:,.0f}) for applicant age {age}")
    return flags


def _check_unemployed_income(row: pd.Series) -> list[str]:
    flags = []
    emp = str(row.get("employment_type", "")).lower()
    income = float(row.get("annual_income", 0) or 0)
    if emp == "unemployed" and income > 500_000:
        flags.append(f"Unemployed applicant reports income {income:,.0f}")
    return flags


def _check_email_pattern(row: pd.Series) -> list[str]:
    flags = []
    email = str(row.get("email", ""))
    suspicious_domains = ["tempmail", "guerrilla", "throwam", "mailinator", "yopmail",
                          "trashmail", "fakeinbox", "sharklasers", "dispostable"]
    domain = email.split("@")[-1].lower() if "@" in email else ""
    if any(d in domain for d in suspicious_domains):
        flags.append(f"Suspicious disposable email domain: {domain}")
    # Randomly-looking email (lots of digits, no real name pattern)
    local = email.split("@")[0]
    if re.search(r"\d{5,}", local):
        flags.append("Email local part contains excessive digits (possibly auto-generated)")
    return flags


def _check_dob(row: pd.Series) -> list[str]:
    flags = []
    dob_str = str(row.get("dob", ""))
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        # Suspiciously round date: Jan 1st
        if dob.month == 1 and dob.day == 1:
            flags.append("Date of birth is Jan 1st – often a placeholder")
        age = (datetime.now() - dob).days // 365
        if age < 18:
            flags.append(f"Applicant appears underage (age {age})")
        if age > 90:
            flags.append(f"Applicant age {age} is unusually high")
    except ValueError:
        flags.append(f"Invalid date of birth format: {dob_str}")
    return flags


def _check_pincode(row: pd.Series) -> list[str]:
    flags = []
    pincode = str(row.get("pincode", ""))
    if re.match(r"^0+\d*$", pincode):
        flags.append(f"Pincode starts with zeros: {pincode} – possibly invalid")
    return flags


def _check_id_ref(row: pd.Series) -> list[str]:
    flags = []
    id_ref = str(row.get("id_ref", ""))
    # All-zeros suffix
    if re.search(r"0{8,}", id_ref):
        flags.append(f"ID reference has suspicious uniform pattern: {id_ref}")
    return flags


def _check_future_application(row: pd.Series) -> list[str]:
    flags = []
    app_date = str(row.get("application_date", ""))
    try:
        d = datetime.strptime(app_date, "%Y-%m-%d")
        if d > datetime.now():
            flags.append(f"Application date is in the future: {app_date}")
    except ValueError:
        pass
    return flags


# ── Batch Analysis ────────────────────────────────────────────────────────

CHECKS = [
    _check_age_income,
    _check_unemployed_income,
    _check_email_pattern,
    _check_dob,
    _check_pincode,
    _check_id_ref,
    _check_future_application,
]


def analyse_identity(row: pd.Series) -> dict:
    """
    Run all consistency checks on a single applicant row.
    Returns {"flags": [str], "inconsistency_count": int}.
    """
    all_flags = []
    for check in CHECKS:
        try:
            all_flags.extend(check(row))
        except Exception:
            pass
    return {
        "flags": all_flags,
        "inconsistency_count": len(all_flags),
    }


def analyse_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich DataFrame with identity analysis columns.
    Adds: identity_flags (list), identity_inconsistency_count (int).
    """
    df = df.copy()
    results = df.apply(analyse_identity, axis=1)
    df["identity_flags"] = results.apply(lambda r: r["flags"])
    df["identity_inconsistency_count"] = results.apply(lambda r: r["inconsistency_count"])
    return df
