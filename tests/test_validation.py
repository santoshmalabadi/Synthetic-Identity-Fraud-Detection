"""Tests for validation module."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.validation import validate_applicant_dict
import pandas as pd


def test_valid_applicant():
    data = {
        "full_name": "Rahul Sharma",
        "dob": "1990-06-15",
        "email": "rahul@example.com",
        "phone": "9876543210",
        "address": "123 Main Street",
        "city": "Mumbai",
        "annual_income": 600000,
    }
    valid, errors, warnings = validate_applicant_dict(data)
    assert valid, f"Should be valid: {errors}"
    print("✅ test_valid_applicant passed")


def test_missing_name():
    data = {"full_name": "", "dob": "1990-06-15",
            "email": "a@b.com", "phone": "9876543210",
            "address": "123 Main", "annual_income": 500000}
    valid, errors, _ = validate_applicant_dict(data)
    assert not valid
    assert any("name" in e.lower() for e in errors)
    print("✅ test_missing_name passed")


def test_underage():
    data = {"full_name": "Child", "dob": "2015-01-01",
            "email": "a@b.com", "phone": "9876543210",
            "address": "123 Main", "annual_income": 0}
    valid, errors, _ = validate_applicant_dict(data)
    assert not valid
    assert any("underage" in e.lower() or "18" in e for e in errors)
    print("✅ test_underage passed")


def test_invalid_email():
    data = {"full_name": "Test", "dob": "1990-01-01",
            "email": "not-an-email", "phone": "9876543210",
            "address": "123 Main", "annual_income": 500000}
    valid, errors, _ = validate_applicant_dict(data)
    assert not valid
    assert any("email" in e.lower() for e in errors)
    print("✅ test_invalid_email passed")


if __name__ == "__main__":
    test_valid_applicant()
    test_missing_name()
    test_underage()
    test_invalid_email()
    print("\n🎉 All validation tests passed.")
