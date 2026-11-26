"""Tests for risk scoring engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from modules.data_generator import generate_dataset
from modules.data_processor import clean_dataframe
from modules.duplicate_detector import detect_duplicates
from modules.risk_scoring import score_dataframe, score_single, _risk_category


def test_risk_category_boundaries():
    assert _risk_category(0)   == "Low"
    assert _risk_category(29)  == "Low"
    assert _risk_category(30)  == "Medium"
    assert _risk_category(59)  == "Medium"
    assert _risk_category(60)  == "High"
    assert _risk_category(79)  == "High"
    assert _risk_category(80)  == "Critical"
    assert _risk_category(100) == "Critical"
    print("✅ test_risk_category_boundaries passed")


def test_score_range():
    df = generate_dataset(normal_count=20, synthetic_count=10, fraud_ring_count=10, seed=1)
    df = clean_dataframe(df)
    df = detect_duplicates(df)
    df = score_dataframe(df)
    assert df["risk_score"].between(0, 100).all(), "Scores must be in [0,100]"
    assert "risk_category" in df.columns
    assert "risk_flags"    in df.columns
    print(f"✅ Scores in range. Distribution: {df['risk_category'].value_counts().to_dict()}")


def test_fraud_ring_scores_higher():
    df = generate_dataset(normal_count=20, fraud_ring_count=15, seed=42)
    df = clean_dataframe(df)
    df = detect_duplicates(df)
    df = score_dataframe(df)
    normal_avg = df[df["applicant_type"]=="normal"]["risk_score"].mean()
    ring_avg   = df[df["applicant_type"]=="fraud_ring"]["risk_score"].mean()
    print(f"Normal avg score: {normal_avg:.1f}, Ring avg score: {ring_avg:.1f}")
    assert ring_avg >= normal_avg, "Fraud-ring applicants should on average score >= normals"
    print("✅ test_fraud_ring_scores_higher passed")


if __name__ == "__main__":
    test_risk_category_boundaries()
    test_score_range()
    test_fraud_ring_scores_higher()
    print("\n🎉 All scoring tests passed.")
