"""Tests for data_generator module."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.data_generator import (
    generate_dataset, scenario_synthetic_identity,
    scenario_fraud_ring, scenario_genuine_family, scenario_new_identity,
)

def test_generate_dataset_basic():
    df = generate_dataset(normal_count=10, synthetic_count=5,
                           fraud_ring_count=5, new_count=2, seed=42)
    assert len(df) == 22, f"Expected 22, got {len(df)}"
    assert "applicant_id" in df.columns
    assert "ground_truth_label" in df.columns
    print("✅ test_generate_dataset_basic passed")

def test_no_real_id_numbers():
    df = generate_dataset(normal_count=20, seed=1)
    assert df["id_ref"].str.startswith("SYN").sum() + \
           df["id_ref"].str.startswith("REF").sum() + \
           df["id_ref"].str.startswith("DOC").sum() == len(df), \
        "ID refs should start with SYN/REF/DOC"
    print("✅ test_no_real_id_numbers passed")

def test_scenarios():
    s1 = scenario_synthetic_identity()
    assert len(s1) == 1
    assert s1.iloc[0]["ground_truth_label"] == "synthetic_fraud"

    s2 = scenario_fraud_ring()
    assert len(s2) == 5
    assert all(s2["ground_truth_label"] == "fraud_ring")

    s3 = scenario_genuine_family()
    assert len(s3) == 3
    assert all(s3["ground_truth_label"] == "genuine")
    # All share same address
    assert s3["address"].nunique() == 1

    s4 = scenario_new_identity()
    assert len(s4) == 1
    print("✅ test_scenarios passed")

if __name__ == "__main__":
    test_generate_dataset_basic()
    test_no_real_id_numbers()
    test_scenarios()
    print("\n🎉 All generator tests passed.")
