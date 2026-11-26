"""Tests for graph engine and fraud ring detector."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.data_generator import generate_dataset
from modules.data_processor import clean_dataframe
from modules.graph_engine import build_graph, get_neighbors_info, compute_graph_stats, get_applicant_components
from modules.fraud_ring_detector import detect_fraud_rings


def test_graph_builds():
    df = generate_dataset(normal_count=20, synthetic_count=5, fraud_ring_count=10, seed=42)
    df = clean_dataframe(df)
    G  = build_graph(df)
    assert G.number_of_nodes() > 0
    print(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def test_fraud_ring_detection():
    df = generate_dataset(normal_count=10, synthetic_count=0, fraud_ring_count=15, seed=99)
    df = clean_dataframe(df)
    G  = build_graph(df)
    rings = detect_fraud_rings(df, G)
    assert isinstance(rings, list)
    print(f"✅ Rings detected: {len(rings)}")


def test_neighbors_info():
    df = generate_dataset(normal_count=20, fraud_ring_count=10, seed=7)
    df = clean_dataframe(df)
    G  = build_graph(df)
    aid = df.iloc[0]["applicant_id"]
    info = get_neighbors_info(G, aid)
    assert "connected_applicants" in info
    assert "shared_attrs" in info
    print(f"✅ Neighbors info OK for {aid}: {len(info['connected_applicants'])} connected")


if __name__ == "__main__":
    test_graph_builds()
    test_fraud_ring_detection()
    test_neighbors_info()
    print("\n🎉 All graph tests passed.")
