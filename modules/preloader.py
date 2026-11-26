"""
Preload demo data into the database on first run.
Called from app.py if the DB is empty.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_generator import generate_dataset
from modules.data_processor import clean_dataframe
from modules.duplicate_detector import detect_duplicates
from modules.risk_scoring import score_dataframe, apply_ring_membership_bonus
from modules.graph_engine import build_graph
from modules.fraud_ring_detector import detect_fraud_rings
from modules.database_manager import (
    init_db, insert_applicants, save_fraud_rings, get_total_count
)


def preload_if_empty(seed: int = 42):
    """
    If the database is empty, generate and persist a demo dataset automatically.
    This ensures the dashboard is populated on first launch.
    """
    init_db()
    if get_total_count() > 0:
        return  # Already has data

    df = generate_dataset(
        normal_count=60,
        synthetic_count=20,
        fraud_ring_count=15,
        new_count=5,
        seed=seed,
    )
    df = clean_dataframe(df)
    df = detect_duplicates(df)
    df = score_dataframe(df)

    G     = build_graph(df)
    rings = detect_fraud_rings(df, G)
    df    = apply_ring_membership_bonus(df, rings)
    save_fraud_rings(rings)
    insert_applicants(df)
