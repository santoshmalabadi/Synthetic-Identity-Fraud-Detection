"""
Central configuration for Synthetic Identity Fraud Detection Platform.
All thresholds, weights, and settings are configurable here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GENERATED_DIR = os.path.join(DATA_DIR, "generated")
DATABASE_DIR = os.path.join(DATA_DIR, "database")
DB_PATH = os.path.join(DATABASE_DIR, "fraud_detection.db")

# ── Data Generation ────────────────────────────────────────────────────────
DEFAULT_NORMAL_COUNT = 60
DEFAULT_SYNTHETIC_COUNT = 20
DEFAULT_FRAUD_RING_COUNT = 15
DEFAULT_NEW_COUNT = 5

TOTAL_DEFAULT = DEFAULT_NORMAL_COUNT + DEFAULT_SYNTHETIC_COUNT + DEFAULT_FRAUD_RING_COUNT + DEFAULT_NEW_COUNT

# ── Risk Score Thresholds ──────────────────────────────────────────────────
RISK_LOW_MAX = 29
RISK_MEDIUM_MAX = 59
RISK_HIGH_MAX = 79
# 80–100 = Critical

RISK_CATEGORIES = {
    "Low": (0, RISK_LOW_MAX),
    "Medium": (RISK_LOW_MAX + 1, RISK_MEDIUM_MAX),
    "High": (RISK_MEDIUM_MAX + 1, RISK_HIGH_MAX),
    "Critical": (RISK_HIGH_MAX + 1, 100),
}

# ── Risk Scoring Weights ───────────────────────────────────────────────────
RISK_WEIGHTS = {
    "shared_phone": 20,
    "shared_email": 20,
    "shared_device": 25,
    "shared_address": 10,
    "identity_inconsistency": 15,
    "multiple_applications": 10,
    "fraud_ring_member": 30,
    "suspicious_income": 5,
    "address_email_mismatch": 8,
    "velocity_flag": 12,
}

# Shared-attribute thresholds: how many applicants sharing before suspicious
SHARED_PHONE_THRESHOLD = 2
SHARED_EMAIL_THRESHOLD = 2
SHARED_DEVICE_THRESHOLD = 3
SHARED_ADDRESS_THRESHOLD = 4   # families legitimately share

# ── Fraud Ring Discovery ───────────────────────────────────────────────────
MIN_FRAUD_RING_SIZE = 2
FRAUD_RING_MIN_SHARED_ATTRS = 1   # edges needed to form a cluster

# ── Application Status Labels ──────────────────────────────────────────────
STATUS_NEW = "New"
STATUS_REVIEW = "Under Review"
STATUS_SUSPICIOUS = "Suspicious"
STATUS_CLEARED = "Cleared"
STATUS_CONFIRMED_FRAUD = "Confirmed Fraud"
STATUS_ANALYST_REVIEWED = "Analyst Reviewed"

ALL_STATUSES = [
    STATUS_NEW,
    STATUS_REVIEW,
    STATUS_SUSPICIOUS,
    STATUS_CLEARED,
    STATUS_CONFIRMED_FRAUD,
    STATUS_ANALYST_REVIEWED,
]

# ── UI Theme Colours ───────────────────────────────────────────────────────
COLOR_BG = "#0a0e1a"
COLOR_CARD = "#0d1b2a"
COLOR_ACCENT = "#1e90ff"
COLOR_CYAN = "#00d4ff"
COLOR_GREEN = "#00c896"
COLOR_AMBER = "#ffb300"
COLOR_RED = "#ff4c4c"
COLOR_TEXT = "#e0e6f0"
COLOR_MUTED = "#8899aa"

RISK_COLORS = {
    "Low": COLOR_GREEN,
    "Medium": COLOR_AMBER,
    "High": "#ff7700",
    "Critical": COLOR_RED,
}

# ── Scenario Seeds ─────────────────────────────────────────────────────────
SCENARIO_SEEDS = {
    "synthetic_identity": 42,
    "fraud_ring": 99,
    "genuine_family": 7,
    "new_identity": 123,
}
