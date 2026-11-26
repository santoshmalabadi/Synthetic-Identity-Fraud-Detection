"""
Anomaly Detector – optional ML-based anomaly detection using Isolation Forest.
Supplements rule-based scoring with statistical outlier detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder


NUMERIC_FEATURES = ["annual_income", "age"]
CATEGORICAL_FEATURES = ["employment_type"]


def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    from modules.data_processor import calculate_age
    df = df.copy()
    if "age" not in df.columns:
        df["age"] = df["dob"].apply(lambda d: calculate_age(str(d)) or 30)
    df["annual_income"] = pd.to_numeric(df["annual_income"], errors="coerce").fillna(0)
    df["age"]           = pd.to_numeric(df["age"],           errors="coerce").fillna(30)

    # Encode categorical
    le = LabelEncoder()
    df["emp_encoded"] = le.fit_transform(
        df["employment_type"].fillna("Unknown").astype(str)
    )

    features = df[NUMERIC_FEATURES + ["emp_encoded"]].values
    return features


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.15) -> pd.DataFrame:
    """
    Run Isolation Forest on numeric applicant features.
    Adds column 'anomaly_score' (higher = more anomalous, in [0, 1]).
    Returns enriched DataFrame.
    """
    if df.empty or len(df) < 10:
        df = df.copy()
        df["anomaly_score"] = 0.0
        df["is_anomaly"]    = False
        return df

    df = df.copy()
    X  = _build_feature_matrix(df)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X)

    # decision_function: lower = more anomalous
    raw_scores = model.decision_function(X)
    # Normalise to [0, 1] where 1 = most anomalous
    score_range = raw_scores.max() - raw_scores.min()
    normalised = 1.0 - (raw_scores - raw_scores.min()) / (score_range + 1e-9)
    df["anomaly_score"] = np.round(normalised, 4)
    df["is_anomaly"]    = model.predict(X) == -1
    return df
