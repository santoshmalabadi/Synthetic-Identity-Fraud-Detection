"""
Performance Evaluator – measures detection quality against ground-truth labels.
Never fabricates results; uses actual generated label data.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from config import RISK_LOW_MAX, RISK_MEDIUM_MAX, RISK_HIGH_MAX


FRAUD_LABELS = {"synthetic_fraud", "fraud_ring"}


def _binary_ground_truth(df: pd.DataFrame) -> pd.Series:
    """1 = fraud, 0 = genuine."""
    return df["ground_truth_label"].apply(
        lambda x: 1 if str(x) in FRAUD_LABELS else 0
    )


def _binary_prediction(df: pd.DataFrame, threshold: float) -> pd.Series:
    """1 = predicted fraud (risk_score >= threshold), 0 = not."""
    return (df["risk_score"] >= threshold).astype(int)


def evaluate(df: pd.DataFrame, threshold: float = 60.0) -> dict:
    """
    Evaluate detection performance at a given risk-score threshold.
    Returns metric dict with confusion matrix and classification report.
    """
    if df.empty or "ground_truth_label" not in df.columns:
        return {"error": "No labelled data available."}

    labelled = df[df["ground_truth_label"].notna()].copy()
    if labelled.empty:
        return {"error": "No labelled data available."}

    y_true = _binary_ground_truth(labelled)
    y_pred = _binary_prediction(labelled, threshold)

    if y_true.nunique() < 2:
        return {"error": "Only one class present in ground truth – cannot evaluate."}

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    total = len(y_true)
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold":        threshold,
        "total_samples":    total,
        "fraud_samples":    int(y_true.sum()),
        "genuine_samples":  int(total - y_true.sum()),
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "false_positive_rate": round(fp_rate, 4),
        "true_positives":   int(tp),
        "false_positives":  int(fp),
        "true_negatives":   int(tn),
        "false_negatives":  int(fn),
        "confusion_matrix": cm.tolist(),
        "report":           classification_report(
                                y_true, y_pred,
                                target_names=["Genuine", "Fraud"],
                                zero_division=0
                            ),
    }


def evaluate_multiple_thresholds(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate at several thresholds and return a comparison DataFrame.
    """
    if thresholds is None:
        thresholds = [20, 30, 40, 50, 60, 70, 80]

    rows = []
    for t in thresholds:
        result = evaluate(df, threshold=t)
        if "error" not in result:
            rows.append({
                "Threshold":    t,
                "Accuracy":     result["accuracy"],
                "Precision":    result["precision"],
                "Recall":       result["recall"],
                "F1 Score":     result["f1_score"],
                "FP Rate":      result["false_positive_rate"],
                "TP":           result["true_positives"],
                "FP":           result["false_positives"],
                "FN":           result["false_negatives"],
                "TN":           result["true_negatives"],
            })
    return pd.DataFrame(rows)
