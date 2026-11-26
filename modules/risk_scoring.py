"""
Risk Scoring Engine – calculates an explainable 0-100 risk score per applicant.
Weights are configurable via config.py.
"""

import pandas as pd
import numpy as np
from config import (
    RISK_WEIGHTS, RISK_LOW_MAX, RISK_MEDIUM_MAX, RISK_HIGH_MAX,
    SHARED_PHONE_THRESHOLD, SHARED_EMAIL_THRESHOLD,
    SHARED_DEVICE_THRESHOLD, SHARED_ADDRESS_THRESHOLD,
)
from modules.data_processor import normalise_phone, normalise_email, normalise_address
from modules.identity_analyzer import analyse_identity
from modules.duplicate_detector import detect_duplicates


def apply_ring_membership_bonus(df: pd.DataFrame, rings: list) -> pd.DataFrame:
    """
    After fraud-ring detection, boost risk scores for members of identified rings.
    Should be called after score_dataframe() and detect_fraud_rings().
    """
    if not rings or df.empty:
        return df

    df = df.copy()
    ring_member_ids = {}
    for ring in rings:
        for mid in ring.get("member_ids", []):
            ring_size    = ring.get("member_count", 2)
            risk_level   = ring.get("risk_level", "Medium")
            ring_member_ids[mid] = (ring_size, risk_level)

    bonus_map = {"Critical": 35, "High": 25, "Medium": 15}

    for idx, row in df.iterrows():
        aid = row["applicant_id"]
        if aid in ring_member_ids:
            ring_size, risk_level = ring_member_ids[aid]
            bonus = bonus_map.get(risk_level, 12)
            # Scale bonus with ring size (larger ring = more confident signal)
            bonus = min(bonus + (ring_size - 2) * 4, RISK_WEIGHTS["fraud_ring_member"])
            new_score    = min(float(row["risk_score"]) + bonus, 100)
            new_category = _risk_category(new_score)
            flags        = row.get("risk_flags", [])
            if isinstance(flags, str):
                import json
                try:    flags = json.loads(flags)
                except: flags = []
            flags = list(flags) + [
                f"Member of fraud ring ({ring_size} applicants, {risk_level} risk) (+{bonus} pts)"
            ]
            expl = str(row.get("risk_explanation", ""))
            expl += f"\n  • Fraud ring membership: +{bonus} pts → {new_score:.0f}/100 ({new_category})"

            df.at[idx, "risk_score"]       = new_score
            df.at[idx, "risk_category"]    = new_category
            df.at[idx, "risk_flags"]       = flags
            df.at[idx, "risk_explanation"] = expl

    return df


def _risk_category(score: float) -> str:
    if score <= RISK_LOW_MAX:
        return "Low"
    elif score <= RISK_MEDIUM_MAX:
        return "Medium"
    elif score <= RISK_HIGH_MAX:
        return "High"
    return "Critical"


def score_single(row: pd.Series, df_context: pd.DataFrame) -> dict:
    """
    Score one applicant using context (full dataset) for attribute-sharing checks.
    Returns dict with score, category, flags, explanation, recommendation.
    """
    flags       = []
    score_parts = {}

    # ── 1. Shared attributes ───────────────────────────────────────────────
    phone_norm   = normalise_phone(str(row.get("phone", "")))
    email_norm   = normalise_email(str(row.get("email", "")))
    address_norm = normalise_address(str(row.get("address", "")))
    device_id    = str(row.get("device_id", ""))

    if not df_context.empty:
        ctx = df_context.copy()
        if "phone_norm" not in ctx.columns:
            ctx["phone_norm"]   = ctx["phone"].apply(normalise_phone)
        if "email_norm" not in ctx.columns:
            ctx["email_norm"]   = ctx["email"].apply(normalise_email)
        if "address_norm" not in ctx.columns:
            ctx["address_norm"] = ctx["address"].apply(normalise_address)

        phone_count   = (ctx["phone_norm"]   == phone_norm).sum()   if phone_norm   else 0
        email_count   = (ctx["email_norm"]   == email_norm).sum()   if email_norm   else 0
        device_count  = (ctx["device_id"]    == device_id).sum()    if device_id    else 0
        address_count = (ctx["address_norm"] == address_norm).sum() if address_norm else 0
    else:
        phone_count = email_count = device_count = address_count = 0

    # Subtract self
    phone_count   = max(0, phone_count   - 1)
    email_count   = max(0, email_count   - 1)
    device_count  = max(0, device_count  - 1)
    address_count = max(0, address_count - 1)

    if phone_count >= SHARED_PHONE_THRESHOLD - 1:
        pts = RISK_WEIGHTS["shared_phone"]
        score_parts["shared_phone"] = pts
        flags.append(f"Phone shared with {phone_count} other applicant(s) (+{pts} pts)")

    if email_count >= SHARED_EMAIL_THRESHOLD - 1:
        pts = RISK_WEIGHTS["shared_email"]
        score_parts["shared_email"] = pts
        flags.append(f"Email shared with {email_count} other applicant(s) (+{pts} pts)")

    if device_count >= SHARED_DEVICE_THRESHOLD - 1:
        pts = RISK_WEIGHTS["shared_device"]
        score_parts["shared_device"] = pts
        flags.append(f"Device ID shared with {device_count} other applicant(s) (+{pts} pts)")

    if address_count >= SHARED_ADDRESS_THRESHOLD - 1:
        pts = RISK_WEIGHTS["shared_address"]
        score_parts["shared_address"] = pts
        flags.append(f"Address shared with {address_count} other applicant(s) (+{pts} pts)")

    # ── 2. Identity inconsistencies ────────────────────────────────────────
    identity_result = analyse_identity(row)
    n_inconsistencies = identity_result["inconsistency_count"]
    if n_inconsistencies > 0:
        pts = min(RISK_WEIGHTS["identity_inconsistency"] * n_inconsistencies, 30)
        score_parts["identity_inconsistency"] = pts
        for f in identity_result["flags"]:
            flags.append(f"{f} (+{RISK_WEIGHTS['identity_inconsistency']} pts)")

    # ── 3. Multiple shared attributes (compound signal) ────────────────────
    n_shared = sum(1 for k in ("shared_phone", "shared_email",
                               "shared_device", "shared_address")
                   if k in score_parts)
    if n_shared >= 2:
        pts = RISK_WEIGHTS["multiple_applications"]
        score_parts["multiple_shared"] = pts
        flags.append(f"Multiple ({n_shared}) shared attributes detected (+{pts} pts)")

    # ── 4. Velocity / future date ──────────────────────────────────────────
    from datetime import datetime
    app_date = str(row.get("application_date", ""))
    try:
        d = datetime.strptime(app_date, "%Y-%m-%d")
        if d > datetime.now():
            pts = RISK_WEIGHTS["velocity_flag"]
            score_parts["velocity"] = pts
            flags.append(f"Future application date ({app_date}) (+{pts} pts)")
    except ValueError:
        pass

    # ── Compute final score ────────────────────────────────────────────────
    raw_score = sum(score_parts.values())
    final_score = float(min(raw_score, 100))
    category = _risk_category(final_score)

    # ── Recommendation ─────────────────────────────────────────────────────
    if category == "Low":
        recommendation = "No immediate action required. Continue standard verification."
    elif category == "Medium":
        recommendation = "Request additional identity verification documents."
    elif category == "High":
        recommendation = "Escalate to fraud analyst for manual review before approval."
    else:
        recommendation = "Immediate analyst review required. Do not process until cleared."

    explanation_parts = ["Risk Score Breakdown:"]
    for key, pts in score_parts.items():
        explanation_parts.append(f"  • {key.replace('_', ' ').title()}: +{pts}")
    explanation_parts.append(f"  Total: {final_score:.0f}/100 → {category} Risk")
    explanation_parts.append(f"Recommendation: {recommendation}")

    return {
        "risk_score":       final_score,
        "risk_category":    category,
        "risk_flags":       flags,
        "risk_explanation": "\n".join(explanation_parts),
        "recommendation":   recommendation,
        "score_parts":      score_parts,
    }


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all applicants in a DataFrame.
    Returns the DataFrame with risk columns populated.
    """
    if df.empty:
        return df

    df = df.copy()
    scores       = []
    categories   = []
    flag_lists   = []
    explanations = []

    for _, row in df.iterrows():
        result = score_single(row, df)
        scores.append(result["risk_score"])
        categories.append(result["risk_category"])
        flag_lists.append(result["risk_flags"])
        explanations.append(result["risk_explanation"])

    df["risk_score"]       = scores
    df["risk_category"]    = categories
    df["risk_flags"]       = flag_lists
    df["risk_explanation"] = explanations
    return df
