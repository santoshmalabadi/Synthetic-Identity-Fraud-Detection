"""
Fraud Explanation Generator – produces dynamic 2-3 line natural language
explanations for every applicant based on actual detection results.

Uses: score_parts, risk_flags, risk_category, DB context, applicant_type.
Never modifies the risk-scoring algorithm. Read-only.
"""

from __future__ import annotations

import pandas as pd

from modules.data_processor import normalise_phone, normalise_email, normalise_address


# ── Helpers ───────────────────────────────────────────────────────────────────

def _phone_peers(phone: str, df_context: pd.DataFrame) -> pd.DataFrame:
    """Return all OTHER applicants sharing the same normalised phone number."""
    if df_context.empty or not phone:
        return pd.DataFrame()
    nq = normalise_phone(phone)
    if not nq:
        return pd.DataFrame()
    if "phone_norm" not in df_context.columns:
        df_context = df_context.copy()
        df_context["phone_norm"] = df_context["phone"].apply(normalise_phone)
    return df_context[df_context["phone_norm"] == nq]


def _email_peers(email: str, df_context: pd.DataFrame) -> pd.DataFrame:
    if df_context.empty or not email:
        return pd.DataFrame()
    nq = normalise_email(email)
    if not nq:
        return pd.DataFrame()
    if "email_norm" not in df_context.columns:
        df_context = df_context.copy()
        df_context["email_norm"] = df_context["email"].apply(normalise_email)
    return df_context[df_context["email_norm"] == nq]


def _device_peers(device_id: str, df_context: pd.DataFrame) -> pd.DataFrame:
    if df_context.empty or not device_id:
        return pd.DataFrame()
    return df_context[df_context["device_id"] == device_id.strip()]


def _address_peers(address: str, df_context: pd.DataFrame) -> pd.DataFrame:
    if df_context.empty or not address:
        return pd.DataFrame()
    nq = normalise_address(address)
    if not nq:
        return pd.DataFrame()
    if "address_norm" not in df_context.columns:
        df_context = df_context.copy()
        df_context["address_norm"] = df_context["address"].apply(normalise_address)
    return df_context[df_context["address_norm"] == nq]


def _ring_members_in_peers(peers: pd.DataFrame) -> int:
    """Count peers who are fraud ring members."""
    if peers.empty:
        return 0
    return int(peers["applicant_type"].isin(["fraud_ring"]).sum())


def _high_risk_peers(peers: pd.DataFrame) -> int:
    if peers.empty:
        return 0
    return int(peers["risk_category"].isin(["High", "Critical"]).sum())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def generate_fraud_explanation(
    applicant: pd.Series,
    score_result: dict,
    df_context: pd.DataFrame,
) -> str:
    """
    Generate a dynamic 2–3 line explanation for why an applicant was
    classified at their particular risk level.

    Parameters
    ----------
    applicant    : the applicant's Series (from the submitted form record)
    score_result : dict returned by score_single()
    df_context   : full applicant DataFrame for peer lookups

    Returns
    -------
    str : 2–3 sentence natural-language explanation
    """
    category    = score_result.get("risk_category", "Low")
    score       = float(score_result.get("risk_score", 0))
    score_parts = score_result.get("score_parts", {})
    atype       = str(applicant.get("applicant_type", "normal")).lower()

    phone   = str(applicant.get("phone", ""))
    email   = str(applicant.get("email", ""))
    device  = str(applicant.get("device_id", ""))
    address = str(applicant.get("address", ""))

    # Gather peer data (exclude self)
    aid = str(applicant.get("applicant_id", ""))
    ctx = df_context[df_context["applicant_id"] != aid] if not df_context.empty else df_context

    phone_peers   = _phone_peers(phone, ctx)
    email_peers   = _email_peers(email, ctx)
    device_peers  = _device_peers(device, ctx)
    address_peers = _address_peers(address, ctx)

    n_phone   = len(phone_peers)
    n_email   = len(email_peers)
    n_device  = len(device_peers)
    n_address = len(address_peers)

    ring_via_phone   = _ring_members_in_peers(phone_peers)
    ring_via_device  = _ring_members_in_peers(device_peers)
    high_risk_phone  = _high_risk_peers(phone_peers)
    high_risk_device = _high_risk_peers(device_peers)

    is_fraud_ring  = "fraud_ring"         in atype
    is_synthetic   = "synthetic_identity" in atype
    is_new_id      = "new_identity"       in atype

    shared_count  = len(score_parts)        # number of distinct shared attributes
    identity_flag = "identity_inconsistency" in score_parts

    # ── Build sentences ───────────────────────────────────────────────────────
    sentences: list[str] = []

    # --- Sentence 1: primary signal -------------------------------------------
    if category == "Critical":
        if n_phone >= 3 and ring_via_phone >= 1:
            sentences.append(
                f"This applicant's phone number is shared with {n_phone} other applicant(s), "
                f"of whom {ring_via_phone} are confirmed fraud ring member(s), "
                f"triggering a critical risk classification."
            )
        elif n_phone >= 3:
            sentences.append(
                f"This applicant's phone number is shared with {n_phone} other applicant(s). "
                f"Multiple identities linked to a single phone number is a strong synthetic "
                f"identity indicator, resulting in a critical risk score of {score:.0f}/100."
            )
        elif is_fraud_ring:
            sentences.append(
                f"This applicant is identified as a fraud ring member with a risk score of "
                f"{score:.0f}/100. Fraud ring membership indicates coordinated synthetic "
                f"identity activity with other connected applicants."
            )
        elif shared_count >= 3:
            sentences.append(
                f"This applicant has {shared_count} shared attributes (phone, device, and/or "
                f"address) overlapping with other records in the database, producing a "
                f"critical composite risk score of {score:.0f}/100."
            )
        else:
            sentences.append(
                f"Multiple high-risk signals were detected during assessment, resulting in a "
                f"critical risk score of {score:.0f}/100. The combination of shared attributes "
                f"and identity indicators places this applicant in the highest risk tier."
            )

    elif category == "High":
        if n_phone >= 2:
            sentences.append(
                f"This applicant's phone number is shared with {n_phone} other applicant(s). "
                f"{'Of these, ' + str(high_risk_phone) + ' have elevated risk scores, suggesting possible coordinated activity.' if high_risk_phone else 'The repeated use of the same phone number across identities is a notable risk signal.'}"
            )
        elif n_device >= 2:
            sentences.append(
                f"This applicant's device ID is registered to {n_device} other applicant(s). "
                f"Device reuse across multiple identities is a strong indicator of "
                f"synthetic identity misuse, contributing to a high risk score of {score:.0f}/100."
            )
        elif identity_flag:
            sentences.append(
                f"Identity inconsistencies were detected during assessment — "
                f"details in the submitted profile do not align with expected patterns, "
                f"raising the risk score to {score:.0f}/100 (High)."
            )
        else:
            sentences.append(
                f"The applicant's profile contains {shared_count} shared attribute(s) with "
                f"existing records, producing a high risk score of {score:.0f}/100. "
                f"Manual review is recommended before proceeding."
            )

    elif category == "Medium":
        if n_phone >= 1:
            sentences.append(
                f"This applicant shares a phone number with {n_phone} other applicant(s) "
                f"in the database. While phone sharing alone is not conclusive evidence of "
                f"fraud, the association warrants additional verification."
            )
        elif n_email >= 1 or n_device >= 1 or n_address >= 1:
            shared_desc = " and ".join(
                p for p, n in [("email", n_email), ("device", n_device), ("address", n_address)]
                if n >= 1
            )
            sentences.append(
                f"A shared {shared_desc} was detected across multiple applicant records. "
                f"This produced a medium risk score of {score:.0f}/100, indicating a "
                f"moderate level of investigation priority."
            )
        elif identity_flag:
            sentences.append(
                f"Minor identity inconsistencies were flagged during the assessment process. "
                f"These anomalies resulted in a medium risk score of {score:.0f}/100 and "
                f"suggest requesting supplementary identity documents."
            )
        else:
            sentences.append(
                f"The applicant's risk score of {score:.0f}/100 falls in the medium range. "
                f"Some risk indicators were present, but no single dominant signal was identified. "
                f"Standard additional verification is recommended."
            )

    else:  # Low
        sentences.append(
            "No suspicious phone number associations, shared device IDs, or repeated "
            "address links were identified for this applicant. "
            f"The risk score of {score:.0f}/100 reflects a clean profile with no "
            "high-risk signals detected."
        )

    # --- Sentence 2: supporting evidence / connection context -----------------
    if category in ("Critical", "High"):
        parts = []
        if n_phone >= 1 and ring_via_phone > 0:
            parts.append(
                f"{ring_via_phone} phone-linked applicant(s) are confirmed fraud ring member(s)"
            )
        if n_device >= 2:
            parts.append(f"device ID reused across {n_device} identities")
        if n_address >= 2:
            parts.append(f"address shared with {n_address} other record(s)")
        if n_email >= 2:
            parts.append(f"email address linked to {n_email} accounts")
        if identity_flag:
            parts.append("identity inconsistencies in the submitted profile")
        if is_synthetic:
            parts.append("profile matches a synthetic identity pattern")

        if parts and len(sentences) < 2:
            sentences.append(
                "Contributing factors include: " + "; ".join(parts) + "."
            )
        elif parts and len(sentences[0]) < 180:
            # Append a compact supporting line
            sentences.append("Additional signals: " + "; ".join(parts[:3]) + ".")

    elif category == "Medium":
        if high_risk_phone > 0:
            sentences.append(
                f"{high_risk_phone} of the linked applicant(s) carry a High or Critical "
                f"risk score, suggesting possible coordinated identity misuse."
            )
        elif is_new_id:
            sentences.append(
                "The applicant's profile matches a new-identity pattern, "
                "which may indicate a recently fabricated record."
            )

    # --- Sentence 3: disclaimer / recommendation ------------------------------
    if category == "Critical":
        sentences.append(
            "⚠️ Shared attributes are relationship signals — fraud is not automatically "
            "confirmed. Immediate analyst review is required before any further action."
        )
    elif category == "High":
        sentences.append(
            "These findings indicate elevated investigation priority. "
            "Analyst review and additional document verification are strongly advised."
        )
    elif category == "Medium":
        sentences.append(
            "No connection to known fraud rings was established. "
            "Standard additional verification steps are sufficient at this stage."
        )
    else:
        sentences.append(
            "No connections to known fraud rings or high-risk applicants were detected. "
            "Standard processing may continue."
        )

    # Join and trim to 3 sentences max
    final = " ".join(sentences[:3])
    return final
