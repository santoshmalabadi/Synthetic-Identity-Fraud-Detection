"""
Duplicate Detector – finds repeated attributes across applicant records.

Also provides the low-level attribute index used by the relationship graph
engine:  build_attribute_index / pair_links / shares_for.
"""

from __future__ import annotations

import pandas as pd
from config import (
    SHARED_PHONE_THRESHOLD, SHARED_EMAIL_THRESHOLD,
    SHARED_DEVICE_THRESHOLD, SHARED_ADDRESS_THRESHOLD,
)
from modules.data_processor import normalise_phone, normalise_email, normalise_address


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns for how many applicants share each key attribute.
    Returns enriched DataFrame with *_count and *_suspicious columns.
    """
    df = df.copy()

    # Normalise if not already done
    if "phone_norm" not in df.columns:
        df["phone_norm"] = df["phone"].apply(normalise_phone)
    if "email_norm" not in df.columns:
        df["email_norm"] = df["email"].apply(normalise_email)
    if "address_norm" not in df.columns:
        df["address_norm"] = df["address"].apply(normalise_address)

    # Count occurrences per normalised value
    for col, threshold, label in [
        ("phone_norm",   SHARED_PHONE_THRESHOLD,   "phone"),
        ("email_norm",   SHARED_EMAIL_THRESHOLD,   "email"),
        ("device_id",    SHARED_DEVICE_THRESHOLD,  "device"),
        ("address_norm", SHARED_ADDRESS_THRESHOLD, "address"),
    ]:
        counts = df[col].map(df[col].value_counts())
        df[f"{label}_shared_count"] = counts.fillna(1).astype(int)
        df[f"{label}_suspicious"]   = df[f"{label}_shared_count"] >= threshold

    return df


def get_shared_attribute_groups(df: pd.DataFrame, attribute: str) -> dict:
    """
    Returns {attribute_value: [applicant_ids]} for values shared by >= threshold.
    attribute: 'phone' | 'email' | 'device' | 'address'
    """
    col_map = {
        "phone":   ("phone_norm",   SHARED_PHONE_THRESHOLD),
        "email":   ("email_norm",   SHARED_EMAIL_THRESHOLD),
        "device":  ("device_id",    SHARED_DEVICE_THRESHOLD),
        "address": ("address_norm", SHARED_ADDRESS_THRESHOLD),
    }
    if attribute not in col_map:
        return {}
    col, threshold = col_map[attribute]

    if "phone_norm" not in df.columns:
        df = detect_duplicates(df)

    groups = {}
    for val, grp in df.groupby(col):
        if len(grp) >= threshold and val not in ("", None):
            groups[val] = grp["applicant_id"].tolist()
    return groups


def get_all_shared_groups(df: pd.DataFrame) -> dict:
    """Return shared groups for all four key attributes."""
    if "phone_norm" not in df.columns:
        df = detect_duplicates(df)
    return {
        attr: get_shared_attribute_groups(df, attr)
        for attr in ("phone", "email", "device", "address")
    }


# ══════════════════════════════════════════════════════════════════════════════
# Richer attribute index — used by build_relationship_graph in graph_engine
# ══════════════════════════════════════════════════════════════════════════════

def build_attribute_index(df: pd.DataFrame) -> dict:
    """
    Build a comprehensive attribute index from an applicant DataFrame.

    Returns a dict with:
      "reuse"        – list of {attr_type, normalized_value, display_value,
                                applicant_ids} for attributes shared by >= 2
      "by_applicant" – {applicant_id: {attr_type: normalized_value}}
      "display"      – {(attr_type, normalized_value): display_value}
    """
    if df is None or df.empty:
        return {"reuse": [], "by_applicant": {}, "display": {}}

    work = df.copy()
    work["phone_norm"]   = work["phone"].apply(normalise_phone)
    work["email_norm"]   = work["email"].apply(normalise_email)
    work["address_norm"] = work["address"].apply(normalise_address)

    attr_col_map = {
        "phone":   ("phone_norm",   "phone"),
        "email":   ("email_norm",   "email"),
        "device":  ("device_id",    "device_id"),
        "address": ("address_norm", "address"),
    }

    # by_applicant index: applicant_id → {attr_type: normalised_value}
    by_applicant: dict[str, dict[str, str]] = {}
    for _, row in work.iterrows():
        aid = str(row["applicant_id"])
        by_applicant[aid] = {
            "phone":   str(row["phone_norm"]).strip(),
            "email":   str(row["email_norm"]).strip(),
            "device":  str(row["device_id"]).strip(),
            "address": str(row["address_norm"]).strip(),
        }

    # display map: (attr_type, norm_value) → human-readable original value
    display: dict[tuple, str] = {}
    for _, row in work.iterrows():
        display[("phone",   str(row["phone_norm"]).strip())]   = str(row["phone"])
        display[("email",   str(row["email_norm"]).strip())]   = str(row["email"])
        display[("device",  str(row["device_id"]).strip())]    = str(row["device_id"])
        display[("address", str(row["address_norm"]).strip())] = str(row["address"])

    # reuse list: attrs shared by >= 2 applicants (no threshold floor—every reuse matters)
    reuse: list[dict] = []
    for attr_type, (norm_col, _) in attr_col_map.items():
        mapping: dict[str, list[str]] = {}
        for _, row in work.iterrows():
            val = str(row[norm_col]).strip()
            if not val or val.lower() in ("nan", "none", ""):
                continue
            mapping.setdefault(val, []).append(str(row["applicant_id"]))

        for norm_val, app_ids in mapping.items():
            if len(app_ids) >= 2:
                reuse.append({
                    "attr_type":        attr_type,
                    "normalized_value": norm_val,
                    "display_value":    display.get((attr_type, norm_val), norm_val),
                    "applicant_ids":    app_ids,
                })

    return {"reuse": reuse, "by_applicant": by_applicant, "display": display}


def pair_links(index: dict) -> dict:
    """
    Derive pairwise applicant links from an attribute index.

    Returns {(applicant_id_a, applicant_id_b): [attr_type, ...]}
    where the tuple is always in sorted order (a < b).
    """
    pairs: dict[tuple, list[str]] = {}
    for item in index["reuse"]:
        ids   = item["applicant_ids"]
        atype = item["attr_type"]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                if key not in pairs:
                    pairs[key] = []
                if atype not in pairs[key]:
                    pairs[key].append(atype)
    return pairs


def shares_for(applicant_id: str, index: dict) -> dict:
    """
    Return {attr_type: [other_applicant_ids]} for all attributes that
    `applicant_id` shares with at least one other applicant.
    """
    result: dict[str, list[str]] = {}
    for item in index["reuse"]:
        if applicant_id not in item["applicant_ids"]:
            continue
        others = [a for a in item["applicant_ids"] if a != applicant_id]
        if others:
            atype = item["attr_type"]
            result.setdefault(atype, [])
            for o in others:
                if o not in result[atype]:
                    result[atype].append(o)
    return result
