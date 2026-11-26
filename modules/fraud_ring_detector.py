"""
Fraud Ring Detector – discovers suspicious connected groups via graph analysis.
"""

import pandas as pd
import networkx as nx
from modules.graph_engine import build_graph, get_applicant_components
from config import MIN_FRAUD_RING_SIZE


def detect_fraud_rings(df: pd.DataFrame, G: nx.Graph = None) -> list[dict]:
    """
    Identify suspicious connected groups.
    Returns list of ring dicts with member IDs, shared attributes, risk level.
    """
    if df.empty:
        return []

    if G is None:
        G = build_graph(df)

    components = get_applicant_components(G)
    rings = []

    for idx, app_set in enumerate(components):
        if len(app_set) < MIN_FRAUD_RING_SIZE:
            continue

        member_ids = list(app_set)

        # Gather shared attributes for this ring
        shared_attrs: dict[str, list] = {}
        for app_id in member_ids:
            for attr_node in G.neighbors(app_id):
                nd = G.nodes[attr_node]
                if nd.get("node_type") != "attribute":
                    continue
                attr_type = nd.get("attr_type", "")
                value     = nd.get("value", "")
                key       = f"{attr_type}::{value}"
                peers = [
                    n for n in G.neighbors(attr_node)
                    if G.nodes[n].get("node_type") == "applicant"
                       and n in app_set
                ]
                if len(peers) >= 2:
                    shared_attrs.setdefault(attr_type, [])
                    if value not in shared_attrs[attr_type]:
                        shared_attrs[attr_type].append(value)

        # Risk level based on group size and attribute types
        n = len(member_ids)
        high_risk_attrs = {"phone", "email", "device"}
        has_high = bool(high_risk_attrs & set(shared_attrs.keys()))
        if n >= 5 or (n >= 3 and has_high):
            risk_level = "Critical"
        elif n >= 3 or has_high:
            risk_level = "High"
        else:
            risk_level = "Medium"

        # Average risk score of members
        member_df = df[df["applicant_id"].isin(member_ids)]
        avg_risk   = round(member_df["risk_score"].mean(), 1) if not member_df.empty else 0.0

        # Build explanation
        reasons = []
        for attr_type, values in shared_attrs.items():
            reasons.append(f"Shared {attr_type}: {', '.join(str(v) for v in values[:3])}")

        rings.append({
            "ring_id":    idx + 1,
            "ring_label": f"Ring-{idx + 1:03d}",
            "member_ids": member_ids,
            "member_count": n,
            "shared_attrs": shared_attrs,
            "risk_level": risk_level,
            "avg_risk_score": avg_risk,
            "reasons": reasons,
        })

    # Sort by member count descending
    rings.sort(key=lambda r: r["member_count"], reverse=True)
    return rings


def get_ring_summary(rings: list[dict]) -> pd.DataFrame:
    """Return a summary DataFrame of all rings."""
    rows = []
    for r in rings:
        attr_summary = "; ".join(
            f"{k}: {len(v)}" for k, v in r.get("shared_attrs", {}).items()
        )
        rows.append({
            "Ring ID":       r["ring_label"],
            "Members":       r["member_count"],
            "Risk Level":    r["risk_level"],
            "Avg Risk Score": r["avg_risk_score"],
            "Shared Attributes": attr_summary,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()
