"""
Graph Engine – builds a bipartite applicant-attribute graph using NetworkX.

Also provides build_relationship_graph, a richer analysis graph that layers
applicant-to-applicant edges on top of the bipartite structure, exposing
connected components, pairwise links, and shared-attribute summaries.

Applicants are linked only when they share a phone, email, address,
device, or fictional identity reference. A link is a lead, not a verdict.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
from modules.data_processor import normalise_phone, normalise_email, normalise_address
from config import (
    SHARED_PHONE_THRESHOLD, SHARED_EMAIL_THRESHOLD,
    SHARED_DEVICE_THRESHOLD, SHARED_ADDRESS_THRESHOLD,
)


ATTR_THRESHOLDS = {
    "phone":   SHARED_PHONE_THRESHOLD,
    "email":   SHARED_EMAIL_THRESHOLD,
    "device":  SHARED_DEVICE_THRESHOLD,
    "address": SHARED_ADDRESS_THRESHOLD,
}


def build_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Build a bipartite graph:
      - Applicant nodes  (type='applicant')
      - Attribute nodes  (type='attribute', attr_type=phone|email|device|address)
    Edges connect an applicant to each of its key attributes when that
    attribute is shared by at least `threshold` applicants.
    """
    G = nx.Graph()

    if df.empty:
        return G

    # Normalise
    df = df.copy()
    df["phone_norm"]   = df["phone"].apply(normalise_phone)
    df["email_norm"]   = df["email"].apply(normalise_email)
    df["address_norm"] = df["address"].apply(normalise_address)

    attr_col_map = {
        "phone":   "phone_norm",
        "email":   "email_norm",
        "device":  "device_id",
        "address": "address_norm",
    }

    # Precompute value → applicant_ids for each attribute type
    attr_to_apps: dict[str, dict[str, list]] = {}
    for attr_type, col in attr_col_map.items():
        mapping: dict[str, list] = {}
        for _, row in df.iterrows():
            val = str(row[col]).strip()
            if val and val.lower() not in ("nan", "none", ""):
                mapping.setdefault(val, []).append(row["applicant_id"])
        attr_to_apps[attr_type] = mapping

    # Add applicant nodes
    for _, row in df.iterrows():
        aid = row["applicant_id"]
        G.add_node(
            aid,
            node_type="applicant",
            full_name=row.get("full_name", ""),
            risk_score=float(row.get("risk_score", 0)),
            risk_category=row.get("risk_category", "Low"),
            applicant_type=row.get("applicant_type", "normal"),
            application_status=row.get("application_status", "New"),
        )

    # Add attribute nodes and edges
    for attr_type, mapping in attr_to_apps.items():
        threshold = ATTR_THRESHOLDS[attr_type]
        for val, app_ids in mapping.items():
            if len(app_ids) >= threshold:
                attr_node = f"{attr_type}::{val}"
                G.add_node(
                    attr_node,
                    node_type="attribute",
                    attr_type=attr_type,
                    value=val,
                    shared_count=len(app_ids),
                )
                for aid in app_ids:
                    if G.has_node(aid):
                        G.add_edge(aid, attr_node, attr_type=attr_type, value=val)

    return G


def get_connected_components(G: nx.Graph) -> list[set]:
    """Return list of connected component node sets (applicants + attributes)."""
    return list(nx.connected_components(G))


def get_applicant_components(G: nx.Graph) -> list[set]:
    """Return components containing only applicant node IDs."""
    components = []
    for comp in nx.connected_components(G):
        app_nodes = {n for n in comp if G.nodes[n].get("node_type") == "applicant"}
        if len(app_nodes) >= 2:
            components.append(app_nodes)
    return components


def get_neighbors_info(G: nx.Graph, applicant_id: str) -> dict:
    """
    For a given applicant, return:
      - connected applicants (via shared attributes)
      - shared attribute details
    """
    if not G.has_node(applicant_id):
        return {"connected_applicants": [], "shared_attrs": []}

    shared_attrs = []
    connected_apps = set()

    for attr_node in G.neighbors(applicant_id):
        node_data = G.nodes[attr_node]
        if node_data.get("node_type") != "attribute":
            continue
        attr_type  = node_data.get("attr_type", "")
        value      = node_data.get("value", "")
        count      = node_data.get("shared_count", 1)
        peer_apps  = [
            n for n in G.neighbors(attr_node)
            if n != applicant_id and G.nodes[n].get("node_type") == "applicant"
        ]
        connected_apps.update(peer_apps)
        shared_attrs.append({
            "attr_type": attr_type,
            "value":     value,
            "shared_count": count,
            "peer_applicants": peer_apps,
        })

    return {
        "connected_applicants": list(connected_apps),
        "shared_attrs": shared_attrs,
    }


def compute_graph_stats(G: nx.Graph) -> dict:
    applicant_nodes = [n for n, d in G.nodes(data=True)
                       if d.get("node_type") == "applicant"]
    attribute_nodes = [n for n, d in G.nodes(data=True)
                       if d.get("node_type") == "attribute"]
    components      = get_applicant_components(G)
    return {
        "total_applicant_nodes": len(applicant_nodes),
        "total_attribute_nodes": len(attribute_nodes),
        "total_edges":           G.number_of_edges(),
        "connected_components":  len(components),
        "largest_component_size": max((len(c) for c in components), default=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Richer relationship graph (applicant-to-applicant + bipartite)
# ══════════════════════════════════════════════════════════════════════════════

def build_relationship_graph(df: pd.DataFrame) -> dict:
    """
    Build a comprehensive relationship graph from an applicant DataFrame.

    Returns a dict with:
      "index"      – attribute index (from duplicate_detector)
      "bipartite"  – nx.Graph (applicants + attribute nodes, any reuse >= 2)
      "applicants" – nx.Graph (applicants only, edges where attributes are shared)
      "names"      – {applicant_id: full_name}
      "pairs"      – {(a, b): [attr_type, ...]} pairwise links in sorted-id order
      "components" – list of sorted applicant-id lists for groups >= 2 members
    """
    from modules.duplicate_detector import build_attribute_index, pair_links

    index = build_attribute_index(df)
    names: dict[str, str] = {}

    bipartite = nx.Graph()
    if df is not None and not df.empty:
        for record in df.to_dict(orient="records"):
            aid   = str(record["applicant_id"])
            names[aid] = str(record.get("full_name") or aid)
            bipartite.add_node(
                aid,
                kind="applicant",
                node_type="applicant",
                label=names[aid],
                full_name=names[aid],
                risk_score=float(record.get("risk_score", 0)),
                risk_category=str(record.get("risk_category", "Low")),
                applicant_type=str(record.get("applicant_type", "normal")),
                application_status=str(record.get("application_status", "New")),
            )

    for item in index["reuse"]:
        attr_node = f"{item['attr_type']}::{item['normalized_value']}"
        bipartite.add_node(
            attr_node,
            kind="attribute",
            node_type="attribute",
            attr_type=item["attr_type"],
            label=item["display_value"],
            value=item["normalized_value"],
            display_value=item["display_value"],
            shared_count=len(item["applicant_ids"]),
        )
        for aid in item["applicant_ids"]:
            bipartite.add_edge(aid, attr_node, attr_type=item["attr_type"])

    # Applicant-only graph (edges = shared attributes)
    applicant_graph = nx.Graph()
    applicant_graph.add_nodes_from(names)
    pairs = pair_links(index)
    for (left, right), types in pairs.items():
        applicant_graph.add_edge(left, right, types=types)

    components = [
        sorted(component)
        for component in nx.connected_components(applicant_graph)
        if len(component) >= 2
    ]
    components.sort(key=lambda m: (-len(m), m[0] if m else ""))

    return {
        "index":      index,
        "bipartite":  bipartite,
        "applicants": applicant_graph,
        "names":      names,
        "pairs":      pairs,
        "components": components,
    }


def connected_details(applicant_id: str, graph_result: dict) -> list[dict]:
    """
    Return a list of dicts describing every applicant connected to the given
    applicant_id via shared attributes.

    Each dict has: applicant_id, full_name, via (list of attr_types).
    """
    details  = []
    pairs    = graph_result["pairs"]
    names    = graph_result["names"]
    for (left, right), types in pairs.items():
        if applicant_id not in (left, right):
            continue
        other = right if left == applicant_id else left
        details.append({
            "applicant_id": other,
            "full_name":    names.get(other, other),
            "via":          types,
        })
    details.sort(key=lambda item: item["applicant_id"])
    return details


def suspicious_attribute_rows(applicant_id: str, graph_result: dict) -> list[dict]:
    """
    Return rows describing which attributes of applicant_id are shared
    with other applicants.

    Each row: type, value, shared_with (list of ids), count.
    """
    from modules.duplicate_detector import shares_for
    shares = shares_for(applicant_id, graph_result["index"])
    rows   = []
    for attr_type, others in shares.items():
        if not others:
            continue
        normalized = (
            graph_result["index"]["by_applicant"]
            .get(applicant_id, {})
            .get(attr_type, "")
        )
        display = graph_result["index"]["display"].get(
            (attr_type, normalized), normalized
        )
        rows.append({
            "type":        attr_type,
            "value":       display,
            "shared_with": others,
            "count":       len(others),
        })
    return rows


def component_attribute_summary(
    member_ids: list[str], graph_result: dict
) -> list[dict]:
    """
    For a connected component (list of applicant IDs), return a summary of
    all attributes shared by >= 2 members.

    Each entry: type, value, count, applicant_ids.
    """
    member_set = set(member_ids)
    summary    = []
    for item in graph_result["index"]["reuse"]:
        holders = [
            aid for aid in item["applicant_ids"] if aid in member_set
        ]
        if len(holders) < 2:
            continue
        summary.append({
            "type":          item["attr_type"],
            "value":         item["display_value"],
            "count":         len(holders),
            "applicant_ids": holders,
        })
    summary.sort(key=lambda item: (-item["count"], item["type"]))
    return summary
