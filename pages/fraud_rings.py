"""
Fraud Ring Detection page – Module 9.
Discover, display, and explore suspicious connected applicant groups.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from modules.database_manager import get_all_applicants, save_fraud_rings, get_fraud_rings, init_db
from modules.graph_engine import build_graph
from modules.fraud_ring_detector import detect_fraud_rings, get_ring_summary
from config import RISK_COLORS


def _ring_subgraph_figure(G: nx.Graph, member_ids: list, ring_label: str) -> go.Figure:
    """Build a plotly figure for a single ring's subgraph."""
    sub_nodes = set(member_ids)
    for m in member_ids:
        if G.has_node(m):
            for n in G.neighbors(m):
                sub_nodes.add(n)
    SG = G.subgraph(sub_nodes)
    if SG.number_of_nodes() == 0:
        return go.Figure()
    pos = nx.spring_layout(SG, seed=42, k=2.0)

    edge_x, edge_y = [], []
    for u, v in SG.edges():
        x0, y0 = pos.get(u, (0, 0))
        x1, y1 = pos.get(v, (0, 0))
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    app_x, app_y, app_text, app_colors, app_hover = [], [], [], [], []
    att_x, att_y, att_hover = [], [], []

    for node, data in SG.nodes(data=True):
        x, y = pos.get(node, (0, 0))
        if data.get("node_type") == "applicant":
            cat = data.get("risk_category", "Low")
            app_x.append(x); app_y.append(y)
            app_text.append(data.get("full_name", node)[:18])
            app_colors.append(RISK_COLORS.get(cat, "#888"))
            app_hover.append(
                f"<b>{data.get('full_name','')}</b><br>"
                f"ID: {node}<br>Risk: {data.get('risk_score',0):.0f} ({cat})"
            )
        else:
            att_x.append(x); att_y.append(y)
            att_hover.append(
                f"<b>{data.get('attr_type','').upper()}</b>: {data.get('value','')}"
            )

    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, mode="lines",
                       line=dict(width=1.2, color="rgba(100,160,255,0.4)"),
                       hoverinfo="none"),
            go.Scatter(x=att_x, y=att_y, mode="markers",
                       marker=dict(size=10, symbol="diamond",
                                   color="rgba(255,180,0,0.8)",
                                   line=dict(width=1, color="#ffb300")),
                       hovertext=att_hover, hoverinfo="text", name="Shared Attr"),
            go.Scatter(x=app_x, y=app_y, mode="markers+text",
                       marker=dict(size=14, color=app_colors,
                                   line=dict(width=1.5, color="#fff")),
                       text=app_text, textposition="top center",
                       textfont=dict(size=8, color="#c0d0e0"),
                       hovertext=app_hover, hoverinfo="text", name="Applicants"),
        ],
        layout=go.Layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
            title=dict(text=ring_label, font=dict(color="#00d4ff", size=14)),
            showlegend=True,
            legend=dict(bgcolor="rgba(10,14,26,0.8)", bordercolor="#1e90ff",
                        borderwidth=1, font=dict(color="#c0d0e0")),
            margin=dict(l=0, r=0, t=36, b=0), height=420,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    )
    return fig


def render():
    st.title("🔴 Fraud Ring Detection")
    st.markdown("Discover suspicious connected groups of applicants through graph analysis.")
    st.markdown("---")

    init_db()
    df = get_all_applicants()

    if df.empty:
        st.info("No applicant data. Generate a dataset first.")
        return

    col_detect, _ = st.columns([1, 3])
    with col_detect:
        if st.button("🔍 Detect Fraud Rings Now", use_container_width=True):
            with st.spinner("Building graph and detecting rings..."):
                G     = build_graph(df)
                rings = detect_fraud_rings(df, G)
                save_fraud_rings(rings)
                st.session_state["rings"] = rings
                st.session_state["G"]     = G
                st.success(f"✅ Detected {len(rings)} suspicious group(s).")

    rings = st.session_state.get("rings") or get_fraud_rings()
    G     = st.session_state.get("G")

    if not rings:
        st.info("No fraud rings detected yet. Click **Detect Fraud Rings Now**.")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    st.subheader(f"📋 Detected Groups ({len(rings)})")
    summary_df = get_ring_summary(rings)
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True)

    # ── KPIs ──────────────────────────────────────────────────────────────
    total_members   = sum(r.get("member_count", 0) for r in rings)
    critical_rings  = sum(1 for r in rings if r.get("risk_level") == "Critical")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Groups",          len(rings))
    c2.metric("Total Members",         total_members)
    c3.metric("Critical-Risk Groups",  critical_rings)
    st.markdown("---")

    # ── Per-ring detail ────────────────────────────────────────────────────
    st.subheader("🔎 Explore Individual Groups")
    ring_options = [r.get("ring_label", f"Ring-{i+1}") for i, r in enumerate(rings)]
    selected_ring_label = st.selectbox("Select a group to investigate", ring_options)
    selected_ring = next(
        (r for r in rings if r.get("ring_label") == selected_ring_label), None
    )

    if selected_ring:
        member_ids  = selected_ring.get("member_ids", [])
        shared_attrs= selected_ring.get("shared_attrs", {})
        risk_level  = selected_ring.get("risk_level", "")
        reasons     = selected_ring.get("reasons", [])

        color = RISK_COLORS.get(risk_level, "#888")
        st.markdown(
            f"<div style='background:#111d2e;border:2px solid {color};"
            f"border-radius:10px;padding:16px 20px;margin-bottom:12px'>"
            f"<b style='color:{color};font-size:1.1rem'>{selected_ring_label} – "
            f"{risk_level} Risk</b><br>"
            f"Members: {len(member_ids)} | "
            f"Avg Risk: {selected_ring.get('avg_risk_score',0):.1f}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Reasons
        if reasons:
            st.markdown("**Why this group is suspicious:**")
            for r in reasons:
                st.markdown(f"  • {r}")

        # Shared attributes
        if shared_attrs:
            st.markdown("**Shared Attributes:**")
            for attr_type, values in shared_attrs.items():
                st.markdown(f"  `{attr_type}`: {', '.join(str(v) for v in values[:5])}")

        # Member table
        member_df = df[df["applicant_id"].isin(member_ids)]
        if not member_df.empty:
            st.markdown("**Members:**")
            cols = ["applicant_id","full_name","risk_score","risk_category",
                    "applicant_type","application_status"]
            cols = [c for c in cols if c in member_df.columns]
            st.dataframe(member_df[cols].reset_index(drop=True), use_container_width=True)

        # Subgraph visualisation
        if G is not None:
            st.subheader(f"🗺️ Subgraph – {selected_ring_label}")
            fig = _ring_subgraph_figure(G, member_ids, selected_ring_label)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Re-run detection to view graph visualisation.")

        # Important disclaimer
        st.info(
            "ℹ️ **Note:** Shared attributes indicate investigation priority only when "
            "supported by other suspicious signals. Not every connected applicant is fraudulent. "
            "Human review is always recommended before any action."
        )
