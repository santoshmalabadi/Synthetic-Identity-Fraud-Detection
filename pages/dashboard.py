"""
Dashboard page – Module 1.
Shows summary KPIs, interactive charts, and system overview.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.database_manager import get_all_applicants, get_fraud_rings
from config import RISK_COLORS


def _css():
    st.markdown("""
    <style>
    .kpi-card{background:#111d2e;border:1px solid rgba(30,144,255,.18);
      border-radius:10px;padding:18px 22px;text-align:center;margin-bottom:4px}
    .kpi-val{font-size:2.2rem;font-weight:800;color:#00d4ff}
    .kpi-lbl{font-size:.75rem;text-transform:uppercase;letter-spacing:.07em;
      color:#8899aa;margin-top:4px}
    </style>""", unsafe_allow_html=True)


def _kpi(label, value, colour="#00d4ff"):
    return f"""<div class="kpi-card">
      <div class="kpi-val" style="color:{colour}">{value}</div>
      <div class="kpi-lbl">{label}</div></div>"""


def render():
    _css()
    st.title("🛡️ Synthetic Identity Fraud Detection")
    st.markdown("**AI-powered cybersecurity platform for digital banking applicant analysis**")
    st.markdown("---")

    df = get_all_applicants()

    if df.empty:
        st.info("📋 No applicant data loaded yet. Go to **Generate Synthetic Data** to create a dataset.")
        st.markdown("""
        ### How This Platform Works
        1. **Generate** synthetic applicant records with realistic profiles
        2. **Detect** suspicious attribute sharing and identity inconsistencies  
        3. **Analyse** relationships with the NetworkX graph engine
        4. **Discover** fraud rings through connected component analysis
        5. **Score** each applicant with an explainable 0-100 risk score
        6. **Evaluate** detection performance against ground-truth labels
        """)
        return

    # ── KPI Row ───────────────────────────────────────────────────────────
    total      = len(df)
    normal_n   = len(df[df["risk_category"] == "Low"])
    suspicious = len(df[df["risk_category"].isin(["Medium"])])
    high_risk  = len(df[df["risk_category"].isin(["High", "Critical"])])
    avg_score  = round(df["risk_score"].mean(), 1) if "risk_score" in df.columns else 0
    rings      = get_fraud_rings()
    n_rings    = len(rings)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.markdown(_kpi("Total Applicants", total), unsafe_allow_html=True)
    with c2: st.markdown(_kpi("Normal / Low Risk", normal_n, "#00c896"), unsafe_allow_html=True)
    with c3: st.markdown(_kpi("Medium Risk", suspicious, "#ffb300"), unsafe_allow_html=True)
    with c4: st.markdown(_kpi("High / Critical", high_risk, "#ff4c4c"), unsafe_allow_html=True)
    with c5: st.markdown(_kpi("Fraud Rings", n_rings, "#ff7700"), unsafe_allow_html=True)
    with c6: st.markdown(_kpi("Avg Risk Score", f"{avg_score}/100", "#1e90ff"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Risk Level Distribution")
        risk_counts = df["risk_category"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        order = ["Low", "Medium", "High", "Critical"]
        risk_counts["Risk Level"] = pd.Categorical(risk_counts["Risk Level"], categories=order, ordered=True)
        risk_counts = risk_counts.sort_values("Risk Level")
        colors_list = [RISK_COLORS.get(r, "#888") for r in risk_counts["Risk Level"]]
        fig = px.bar(
            risk_counts, x="Risk Level", y="Count",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🍩 Applicant Type Breakdown")
        type_counts = df["applicant_type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig2 = px.pie(
            type_counts, names="Type", values="Count",
            color_discrete_sequence=["#00c896","#ff4c4c","#ffb300","#1e90ff","#00d4ff"],
            template="plotly_dark",
            hole=0.45,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📈 Risk Score Distribution")
        fig3 = px.histogram(
            df, x="risk_score", nbins=20,
            color_discrete_sequence=["#1e90ff"],
            template="plotly_dark",
            labels={"risk_score": "Risk Score"},
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("📅 Applications Over Time")
        if "application_date" in df.columns:
            df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
            timeline = df.dropna(subset=["application_date"])
            timeline = timeline.groupby(timeline["application_date"].dt.to_period("M")).size().reset_index()
            timeline.columns = ["Month", "Count"]
            timeline["Month"] = timeline["Month"].astype(str)
            fig4 = px.line(
                timeline, x="Month", y="Count",
                template="plotly_dark",
                color_discrete_sequence=["#00d4ff"],
                markers=True,
            )
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No application date data available.")

    # ── Shared Attribute Summary ──────────────────────────────────────────
    st.subheader("🔗 Most Frequently Shared Attributes")
    shared_cols = {
        "phone_suspicious": "Shared Phone",
        "email_suspicious": "Shared Email",
        "device_suspicious": "Shared Device",
        "address_suspicious": "Shared Address",
    }
    present = {v: df[k].sum() for k, v in shared_cols.items() if k in df.columns}
    if present:
        attr_df = pd.DataFrame(list(present.items()), columns=["Attribute", "Count"])
        fig5 = px.bar(
            attr_df, x="Attribute", y="Count",
            color="Attribute",
            color_discrete_sequence=["#1e90ff","#00d4ff","#ffb300","#00c896"],
            template="plotly_dark",
        )
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Run **Risk Analysis** first to populate shared-attribute flags.")

    # ── Recent High-Risk Applicants ───────────────────────────────────────
    st.subheader("🚨 Recent High / Critical Risk Applicants")
    high_df = df[df["risk_category"].isin(["High", "Critical"])].sort_values(
        "risk_score", ascending=False
    ).head(10)
    if not high_df.empty:
        display_cols = ["applicant_id", "full_name", "risk_score", "risk_category",
                        "applicant_type", "application_status"]
        display_cols = [c for c in display_cols if c in high_df.columns]
        st.dataframe(high_df[display_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.success("✅ No high-risk applicants detected in current dataset.")

    # ── Fraud Ring Summary ────────────────────────────────────────────────
    if rings:
        st.subheader(f"🔴 Detected Fraud Rings ({n_rings})")
        ring_rows = []
        for r in rings:
            ring_rows.append({
                "Ring ID":    r.get("ring_label",""),
                "Members":    len(r.get("member_ids", [])),
                "Risk Level": r.get("risk_level",""),
                "Shared Attrs": ", ".join(r.get("shared_attrs", {}).keys()),
            })
        st.dataframe(pd.DataFrame(ring_rows), use_container_width=True)
