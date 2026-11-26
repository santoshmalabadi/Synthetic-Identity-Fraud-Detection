"""
Investigation Centre page – Module 8.
Analyst workflow: review, clear, escalate, and annotate suspicious applicants.
"""

import streamlit as st
import pandas as pd
from modules.database_manager import (
    get_all_applicants, update_applicant_status,
    get_investigation_log, init_db,
)
from config import ALL_STATUSES, RISK_COLORS


def render():
    st.title("🔬 Investigation Centre")
    st.markdown("Analyst workflow for reviewing, clearing, and escalating suspicious applicants.")
    st.markdown("---")

    init_db()
    df = get_all_applicants()

    if df.empty:
        st.info("No applicant data. Generate a dataset first.")
        return

    # ── Investigation Queue ───────────────────────────────────────────────
    st.subheader("🚨 Investigation Queue")
    queue_statuses = ["New", "Under Review", "Suspicious"]
    queue = df[df["application_status"].isin(queue_statuses)].sort_values(
        "risk_score", ascending=False
    )

    if queue.empty:
        st.success("✅ Investigation queue is empty – no pending cases.")
    else:
        st.markdown(f"**{len(queue)} case(s) awaiting review**")
        cols = ["applicant_id","full_name","risk_score","risk_category",
                "applicant_type","application_status"]
        cols = [c for c in cols if c in queue.columns]
        st.dataframe(queue[cols].reset_index(drop=True), use_container_width=True)

    # ── Case Review ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Review a Case")

    all_ids = df["applicant_id"].tolist()
    sel_id = st.selectbox("Select Applicant ID to Review", all_ids, key="inv_sel")

    if sel_id:
        row = df[df["applicant_id"] == sel_id].iloc[0]
        score = row.get("risk_score", 0)
        cat   = row.get("risk_category", "Low")
        color = RISK_COLORS.get(cat, "#888")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Name:** {row.get('full_name','')}")
            st.markdown(f"**Email:** {row.get('email','')}")
            st.markdown(f"**Phone:** {row.get('phone','')}")
            st.markdown(f"**Type:** {row.get('applicant_type','')}")
        with c2:
            st.markdown(
                f"**Risk Score:** <span style='color:{color}'>{score:.0f}/100 – {cat}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Current Status:** {row.get('application_status','')}")
            st.markdown(f"**Ground Truth:** {row.get('ground_truth_label','')}")

        # Risk flags
        flags = row.get("risk_flags", [])
        if isinstance(flags, str):
            import json
            try: flags = json.loads(flags)
            except: flags = []
        if flags:
            st.markdown("**Risk Flags:**")
            for f in flags:
                st.markdown(f"  • {f}")

        # Analyst action
        st.markdown("---")
        col_action, col_note = st.columns([1, 2])
        with col_action:
            new_status = st.selectbox(
                "Change Status To",
                ALL_STATUSES,
                index=ALL_STATUSES.index(
                    row.get("application_status", "New")
                ) if row.get("application_status","New") in ALL_STATUSES else 0,
            )
        with col_note:
            analyst_note = st.text_area(
                "Analyst Note / Reason",
                value=row.get("analyst_note",""),
                height=80,
            )

        col_btn1, col_btn2, _ = st.columns([1, 1, 2])
        with col_btn1:
            if st.button("💾 Save Review", use_container_width=True):
                update_applicant_status(sel_id, new_status, analyst_note)
                st.success(f"✅ Case {sel_id} updated to '{new_status}'.")
                st.rerun()
        with col_btn2:
            if st.button("🔄 Mark Under Review", use_container_width=True):
                update_applicant_status(sel_id, "Under Review", analyst_note)
                st.info(f"Case {sel_id} marked as Under Review.")
                st.rerun()

    # ── Status Summary ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Case Status Summary")
    status_counts = df["application_status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    st.dataframe(status_counts, use_container_width=True)

    import plotly.express as px
    fig = px.bar(
        status_counts, x="Status", y="Count",
        color="Status",
        color_discrete_sequence=["#1e90ff","#00d4ff","#ffb300","#ff4c4c","#00c896","#888"],
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, margin=dict(t=10,b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Investigation Log ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📜 Investigation Log")
    log_df = get_investigation_log()
    if not log_df.empty:
        st.dataframe(log_df.head(50).reset_index(drop=True), use_container_width=True)
    else:
        st.info("No investigation actions recorded yet.")

    # ── Confirmed Fraud Cases ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("✅ Analyst-Confirmed Cases")
    confirmed = df[df["application_status"] == "Confirmed Fraud"]
    cleared   = df[df["application_status"] == "Cleared"]

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"**Confirmed Fraud: {len(confirmed)}**")
        if not confirmed.empty:
            cols = ["applicant_id","full_name","risk_score","applicant_type"]
            cols = [c for c in cols if c in confirmed.columns]
            st.dataframe(confirmed[cols].reset_index(drop=True), use_container_width=True)
    with col4:
        st.markdown(f"**Cleared (False Positives): {len(cleared)}**")
        if not cleared.empty:
            cols = ["applicant_id","full_name","risk_score","analyst_note"]
            cols = [c for c in cols if c in cleared.columns]
            st.dataframe(cleared[cols].reset_index(drop=True), use_container_width=True)
