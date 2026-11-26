"""
Risk Analysis page – Module 6.
Full explainable risk scoring with configurable weights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.database_manager import get_all_applicants, update_applicant_risk, init_db
from modules.data_processor import clean_dataframe
from modules.duplicate_detector import detect_duplicates
from modules.risk_scoring import score_dataframe, score_single, apply_ring_membership_bonus
from config import RISK_COLORS, RISK_WEIGHTS


def render():
    st.title("📊 Risk Analysis")
    st.markdown("Calculate and explore explainable risk scores for all applicants.")
    st.markdown("---")

    init_db()
    df = get_all_applicants()

    if df.empty:
        st.info("No applicant data. Generate a dataset first.")
        return

    # ── Run / Re-run Scoring ──────────────────────────────────────────────
    st.subheader("⚙️ Scoring Configuration")
    with st.expander("Adjust Risk Weights (prototype thresholds)", expanded=False):
        weights = {}
        c1, c2 = st.columns(2)
        items = list(RISK_WEIGHTS.items())
        for i, (k, v) in enumerate(items):
            col = c1 if i < len(items) // 2 else c2
            with col:
                weights[k] = st.slider(
                    k.replace("_"," ").title(), 0, 50, v, 1, key=f"w_{k}"
                )
        st.caption("Changes apply only within this session.")

    if st.button("🔄 Re-score All Applicants", use_container_width=True):
        with st.spinner("Scoring applicants..."):
            df = clean_dataframe(df)
            df = detect_duplicates(df)
            df = score_dataframe(df)
            # Apply fraud-ring membership bonus using saved rings
            from modules.database_manager import get_fraud_rings
            from modules.risk_scoring import apply_ring_membership_bonus
            rings = get_fraud_rings()
            if rings:
                df = apply_ring_membership_bonus(df, rings)
            # Persist scores back to DB
            for _, row in df.iterrows():
                update_applicant_risk(
                    row["applicant_id"],
                    row["risk_score"],
                    row["risk_category"],
                    row.get("risk_flags", []),
                    row.get("risk_explanation", ""),
                )
            st.session_state["scored_df"] = df
            st.success(f"✅ Scored {len(df)} applicants.")

    scored_df = st.session_state.get("scored_df", df)

    # ── Summary KPIs ───────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    for cat, col in zip(["Low","Medium","High","Critical"], [c1,c2,c3,c4]):
        n = len(scored_df[scored_df["risk_category"] == cat])
        col.metric(f"{cat} Risk", n)

    # ── Charts ─────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Score Distribution")
        fig = px.histogram(
            scored_df, x="risk_score", nbins=25,
            color_discrete_sequence=["#1e90ff"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Risk Category by Applicant Type")
        cross = scored_df.groupby(["applicant_type","risk_category"]).size().reset_index(name="Count")
        fig2 = px.bar(
            cross, x="applicant_type", y="Count", color="risk_category",
            color_discrete_map=RISK_COLORS,
            barmode="stack", template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10,b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Score vs Income scatter ────────────────────────────────────────────
    st.subheader("Risk Score vs Annual Income")
    fig3 = px.scatter(
        scored_df, x="annual_income", y="risk_score",
        color="risk_category", color_discrete_map=RISK_COLORS,
        hover_data=["applicant_id","full_name","applicant_type"],
        template="plotly_dark",
        labels={"annual_income": "Annual Income (₹)", "risk_score": "Risk Score"},
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10,b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Filtered Table ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔎 Detailed Risk Table")
    min_score = st.slider("Minimum Risk Score", 0, 100, 0)
    filtered = scored_df[scored_df["risk_score"] >= min_score].sort_values(
        "risk_score", ascending=False
    )
    display_cols = ["applicant_id","full_name","applicant_type",
                    "risk_score","risk_category","application_status"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.markdown(f"Showing **{len(filtered)}** applicants")
    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True)

    # ── Individual Explanation ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧾 Individual Risk Explanation")
    ids = scored_df["applicant_id"].tolist()
    sel = st.selectbox("Select Applicant", ids, key="risk_sel")
    if sel:
        row = scored_df[scored_df["applicant_id"] == sel].iloc[0]
        score = row.get("risk_score", 0)
        cat   = row.get("risk_category", "Low")
        color = RISK_COLORS.get(cat, "#888")
        st.markdown(
            f"### <span style='color:{color}'>{score:.0f}/100 – {cat} Risk</span>",
            unsafe_allow_html=True,
        )
        flags = row.get("risk_flags", [])
        if isinstance(flags, str):
            import json
            try: flags = json.loads(flags)
            except: flags = []
        if flags:
            st.markdown("**Risk Flags:**")
            for f in flags: st.markdown(f"  • {f}")
        expl = row.get("risk_explanation", "")
        if expl:
            st.code(expl, language=None)
        else:
            st.info("No detailed explanation. Re-score applicants first.")
