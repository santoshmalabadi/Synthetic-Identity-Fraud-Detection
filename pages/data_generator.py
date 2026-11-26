"""
Data Generator page – Module 2.
Generate synthetic applicant datasets with configurable counts.
"""

import streamlit as st
import pandas as pd
import os
from modules.data_generator import (
    generate_dataset,
    scenario_synthetic_identity,
    scenario_fraud_ring,
    scenario_genuine_family,
    scenario_new_identity,
)
from modules.data_processor import clean_dataframe
from modules.duplicate_detector import detect_duplicates
from modules.risk_scoring import score_dataframe, apply_ring_membership_bonus
from modules.graph_engine import build_graph
from modules.fraud_ring_detector import detect_fraud_rings
from modules.database_manager import insert_applicants, delete_all_applicants, init_db, save_fraud_rings
from config import (
    DEFAULT_NORMAL_COUNT, DEFAULT_SYNTHETIC_COUNT,
    DEFAULT_FRAUD_RING_COUNT, DEFAULT_NEW_COUNT, GENERATED_DIR,
)


def render():
    st.title("⚙️ Synthetic Applicant Data Generator")
    st.markdown("Generate realistic synthetic banking applicant records for fraud detection testing.")
    st.markdown("---")

    # ── Main Generation ───────────────────────────────────────────────────
    st.subheader("📋 Configure Dataset")
    col1, col2 = st.columns(2)
    with col1:
        n_normal   = st.slider("Normal Applicants",          10, 200, DEFAULT_NORMAL_COUNT,      5)
        n_synth    = st.slider("Synthetic Identity Applicants", 5, 100, DEFAULT_SYNTHETIC_COUNT,  5)
    with col2:
        n_rings    = st.slider("Fraud-Ring Applicants",        5, 100, DEFAULT_FRAUD_RING_COUNT,  5)
        n_new      = st.slider("New / Isolated Applicants",    1,  30, DEFAULT_NEW_COUNT,          1)

    seed = st.number_input("Random Seed (for reproducibility)", value=42, min_value=0)
    total_planned = n_normal + n_synth + n_rings + n_new
    st.info(f"🔢 Total applicants to generate: **{total_planned}**")

    col_gen, col_clear = st.columns(2)

    with col_gen:
        if st.button("🚀 Generate & Save Dataset", use_container_width=True):
            with st.spinner("Generating synthetic applicants..."):
                init_db()
                df = generate_dataset(
                    normal_count=n_normal,
                    synthetic_count=n_synth,
                    fraud_ring_count=n_rings,
                    new_count=n_new,
                    seed=int(seed),
                )
                df = clean_dataframe(df)
                df = detect_duplicates(df)
                df = score_dataframe(df)

                # Graph + ring detection for ring-membership bonus
                G     = build_graph(df)
                rings = detect_fraud_rings(df, G)
                df    = apply_ring_membership_bonus(df, rings)
                save_fraud_rings(rings)

                # Persist
                insert_applicants(df)

                # Save CSV
                os.makedirs(GENERATED_DIR, exist_ok=True)
                csv_path = os.path.join(GENERATED_DIR, "applicants.csv")
                df.drop(columns=["phone_norm","email_norm","address_norm"],
                        errors="ignore").to_csv(csv_path, index=False)

                st.success(f"✅ Generated and saved {len(df)} applicants to database.")
                st.session_state["generated_df"] = df

    with col_clear:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            delete_all_applicants()
            st.session_state.pop("generated_df", None)
            st.warning("All applicant data cleared.")

    # ── Preview ───────────────────────────────────────────────────────────
    if "generated_df" in st.session_state:
        df = st.session_state["generated_df"]
        st.markdown("---")
        st.subheader(f"📊 Dataset Preview ({len(df)} applicants)")

        type_counts = df["applicant_type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        col3, col4 = st.columns([1, 2])
        with col3:
            st.dataframe(type_counts, use_container_width=True)
        with col4:
            risk_counts = df["risk_category"].value_counts().reset_index()
            risk_counts.columns = ["Risk Category", "Count"]
            st.dataframe(risk_counts, use_container_width=True)

        display_cols = ["applicant_id","full_name","applicant_type","risk_score",
                        "risk_category","application_status"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols].head(20).reset_index(drop=True), use_container_width=True)

        # Download
        csv_bytes = df.drop(columns=["phone_norm","email_norm","address_norm"],
                            errors="ignore").to_csv(index=False).encode()
        st.download_button(
            "📥 Download Full Dataset CSV",
            data=csv_bytes,
            file_name="synthetic_applicants.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Demo Scenarios ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎭 Ready-Made Demonstration Scenarios")
    st.markdown("Load a focused scenario instantly for hackathon presentation.")

    sc1, sc2, sc3, sc4 = st.columns(4)

    def _load_scenario(df_scenario, label):
        init_db()
        df_scenario = clean_dataframe(df_scenario)
        df_scenario = detect_duplicates(df_scenario)
        df_scenario = score_dataframe(df_scenario)
        G     = build_graph(df_scenario)
        rings = detect_fraud_rings(df_scenario, G)
        df_scenario = apply_ring_membership_bonus(df_scenario, rings)
        save_fraud_rings(rings)
        insert_applicants(df_scenario)
        st.session_state["scenario_df"] = df_scenario
        st.success(f"✅ Scenario '{label}' loaded ({len(df_scenario)} applicants)")

    with sc1:
        st.markdown("**Scenario 1**\nSynthetic Identity Combination")
        if st.button("▶ Load Scenario 1", use_container_width=True):
            _load_scenario(scenario_synthetic_identity(), "Synthetic Identity")

    with sc2:
        st.markdown("**Scenario 2**\nFraud Ring – Reused Attributes")
        if st.button("▶ Load Scenario 2", use_container_width=True):
            _load_scenario(scenario_fraud_ring(), "Fraud Ring")

    with sc3:
        st.markdown("**Scenario 3**\nGenuine Family – Shared Address")
        if st.button("▶ Load Scenario 3", use_container_width=True):
            _load_scenario(scenario_genuine_family(), "Genuine Family")

    with sc4:
        st.markdown("**Scenario 4**\nNew Identity – No Prior Matches")
        if st.button("▶ Load Scenario 4", use_container_width=True):
            _load_scenario(scenario_new_identity(), "New Identity")

    if "scenario_df" in st.session_state:
        sdf = st.session_state["scenario_df"]
        st.subheader("Scenario Preview")
        cols = ["applicant_id","full_name","applicant_type","risk_score",
                "risk_category","risk_explanation"]
        cols = [c for c in cols if c in sdf.columns]
        st.dataframe(sdf[cols].reset_index(drop=True), use_container_width=True)
