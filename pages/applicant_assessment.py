"""
New Applicant Assessment page – Module 7.
Real-time risk assessment for a new applicant submission.
"""

import streamlit as st
import pandas as pd
from faker import Faker
from modules.database_manager import get_all_applicants, upsert_applicant, init_db
from modules.risk_scoring import score_single
from modules.identity_analyzer import analyse_identity
from modules.duplicate_detector import detect_duplicates
from modules.validation import validate_applicant_dict
from modules.data_generator import _applicant_id, _id_ref, _device_id
from modules.fraud_explainer import generate_fraud_explanation
from config import RISK_COLORS

fake = Faker("en_IN")


def _risk_banner(category: str, score: float):
    color = RISK_COLORS.get(category, "#888")
    icons = {"Low": "✅", "Medium": "⚠️", "High": "🔶", "Critical": "🚨"}
    icon  = icons.get(category, "ℹ️")
    recommendations = {
        "Low":      "No immediate action required. Continue standard verification.",
        "Medium":   "Request additional identity verification documents.",
        "High":     "Escalate to fraud analyst for manual review before approval.",
        "Critical": "Immediate analyst review required. Do not process until cleared.",
    }
    rec = recommendations.get(category, "")
    st.markdown(f"""
    <div style="background:#111d2e;border:2px solid {color};border-radius:12px;
                padding:20px 24px;margin-bottom:16px;">
      <div style="font-size:1.6rem;font-weight:800;color:{color}">
        {icon} {category.upper()} RISK — Score: {score:.0f}/100
      </div>
      <div style="color:#8899aa;margin-top:8px">{rec}</div>
    </div>
    """, unsafe_allow_html=True)


def _fraud_explanation_card(applicant: dict, score_result: dict, df_context, app_id: str, name: str):
    """Display the 'Why Was This Applicant Flagged?' explanation card."""
    category = score_result.get("risk_category", "Low")
    score    = float(score_result.get("risk_score", 0))
    color    = RISK_COLORS.get(category, "#888")
    icons    = {"Low": "✅", "Medium": "⚠️", "High": "🔶", "Critical": "🚨"}
    icon     = icons.get(category, "ℹ️")

    explanation = generate_fraud_explanation(
        pd.Series(applicant), score_result, df_context
    )

    st.markdown(f"""
    <div style="background:#111d2e;border:1px solid {color}44;
                border-left:5px solid {color};border-radius:12px;
                padding:20px 24px;margin-top:16px;margin-bottom:16px">
      <div style="font-size:1rem;font-weight:700;color:{color};margin-bottom:12px">
        {icon} Why Was This Applicant Flagged?
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
                  gap:12px;margin-bottom:14px">
        <div>
          <div style="font-size:.68rem;color:#8899aa;text-transform:uppercase">Applicant ID</div>
          <div style="font-size:.85rem;color:#d0dcea;font-weight:600">{app_id}</div>
        </div>
        <div>
          <div style="font-size:.68rem;color:#8899aa;text-transform:uppercase">Name</div>
          <div style="font-size:.85rem;color:#d0dcea;font-weight:600">{name}</div>
        </div>
        <div>
          <div style="font-size:.68rem;color:#8899aa;text-transform:uppercase">Risk Score</div>
          <div style="font-size:.85rem;color:{color};font-weight:800">{score:.0f}/100</div>
        </div>
        <div>
          <div style="font-size:.68rem;color:#8899aa;text-transform:uppercase">Risk Level</div>
          <div style="font-size:.85rem;font-weight:700;
               background:{color}22;color:{color};border:1px solid {color};
               border-radius:5px;padding:1px 10px;display:inline-block">{category}</div>
        </div>
      </div>
      <div style="background:#0a1628;border-radius:8px;padding:14px 16px;
                  font-size:.88rem;color:#c0d0e0;line-height:1.65">
        {explanation}
      </div>
    </div>
    """, unsafe_allow_html=True)


def render():
    st.title("🔍 New Applicant Risk Assessment")
    st.markdown("Submit a new applicant's details to receive an instant explainable risk score.")
    st.markdown("---")

    init_db()
    existing_df = get_all_applicants()

    with st.form("new_applicant_form"):
        st.subheader("📋 Applicant Information")
        c1, c2 = st.columns(2)
        with c1:
            full_name     = st.text_input("Full Name *")
            dob           = st.text_input("Date of Birth (YYYY-MM-DD) *", value="1990-01-01")
            email         = st.text_input("Email Address *")
            phone         = st.text_input("Phone Number *")
        with c2:
            address       = st.text_input("Residential Address *")
            city          = st.text_input("City")
            state         = st.text_input("State")
            pincode       = st.text_input("Pincode")

        st.subheader("💼 Employment & Income")
        c3, c4 = st.columns(2)
        with c3:
            employer      = st.text_input("Employer / Organisation")
            emp_type      = st.selectbox("Employment Type",
                           ["Salaried","Self-Employed","Contract","Unemployed"])
        with c4:
            annual_income = st.number_input("Annual Income (₹)", min_value=0.0,
                                             value=0.0, step=10000.0)
            app_date      = st.text_input("Application Date",
                                           value=pd.Timestamp.now().strftime("%Y-%m-%d"))

        submitted = st.form_submit_button("🔎 Assess Risk Now", use_container_width=True)

    if submitted:
        record = {
            "applicant_id":    _applicant_id(),
            "full_name":       full_name,
            "dob":             dob,
            "email":           email,
            "phone":           phone,
            "address":         address,
            "city":            city,
            "state":           state,
            "pincode":         pincode,
            "id_ref":          _id_ref(),
            "device_id":       _device_id(),
            "application_date": app_date,
            "employer":        employer,
            "employment_type": emp_type,
            "annual_income":   annual_income,
            "application_status": "New",
            "ground_truth_label": "genuine",
            "applicant_type":  "normal",
            "risk_score":      0.0,
            "risk_category":   "Low",
            "risk_flags":      [],
            "risk_explanation": "",
            "analyst_note":    "",
        }

        valid, errors, warnings = validate_applicant_dict(record)
        for w in warnings:
            st.warning(w)
        if not valid:
            for e in errors:
                st.error(e)
            return

        st.markdown("---")
        st.subheader("📊 Assessment Results")

        # Score against existing records
        result = score_single(pd.Series(record), existing_df)
        record.update({
            "risk_score":       result["risk_score"],
            "risk_category":    result["risk_category"],
            "risk_flags":       result["risk_flags"],
            "risk_explanation": result["risk_explanation"],
        })

        _risk_banner(result["risk_category"], result["risk_score"])

        # Fraud Detection Explanation
        _fraud_explanation_card(
            record, result, existing_df,
            record["applicant_id"], full_name
        )

        # Detail tabs
        tab1, tab2, tab3 = st.tabs(["Risk Flags", "Identity Checks", "Database Matches"])

        with tab1:
            flags = result.get("risk_flags", [])
            if flags:
                for f in flags:
                    st.markdown(f"• {f}")
            else:
                st.success("No suspicious flags detected.")

            st.markdown("**Score Breakdown:**")
            score_parts = result.get("score_parts", {})
            if score_parts:
                for k, v in score_parts.items():
                    st.markdown(f"  `{k.replace('_',' ').title()}` → **+{v} pts**")
            st.code(result.get("risk_explanation",""), language=None)

        with tab2:
            identity_result = analyse_identity(pd.Series(record))
            iflags = identity_result.get("flags", [])
            if iflags:
                st.warning(f"Found {len(iflags)} identity inconsistency/ies:")
                for f in iflags:
                    st.markdown(f"  ⚠️ {f}")
            else:
                st.success("No identity inconsistencies detected.")

        with tab3:
            if existing_df.empty:
                st.info("No existing records to compare against.")
            else:
                from modules.data_processor import normalise_phone, normalise_email
                from modules.data_processor import normalise_address
                p_norm = normalise_phone(str(phone))
                e_norm = normalise_email(str(email))
                a_norm = normalise_address(str(address))
                if "phone_norm" not in existing_df.columns:
                    existing_df["phone_norm"] = existing_df["phone"].apply(normalise_phone)
                if "email_norm" not in existing_df.columns:
                    existing_df["email_norm"] = existing_df["email"].apply(normalise_email)
                if "address_norm" not in existing_df.columns:
                    existing_df["address_norm"] = existing_df["address"].apply(normalise_address)

                matches = existing_df[
                    (existing_df["phone_norm"]   == p_norm)  |
                    (existing_df["email_norm"]   == e_norm)  |
                    (existing_df["address_norm"] == a_norm)  |
                    (existing_df["device_id"]    == record["device_id"])
                ]
                if matches.empty:
                    st.info("No matching records found in existing database.")
                    st.caption("Note: Absence of matches does not confirm genuine identity.")
                else:
                    st.warning(f"Found {len(matches)} matching record(s):")
                    cols = ["applicant_id","full_name","risk_category","applicant_type"]
                    cols = [c for c in cols if c in matches.columns]
                    st.dataframe(matches[cols].reset_index(drop=True), use_container_width=True)

        # Save to DB
        col_save, _ = st.columns([1,2])
        with col_save:
            if st.button("💾 Save Assessment to Database"):
                upsert_applicant(record)
                st.success(f"Saved applicant {record['applicant_id']} to database.")
