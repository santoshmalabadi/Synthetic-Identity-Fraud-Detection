"""
Applicant Management page – Module 3.
View, search, filter, upload, download, and add applicants.
"""

import streamlit as st
import pandas as pd
import io
from modules.database_manager import (
    get_all_applicants, insert_applicants, upsert_applicant, init_db,
    get_applicant_by_id,
)
from modules.data_processor import clean_dataframe
from modules.duplicate_detector import detect_duplicates
from modules.risk_scoring import score_dataframe, score_single
from modules.validation import validate_csv_dataframe, validate_applicant_dict
from modules.data_generator import _applicant_id, _id_ref, _device_id
from modules.fraud_explainer import generate_fraud_explanation
from config import ALL_STATUSES, RISK_COLORS


def render():
    st.title("👥 Applicant Management")
    st.markdown("View, search, filter, and manage all synthetic applicant records.")
    st.markdown("---")

    init_db()
    df = get_all_applicants()

    # ── Upload CSV ────────────────────────────────────────────────────────
    with st.expander("📤 Upload Applicant CSV", expanded=False):
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            try:
                upload_df = pd.read_csv(uploaded)
                upload_df, issues = validate_csv_dataframe(upload_df)
                if issues:
                    for issue in issues:
                        st.warning(issue)
                upload_df = clean_dataframe(upload_df)
                upload_df = detect_duplicates(upload_df)
                upload_df = score_dataframe(upload_df)
                if "applicant_id" not in upload_df.columns or upload_df["applicant_id"].isna().all():
                    upload_df["applicant_id"] = [_applicant_id() for _ in range(len(upload_df))]
                insert_applicants(upload_df)
                st.success(f"✅ Uploaded and saved {len(upload_df)} applicants.")
                df = get_all_applicants()
            except Exception as e:
                st.error(f"Upload failed: {e}")

    if df.empty:
        st.info("No applicants in database. Generate or upload data first.")
        return

    # ── Filters ───────────────────────────────────────────────────────────
    st.subheader("🔍 Search & Filter")
    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("Search by ID or Name", "")
    with col2:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            ["Low", "Medium", "High", "Critical"],
            default=[],
        )
    with col3:
        status_filter = st.multiselect(
            "Filter by Status",
            ALL_STATUSES,
            default=[],
        )

    filtered = df.copy()
    if search_term:
        mask = (
            filtered["applicant_id"].str.contains(search_term, case=False, na=False)
            | filtered["full_name"].str.contains(search_term, case=False, na=False)
        )
        filtered = filtered[mask]
    if risk_filter:
        filtered = filtered[filtered["risk_category"].isin(risk_filter)]
    if status_filter:
        filtered = filtered[filtered["application_status"].isin(status_filter)]

    st.markdown(f"Showing **{len(filtered)}** of **{len(df)}** applicants")

    # ── Table ─────────────────────────────────────────────────────────────
    display_cols = [
        "applicant_id", "full_name", "dob", "email", "phone",
        "applicant_type", "risk_score", "risk_category",
        "application_status", "application_date",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────
    csv_bytes = filtered.drop(
        columns=["phone_norm","email_norm","address_norm"], errors="ignore"
    ).to_csv(index=False).encode()
    st.download_button(
        "📥 Download Filtered Results CSV",
        data=csv_bytes,
        file_name="filtered_applicants.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Applicant Detail View ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔎 View Applicant Details")
    ids = filtered["applicant_id"].tolist()
    if ids:
        selected_id = st.selectbox("Select Applicant ID", ids)
        row = get_applicant_by_id(selected_id)
        if row is not None:
            tab1, tab2, tab3 = st.tabs(["Personal Info", "Risk Details", "Application Info"])
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Name:** {row.get('full_name','')}")
                    st.markdown(f"**DOB:** {row.get('dob','')}")
                    st.markdown(f"**Email:** {row.get('email','')}")
                    st.markdown(f"**Phone:** {row.get('phone','')}")
                with c2:
                    st.markdown(f"**Address:** {row.get('address','')}, {row.get('city','')}")
                    st.markdown(f"**State:** {row.get('state','')} – {row.get('pincode','')}")
                    st.markdown(f"**ID Ref:** {row.get('id_ref','')}")
                    st.markdown(f"**Device ID:** {row.get('device_id','')}")
            with tab2:
                score = row.get("risk_score", 0)
                cat   = row.get("risk_category", "Low")
                color = RISK_COLORS.get(cat, "#888")
                st.markdown(f"### Risk Score: <span style='color:{color}'>{score:.0f}/100 – {cat}</span>",
                            unsafe_allow_html=True)
                flags = row.get("risk_flags", [])
                if isinstance(flags, str):
                    import json
                    try: flags = json.loads(flags)
                    except: flags = []
                if flags:
                    st.markdown("**Risk Flags:**")
                    for f in flags:
                        st.markdown(f"  • {f}")
                expl = row.get("risk_explanation","")
                if expl:
                    st.code(expl, language=None)
            with tab3:
                st.markdown(f"**Employer:** {row.get('employer','')}")
                st.markdown(f"**Employment Type:** {row.get('employment_type','')}")
                st.markdown(f"**Annual Income:** ₹{float(row.get('annual_income',0)):,.0f}")
                st.markdown(f"**Application Date:** {row.get('application_date','')}")
                st.markdown(f"**Status:** {row.get('application_status','')}")
                st.markdown(f"**Type:** {row.get('applicant_type','')}")
                st.markdown(f"**Analyst Note:** {row.get('analyst_note','')}")

            # ── Fraud Detection Explanation ───────────────────────────────
            cat   = row.get("risk_category", "Low")
            score = float(row.get("risk_score", 0))
            color = RISK_COLORS.get(cat, "#888")
            icons = {"Low": "✅", "Medium": "⚠️", "High": "🔶", "Critical": "🚨"}
            icon  = icons.get(cat, "ℹ️")

            # Build a minimal score_result dict from stored data
            import json as _json
            stored_flags = row.get("risk_flags", [])
            if isinstance(stored_flags, str):
                try: stored_flags = _json.loads(stored_flags)
                except: stored_flags = []
            score_result_proxy = {
                "risk_score":    score,
                "risk_category": cat,
                "risk_flags":    stored_flags,
                "risk_explanation": row.get("risk_explanation", ""),
                "score_parts":   {},
            }
            explanation = generate_fraud_explanation(
                pd.Series(row), score_result_proxy, df
            )

            st.markdown(f"""
            <div style="background:#111d2e;border:1px solid {color}44;
                        border-left:5px solid {color};border-radius:12px;
                        padding:20px 24px;margin-top:18px;margin-bottom:8px">
              <div style="font-size:1rem;font-weight:700;color:{color};margin-bottom:12px">
                {icon} Why Was This Applicant Flagged?
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
                          gap:12px;margin-bottom:14px">
                <div>
                  <div style="font-size:.68rem;color:#8899aa;text-transform:uppercase">Applicant ID</div>
                  <div style="font-size:.85rem;color:#d0dcea;font-weight:600">{row.get('applicant_id','')}</div>
                </div>
                <div>
                  <div style="font-size:.68rem;color:#8899aa;text-transform:uppercase">Name</div>
                  <div style="font-size:.85rem;color:#d0dcea;font-weight:600">{row.get('full_name','')}</div>
                </div>
                <div>
                  <div style="font-size:.68rem;color:#8899aa;text-transform:uppercase">Risk Score</div>
                  <div style="font-size:.85rem;color:{color};font-weight:800">{score:.0f}/100</div>
                </div>
                <div>
                  <div style="font-size:.68rem;color:#8899aa;text-transform:uppercase">Risk Level</div>
                  <div style="font-size:.85rem;font-weight:700;
                       background:{color}22;color:{color};border:1px solid {color};
                       border-radius:5px;padding:1px 10px;display:inline-block">{cat}</div>
                </div>
              </div>
              <div style="background:#0a1628;border-radius:8px;padding:14px 16px;
                          font-size:.88rem;color:#c0d0e0;line-height:1.65">
                {explanation}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Add New Applicant ─────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("➕ Add New Synthetic Applicant Record", expanded=False):
        from faker import Faker
        fake = Faker("en_IN")
        with st.form("add_applicant_form"):
            c1, c2 = st.columns(2)
            with c1:
                name   = st.text_input("Full Name", value=fake.name())
                dob    = st.text_input("Date of Birth (YYYY-MM-DD)", value="1990-06-15")
                email  = st.text_input("Email", value=fake.email())
                phone  = st.text_input("Phone", value=fake.phone_number())
            with c2:
                address= st.text_input("Address", value=fake.street_address())
                city   = st.text_input("City",    value=fake.city())
                state  = st.text_input("State",   value=fake.state())
                pincode= st.text_input("Pincode", value=fake.postcode())
            c3, c4 = st.columns(2)
            with c3:
                employer    = st.text_input("Employer", value=fake.company())
                emp_type    = st.selectbox("Employment Type",
                             ["Salaried","Self-Employed","Contract","Unemployed"])
                annual_income = st.number_input("Annual Income (₹)", min_value=0.0,
                                                value=600000.0, step=10000.0)
            with c4:
                app_date = st.text_input("Application Date (YYYY-MM-DD)",
                                         value=pd.Timestamp.now().strftime("%Y-%m-%d"))

            submitted = st.form_submit_button("💾 Add Applicant")
            if submitted:
                record = {
                    "applicant_id": _applicant_id(),
                    "full_name": name, "dob": dob, "email": email, "phone": phone,
                    "address": address, "city": city, "state": state, "pincode": pincode,
                    "id_ref": _id_ref(), "device_id": _device_id(),
                    "application_date": app_date,
                    "employer": employer, "employment_type": emp_type,
                    "annual_income": annual_income,
                    "application_status": "New",
                    "ground_truth_label": "genuine",
                    "applicant_type": "normal",
                    "risk_score": 0.0, "risk_category": "Low",
                    "risk_flags": [], "risk_explanation": "", "analyst_note": "",
                }
                valid, errors, warnings = validate_applicant_dict(record)
                for w in warnings:
                    st.warning(w)
                if not valid:
                    for e in errors:
                        st.error(e)
                else:
                    all_df = get_all_applicants()
                    result = score_single(pd.Series(record), all_df)
                    record.update({
                        "risk_score":       result["risk_score"],
                        "risk_category":    result["risk_category"],
                        "risk_flags":       result["risk_flags"],
                        "risk_explanation": result["risk_explanation"],
                    })
                    upsert_applicant(record)
                    st.success(f"✅ Applicant {record['applicant_id']} added with risk score {result['risk_score']:.0f}.")
