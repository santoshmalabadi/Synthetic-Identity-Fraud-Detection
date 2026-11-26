"""
Relationship Graph Explorer – Phone Number Investigation Tool.

Analyst searches a phone number → sees every applicant sharing that number
in a compact interactive graph, 4 metric cards, and a details table.

Connections are established ONLY on shared phone number.
Risk scores and fraud-ring status come from the existing detection engine.
No applicant is classified as fraudulent solely because a phone is shared.

Only this file is modified. All other modules/pages are unchanged.
"""

from __future__ import annotations

import json
import math
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.data_processor import normalise_phone
from modules.database_manager import get_all_applicants, init_db

# ── Risk palette (matches project theme) ──────────────────────────────────────
RISK_COLOR = {
    "Low":      "#00c896",   # green
    "Medium":   "#ffb300",   # yellow / amber
    "High":     "#ff7700",   # orange
    "Critical": "#ff4c4c",   # red
}
RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

# ── Page-level CSS injection ──────────────────────────────────────────────────
_CSS = """
<style>
.rg-card {
    background: #111d2e;
    border: 1px solid rgba(30,144,255,.22);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.rg-card-val  { font-size: 1.7rem; font-weight: 800; }
.rg-card-lbl  { font-size: .68rem; text-transform: uppercase;
                 letter-spacing: .07em; color: #8899aa; margin-top: 3px; }
.rg-tbl       { width: 100%; border-collapse: collapse; }
.rg-tbl th    { padding: 7px 10px; text-align: left; color: #8899aa;
                 font-size: .68rem; text-transform: uppercase;
                 background: #0a1628; border-bottom: 2px solid rgba(30,144,255,.25); }
.rg-tbl td    { padding: 8px 10px; font-size: .82rem; color: #d0dcea;
                 border-bottom: 1px solid rgba(30,144,255,.07); }
.rg-tbl tr:hover td { background: rgba(30,144,255,.06); }
.rg-badge     { display: inline-block; border-radius: 5px;
                 padding: 1px 10px; font-size: .7rem; font-weight: 700;
                 border: 1px solid currentColor; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _search(df: pd.DataFrame, raw_phone: str) -> tuple[pd.DataFrame, str]:
    """
    Normalise raw_phone and return (matched_df, normalised_query).
    Matching uses normalised digits only — tolerates +91, spaces, dashes.
    """
    nq      = normalise_phone(raw_phone.strip())
    col     = df["phone"].apply(normalise_phone)
    matched = df[col == nq].copy()
    return matched, nq


def _overall_risk(matched: pd.DataFrame) -> str:
    """Return the highest risk category present in matched applicants."""
    if matched.empty:
        return "Low"
    return max(matched["risk_category"].unique(),
               key=lambda c: RISK_ORDER.get(c, 0))


def _fraud_ring_status(row: pd.Series) -> str:
    """Derive fraud-ring membership from existing applicant_type column."""
    atype = str(row.get("applicant_type", "")).lower()
    if "fraud" in atype or "ring" in atype:
        return "⚠️ Fraud Ring Member"
    return "Normal"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_figure(matched: pd.DataFrame, norm_q: str) -> go.Figure:
    """
    Star-layout Plotly figure:
      • Central node  = phone number  (cyan ★)
      • Surrounding   = matched applicants (risk-coloured ●)
    """
    n = len(matched)

    # ── Positions ─────────────────────────────────────────────────────────────
    # Radius scales with applicant count so labels don't collide
    r = max(1.8, 0.45 * n)

    # Central node
    cx, cy = 0.0, 0.0

    # Applicant positions on circle
    app_x, app_y = [], []
    for i in range(n):
        angle = 2 * math.pi * i / max(n, 1) - math.pi / 2  # start top
        app_x.append(r * math.cos(angle))
        app_y.append(r * math.sin(angle))

    # ── Edge traces ───────────────────────────────────────────────────────────
    edge_x, edge_y = [], []
    for ax, ay in zip(app_x, app_y):
        edge_x += [cx, ax, None]
        edge_y += [cy, ay, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="rgba(0,212,255,0.35)", width=1.5),
        hoverinfo="none",
        showlegend=False,
    )

    # ── Central phone node ────────────────────────────────────────────────────
    # Truncate for display
    display_q = norm_q if len(norm_q) <= 14 else norm_q[-10:]
    central_trace = go.Scatter(
        x=[cx], y=[cy],
        mode="markers+text",
        marker=dict(
            size=52, symbol="star",
            color="#00d4ff",
            line=dict(width=2.5, color="#ffffff"),
        ),
        text=[f"📞 {display_q}"],
        textposition="bottom center",
        textfont=dict(size=11, color="#00d4ff", family="monospace"),
        hovertext=[f"<b>📞 Phone Number</b><br>{norm_q}<br>"
                   f"<i>Shared by {n} applicant(s)</i>"],
        hoverinfo="text",
        name="Phone Number",
        showlegend=True,
    )

    # ── Applicant nodes ───────────────────────────────────────────────────────
    # Group by risk category so legend entries are clean
    app_traces = []
    for cat in ["Low", "Medium", "High", "Critical"]:
        bx, by, btext, bhover, bcustom = [], [], [], [], []
        for i, (_, row) in enumerate(matched.iterrows()):
            if row.get("risk_category", "Low") != cat:
                continue
            name   = str(row.get("full_name", ""))
            aid    = str(row.get("applicant_id", ""))
            score  = float(row.get("risk_score", 0))
            status = _fraud_ring_status(row)
            short  = name[:16] + "…" if len(name) > 16 else name

            bx.append(app_x[i])
            by.append(app_y[i])
            btext.append(short)
            bcustom.append(aid)
            bhover.append(
                f"<b>{name}</b><br>"
                f"ID: {aid}<br>"
                f"Risk: <b>{score:.0f}/100 — {cat}</b><br>"
                f"Status: {status}<br>"
                f"<i>Click to view details below</i>"
            )

        if bx:
            clr = RISK_COLOR[cat]
            app_traces.append(go.Scatter(
                x=bx, y=by,
                mode="markers+text",
                marker=dict(
                    size=28,
                    color=clr,
                    line=dict(width=2, color="#0a0e1a"),
                    opacity=0.92,
                ),
                text=btext,
                textposition="top center",
                textfont=dict(size=9, color="#d0dcea"),
                hovertext=bhover,
                hoverinfo="text",
                customdata=bcustom,
                name=f"{cat} Risk",
                showlegend=True,
            ))

    # ── Layout ────────────────────────────────────────────────────────────────
    pad   = 1.2
    x_rng = [-r - pad, r + pad]
    y_rng = [-r - pad - 0.6, r + pad + 0.6]   # extra bottom for central label

    # Dynamic height: compact for few applicants, taller for many
    height = max(380, min(600, 300 + n * 35))

    fig = go.Figure(
        data=[edge_trace, central_trace] + app_traces,
        layout=go.Layout(
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0a0e1a",
            showlegend=True,
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(10,14,26,0.85)",
                bordercolor="#1e90ff", borderwidth=1,
                font=dict(color="#c0d0e0", size=10),
                itemclick=False,
            ),
            hovermode="closest",
            dragmode="pan",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=x_rng),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=y_rng, scaleanchor="x"),
            height=height,
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def _metric_card(label: str, value, colour: str) -> str:
    return (
        f"<div class='rg-card'>"
        f"<div class='rg-card-val' style='color:{colour}'>{value}</div>"
        f"<div class='rg-card-lbl'>{label}</div>"
        f"</div>"
    )


def _render_cards(matched: pd.DataFrame, norm_q: str):
    n        = len(matched)
    avg      = matched["risk_score"].mean() if n else 0.0
    overall  = _overall_risk(matched)
    ov_color = RISK_COLOR.get(overall, "#00d4ff")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_metric_card("Matching Applicants", n, "#1e90ff"),
                unsafe_allow_html=True)
    c2.markdown(_metric_card("Shared Phone Count", n, "#a855f7"),
                unsafe_allow_html=True)
    c3.markdown(_metric_card("Avg Risk Score", f"{avg:.1f}/100",
                             "#ffb300" if avg >= 30 else "#00c896"),
                unsafe_allow_html=True)
    c4.markdown(_metric_card("Overall Fraud Risk", overall, ov_color),
                unsafe_allow_html=True)


def _render_detail_card(row: pd.Series):
    """Detail panel shown when an applicant node is selected."""
    score  = float(row.get("risk_score", 0))
    cat    = str(row.get("risk_category", "Low"))
    color  = RISK_COLOR.get(cat, "#888")
    status = _fraud_ring_status(row)
    flags  = row.get("risk_flags", [])
    if isinstance(flags, str):
        try:    flags = json.loads(flags)
        except: flags = []

    st.markdown(
        f"<div style='background:#0d1b2a;border:1px solid {color}44;"
        f"border-left:4px solid {color};border-radius:10px;padding:16px 20px;"
        f"margin-bottom:12px'>"
        f"<div style='font-size:1rem;font-weight:700;color:{color};margin-bottom:8px'>"
        f"📋 {row.get('full_name','')}  "
        f"<span style='font-size:.75rem;color:#8899aa;font-weight:400'>"
        f"{row.get('applicant_id','')}</span></div>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:.82rem'>"
        f"<div><span style='color:#8899aa'>Phone</span><br>"
        f"<b style='color:#d0dcea'>{row.get('phone','')}</b></div>"
        f"<div><span style='color:#8899aa'>Risk Score</span><br>"
        f"<b style='color:{color}'>{score:.0f}/100</b></div>"
        f"<div><span style='color:#8899aa'>Risk Level</span><br>"
        f"<b class='rg-badge' style='color:{color};border-color:{color};"
        f"background:{color}22'>{cat}</b></div>"
        f"<div><span style='color:#8899aa'>Status</span><br>"
        f"<b style='color:#d0dcea'>{row.get('application_status','New')}</b></div>"
        f"<div><span style='color:#8899aa'>Fraud Ring</span><br>"
        f"<b style='color:{'#ff4c4c' if 'Fraud' in status else '#00c896'}'>{status}</b></div>"
        f"<div><span style='color:#8899aa'>Applicant Type</span><br>"
        f"<b style='color:#d0dcea'>{row.get('applicant_type','')}</b></div>"
        f"</div>"
        + (
            f"<div style='margin-top:10px;font-size:.78rem;color:#ff9966'>"
            f"⚠️ Flags: {' · '.join(str(f) for f in flags[:4])}</div>"
            if flags else ""
        )
        + f"<div style='margin-top:8px;font-size:.72rem;color:#8899aa'>"
        f"ℹ️ Shared phone number is a relationship signal — "
        f"not automatic evidence of fraud.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_table(matched: pd.DataFrame):
    """Compact applicant details table below the graph."""
    rows_html = ""
    sorted_df = matched.sort_values("risk_score", ascending=False).reset_index(drop=True)

    for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
        cat    = str(row.get("risk_category", "Low"))
        score  = float(row.get("risk_score", 0))
        clr    = RISK_COLOR.get(cat, "#888")
        status = _fraud_ring_status(row)
        bar_w  = max(4, int(score))
        badge  = (f"<span class='rg-badge' "
                  f"style='color:{clr};border-color:{clr};background:{clr}22'>{cat}</span>")
        bar    = (f"<div style='background:#1a2a3a;border-radius:3px;height:5px;width:80px'>"
                  f"<div style='background:{clr};width:{bar_w}%;height:5px;border-radius:3px'>"
                  f"</div></div>")
        ring_col = "#ff4c4c" if "Fraud" in status else "#00c896"
        rows_html += f"""
        <tr>
          <td style='color:#00d4ff;font-weight:700'>#{rank}</td>
          <td>
            <div style='font-weight:600;color:#e0e6f0'>{row.get('full_name','')}</div>
            <div style='color:#8899aa;font-size:.7rem'>{row.get('applicant_id','')}</div>
          </td>
          <td style='font-family:monospace;color:#a0b4c8'>{row.get('phone','')}</td>
          <td>
            <div style='font-weight:700;color:{clr}'>{score:.0f}<span style='color:#8899aa;font-size:.7rem'>/100</span></div>
            {bar}
          </td>
          <td>{badge}</td>
          <td style='color:{ring_col};font-size:.78rem'>{status}</td>
          <td style='color:#8899aa;font-size:.75rem'>{row.get('application_status','New')}</td>
        </tr>"""

    table_html = f"""
    <div style='overflow-x:auto;margin-top:4px'>
    <table class='rg-tbl'>
      <thead><tr>
        <th>#</th><th>Applicant</th><th>Phone</th>
        <th>Risk Score</th><th>Risk Level</th>
        <th>Fraud Ring</th><th>Status</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>"""

    st.markdown(table_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render():
    # ── Inject styles ─────────────────────────────────────────────────────────
    st.markdown(_CSS, unsafe_allow_html=True)

    st.title("🕸️ Relationship Graph Explorer")
    st.markdown(
        "Search a **phone number** to instantly see every applicant sharing it, "
        "their risk levels, and relationship graph."
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    init_db()
    df = get_all_applicants()

    if df.empty:
        st.info("📋 No applicant data found. Go to **Generate Synthetic Data** first.")
        return

    # Precompute normalised phones for suggestions
    df["_pn"] = df["phone"].apply(normalise_phone)

    # ── Session state ─────────────────────────────────────────────────────────
    if "rg_phone_result" not in st.session_state:
        st.session_state["rg_phone_result"] = None
    if "rg_selected_aid" not in st.session_state:
        st.session_state["rg_selected_aid"] = None

    # ══════════════════════════════════════════════════════════════════════════
    # ① SEARCH BAR
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div style='background:#0d1b2a;border:1px solid rgba(30,144,255,.28);"
        "border-radius:12px;padding:18px 22px;margin-bottom:16px'>"
        "<div style='font-size:1rem;font-weight:700;color:#00d4ff;margin-bottom:12px'>"
        "📞 Phone Number Search</div>",
        unsafe_allow_html=True,
    )

    col_input, col_search, col_clear = st.columns([4, 1, 1])
    with col_input:
        phone_input = st.text_input(
            "Phone Number",
            value=st.session_state.get("rg_phone_input", ""),
            placeholder="e.g. +91 9876543210 or 9876543210",
            label_visibility="collapsed",
            key="rg_phone_input_field",
        )
    with col_search:
        search_btn = st.button("🔍 Search", use_container_width=True, type="primary")
    with col_clear:
        clear_btn = st.button("✕ Clear", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Quick-select: show top shared phones as clickable chips
    top_phones = df["_pn"].value_counts()
    shared_phones = top_phones[top_phones >= 2].head(5)
    if not shared_phones.empty:
        st.markdown(
            "<div style='font-size:.75rem;color:#8899aa;margin-bottom:6px'>"
            "🔗 <b style='color:#1e90ff'>Quick Search — Top Shared Numbers:</b></div>",
            unsafe_allow_html=True,
        )
        chip_cols = st.columns(len(shared_phones))
        for i, (pn, cnt) in enumerate(shared_phones.items()):
            raw = df[df["_pn"] == pn]["phone"].iloc[0]
            cats = df[df["_pn"] == pn]["risk_category"].value_counts()
            dominant_cat = cats.idxmax() if not cats.empty else "Low"
            chip_color = RISK_COLOR.get(dominant_cat, "#1e90ff")
            with chip_cols[i]:
                if st.button(
                    f"📞 ···{pn[-6:]}\n({cnt} applicants)",
                    key=f"chip_{i}",
                    use_container_width=True,
                ):
                    matched, norm_q = _search(df, raw)
                    st.session_state["rg_phone_result"] = {
                        "matched": matched,
                        "norm_q":  norm_q,
                        "raw":     raw,
                    }
                    st.session_state["rg_selected_aid"] = None
                    st.rerun()

    # ── Handle Clear ──────────────────────────────────────────────────────────
    if clear_btn:
        st.session_state["rg_phone_result"] = None
        st.session_state["rg_selected_aid"] = None
        st.rerun()

    # ── Handle Search ─────────────────────────────────────────────────────────
    if search_btn:
        raw_input = phone_input.strip()
        if not raw_input:
            st.warning("⚠️ Please enter a phone number to search.")
        else:
            # Basic validation: at least 7 digits
            digit_count = len(re.sub(r"\D", "", raw_input))
            if digit_count < 7:
                st.error(
                    f"❌ **Invalid phone number** `{raw_input}` — "
                    "please enter at least 7 digits."
                )
            else:
                with st.spinner("Searching database…"):
                    matched, norm_q = _search(df, raw_input)
                st.session_state["rg_phone_result"] = {
                    "matched": matched,
                    "norm_q":  norm_q,
                    "raw":     raw_input,
                }
                st.session_state["rg_selected_aid"] = None
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # ② RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    result = st.session_state.get("rg_phone_result")

    if result is None:
        # ── Landing state ─────────────────────────────────────────────────────
        st.markdown(
            "<div style='text-align:center;padding:48px 20px;color:#8899aa'>"
            "<div style='font-size:3.5rem'>📞</div>"
            "<div style='font-size:1.05rem;font-weight:700;color:#1e90ff;margin:12px 0'>"
            "Enter a phone number above to investigate connections</div>"
            "<div style='font-size:.85rem;max-width:500px;margin:0 auto'>"
            "The graph will display <b style='color:#00d4ff'>only</b> the applicants "
            "who share that exact phone number — no unrelated applicants shown."
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    matched = result["matched"]
    norm_q  = result["norm_q"]
    raw     = result["raw"]

    st.markdown("---")

    # ── No results ────────────────────────────────────────────────────────────
    if matched.empty:
        st.warning(
            f"❌ No applicants found with phone number **`{raw}`** "
            f"(normalised: `{norm_q}`). "
            "Try a different number or use one of the Quick Search buttons above."
        )
        return

    # ── Single applicant ──────────────────────────────────────────────────────
    if len(matched) == 1:
        st.success(
            f"✅ Phone `{norm_q}` belongs to **1 applicant only** — "
            "no shared-phone connection concerns."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ③ METRIC CARDS
    # ══════════════════════════════════════════════════════════════════════════
    _render_cards(matched, norm_q)
    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Multi-applicant warning banner ────────────────────────────────────────
    if len(matched) > 1:
        overall = _overall_risk(matched)
        ov_col  = RISK_COLOR.get(overall, "#888")
        n_high  = len(matched[matched["risk_category"].isin(["High", "Critical"])])
        st.markdown(
            f"<div style='background:{ov_col}11;border:1px solid {ov_col}44;"
            f"border-left:4px solid {ov_col};border-radius:8px;padding:10px 16px;"
            f"margin-bottom:14px;font-size:.85rem'>"
            f"🔗 <b style='color:{ov_col}'>{len(matched)} applicants</b> share "
            f"phone <code>{norm_q}</code>."
            + (f" <b>{n_high} are High/Critical risk.</b>" if n_high else "")
            + " Shared phone is a <b>relationship signal</b>, not automatic evidence "
            "of fraud. Review each applicant independently.</div>",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ④ INTERACTIVE GRAPH
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"### 🗺️ Relationship Graph — `{norm_q}`"
    )
    st.caption(
        "⭐ Star = Phone Number  |  ● Circle = Applicant  |  "
        "🟢 Low  🟡 Medium  🟠 High  🔴 Critical  |  "
        "Hover for details · scroll to zoom · drag to pan"
    )

    fig = _build_figure(matched, norm_q)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["pan2d", "zoomIn2d", "zoomOut2d", "resetScale2d"],
        },
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ⑤ NODE CLICK — select applicant to view full details
    # ══════════════════════════════════════════════════════════════════════════
    sel_options = ["— select applicant to view details —"] + [
        f"{row['full_name']}  ({row['applicant_id']})"
        for _, row in matched.sort_values("risk_score", ascending=False).iterrows()
    ]
    sel_idx = st.selectbox(
        "🔍 Select an applicant node to view details",
        range(len(sel_options)),
        format_func=lambda i: sel_options[i],
        key="rg_node_sel",
    )

    if sel_idx > 0:
        # Map back to applicant_id
        sorted_matched = matched.sort_values("risk_score", ascending=False).reset_index(drop=True)
        selected_row   = sorted_matched.iloc[sel_idx - 1]
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        _render_detail_card(selected_row)

    # ══════════════════════════════════════════════════════════════════════════
    # ⑥ APPLICANT DETAILS TABLE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📋 Applicant Details")
    _render_table(matched)

    st.markdown(
        "<div style='margin-top:16px;padding:10px 14px;background:#0d1b2a;"
        "border-radius:8px;font-size:.75rem;color:#8899aa;border:1px solid"
        " rgba(30,144,255,.12)'>"
        "ℹ️ <b>Detection note:</b> Graph connections above are established "
        "<b>only</b> on shared phone number. Risk scores and fraud-ring status "
        "are determined by the existing detection engine — not by graph proximity alone."
        "</div>",
        unsafe_allow_html=True,
    )
