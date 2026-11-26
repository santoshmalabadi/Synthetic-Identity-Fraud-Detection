"""
Synthetic Identity Fraud Detection – Main Application Entry Point.
Run with: streamlit run app.py
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from modules.database_manager import init_db
from modules.preloader import preload_if_empty

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Synthetic Identity Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load CSS ──────────────────────────────────────────────────────────────
def _load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


_load_css()

# ── Hide the auto-generated Streamlit pages/ navigation menu ──────────────
# The app uses its own custom sidebar radio for navigation.
# config.toml sets showSidebarNavigation=false; this CSS is a belt-and-suspenders
# fallback for any residual DOM nodes and removes the blank padding gap.
st.markdown("""
<style>
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavLink"],
[data-testid="stSidebarNavContainer"],
section[data-testid="stSidebar"] nav {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
/* Remove blank gap Streamlit reserves above sidebar for the hidden nav */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Initialise DB on first run ────────────────────────────────────────────
init_db()

# ── Preload demo dataset if database is empty ─────────────────────────────
preload_if_empty(seed=42)

# ── Sidebar navigation ────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center;padding:16px 0 8px 0'>
  <div style='font-size:2rem'>🛡️</div>
  <div style='font-size:1.05rem;font-weight:700;color:#00d4ff;letter-spacing:.05em'>
    FraudGuard AI
  </div>
  <div style='font-size:.7rem;color:#8899aa;margin-top:2px'>
    Synthetic Identity Detection
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

PAGES = {
    "🏠 Dashboard":               "dashboard",
    "⚙️ Generate Synthetic Data":  "data_generator",
    "👥 Applicant Management":     "applicants",
    "🔍 New Applicant Assessment": "applicant_assessment",
    "🕸️ Relationship Graph":       "relationship_graph",
    "🔴 Fraud Ring Detection":     "fraud_rings",
    "📊 Risk Analysis":            "risk_analysis",
    "📈 Performance Evaluation":   "evaluation",
    "🔬 Investigation Centre":     "investigation",
    "ℹ️ About Project":            "about",
}

page_name = st.sidebar.radio(
    "Navigation",
    list(PAGES.keys()),
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:.7rem;color:#8899aa;padding:4px 0'>
  <b style='color:#1e90ff'>Quick Start</b><br>
  1. Generate Synthetic Data<br>
  2. Run Risk Analysis → Re-score<br>
  3. Explore Relationship Graph<br>
  4. Detect Fraud Rings<br>
  5. Review in Investigation Centre<br>
  6. Evaluate Performance
</div>
""", unsafe_allow_html=True)

# ── Route to page ─────────────────────────────────────────────────────────
module_name = PAGES[page_name]

try:
    if module_name == "dashboard":
        from pages.dashboard import render
    elif module_name == "data_generator":
        from pages.data_generator import render
    elif module_name == "applicants":
        from pages.applicants import render
    elif module_name == "applicant_assessment":
        from pages.applicant_assessment import render
    elif module_name == "relationship_graph":
        from pages.relationship_graph import render
    elif module_name == "fraud_rings":
        from pages.fraud_rings import render
    elif module_name == "risk_analysis":
        from pages.risk_analysis import render
    elif module_name == "evaluation":
        from pages.evaluation import render
    elif module_name == "investigation":
        from pages.investigation import render
    elif module_name == "about":
        from pages.about import render
    else:
        def render():
            st.error(f"Page '{module_name}' not found.")

    render()

except Exception as e:
    st.error(f"Error loading page: {e}")
    import traceback
    st.code(traceback.format_exc(), language="python")
