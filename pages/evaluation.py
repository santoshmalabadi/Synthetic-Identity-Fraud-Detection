"""
Performance Evaluation page – Module 10.
Measures detection quality against ground-truth labels.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.database_manager import get_all_applicants, init_db
from modules.performance_evaluator import evaluate, evaluate_multiple_thresholds
from config import RISK_COLORS


def _confusion_matrix_fig(cm: list) -> go.Figure:
    z    = [[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]]
    text = [[str(cm[1][1]), str(cm[1][0])], [str(cm[0][1]), str(cm[0][0])]]
    fig  = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted Fraud", "Predicted Genuine"],
        y=["Actual Fraud", "Actual Genuine"],
        text=text,
        texttemplate="%{text}",
        colorscale=[[0,"#0a0e1a"],[0.5,"#1e3a5f"],[1,"#1e90ff"]],
        showscale=False,
    ))
    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#c0d0e0"),
        margin=dict(t=10,b=10), height=320,
    )
    return fig


def render():
    st.title("📈 Performance Evaluation")
    st.markdown(
        "Evaluate detection performance against **hidden ground-truth labels** "
        "from the generated synthetic dataset."
    )
    st.warning(
        "⚠️ These results are measured on a **synthetic test dataset only**. "
        "They do not represent real-world banking performance."
    )
    st.markdown("---")

    init_db()
    df = get_all_applicants()

    if df.empty:
        st.info("No data available. Generate a dataset and run risk scoring first.")
        return

    if "ground_truth_label" not in df.columns:
        st.error("Dataset does not contain ground-truth labels.")
        return

    if df["risk_score"].max() == 0:
        st.warning("Risk scores are all zero. Run Risk Analysis → Re-score first.")

    # ── Threshold selector ────────────────────────────────────────────────
    threshold = st.slider(
        "Risk Score Threshold for Fraud Classification",
        min_value=10, max_value=90, value=60, step=5,
    )
    st.caption(
        f"Applicants with risk score ≥ **{threshold}** are classified as suspected fraud."
    )

    result = evaluate(df, threshold=threshold)

    if "error" in result:
        st.error(result["error"])
        return

    # ── KPIs ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",     f"{result['accuracy']*100:.1f}%")
    c2.metric("Precision",    f"{result['precision']*100:.1f}%")
    c3.metric("Recall",       f"{result['recall']*100:.1f}%")
    c4.metric("F1 Score",     f"{result['f1_score']*100:.1f}%")
    c5.metric("FP Rate",      f"{result['false_positive_rate']*100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Confusion Matrix")
        cm = result.get("confusion_matrix")
        if cm:
            fig_cm = _confusion_matrix_fig(cm)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("**Legend:**")
        st.markdown(
            f"- **True Positives (TP):** {result['true_positives']} – Fraud correctly detected\n"
            f"- **False Positives (FP):** {result['false_positives']} – Genuine applicants flagged\n"
            f"- **True Negatives (TN):** {result['true_negatives']} – Genuine correctly cleared\n"
            f"- **False Negatives (FN):** {result['false_negatives']} – Fraud missed"
        )

    with col2:
        st.subheader("Classification Report")
        report = result.get("report", "")
        st.code(report, language=None)

        st.markdown("**Dataset Split:**")
        st.markdown(
            f"- Total samples: **{result['total_samples']}**\n"
            f"- Fraud samples: **{result['fraud_samples']}**\n"
            f"- Genuine samples: **{result['genuine_samples']}**"
        )

    # ── Multi-threshold comparison ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Threshold Comparison")
    st.caption("Compare detection performance across different risk score thresholds.")

    multi_df = evaluate_multiple_thresholds(df)
    if not multi_df.empty:
        fig_multi = go.Figure()
        for col, color in [
            ("Accuracy",  "#00c896"),
            ("Precision", "#1e90ff"),
            ("Recall",    "#ffb300"),
            ("F1 Score",  "#ff4c4c"),
        ]:
            if col in multi_df.columns:
                fig_multi.add_trace(go.Scatter(
                    x=multi_df["Threshold"], y=multi_df[col],
                    mode="lines+markers", name=col,
                    line=dict(color=color, width=2),
                    marker=dict(size=7),
                ))
        fig_multi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(bgcolor="rgba(10,14,26,0.8)", bordercolor="#1e90ff",
                        borderwidth=1, font=dict(color="#c0d0e0")),
            xaxis=dict(title="Risk Score Threshold", color="#c0d0e0",
                       gridcolor="rgba(100,120,160,0.2)"),
            yaxis=dict(title="Score", range=[0,1], color="#c0d0e0",
                       gridcolor="rgba(100,120,160,0.2)"),
            margin=dict(t=10,b=10), height=380,
        )
        st.plotly_chart(fig_multi, use_container_width=True)

        st.dataframe(multi_df.set_index("Threshold"), use_container_width=True)

    # ── FP/FN Explanation ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 False Positive / False Negative Analysis")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### False Positives (Genuine flagged as fraud)")
        st.markdown("""
        False positives occur when genuine applicants share attributes with others
        (e.g., family members sharing an address) or have unusual but legitimate 
        income patterns. These require human review to clear.
        """)
        # Show FP applicants
        from config import RISK_LOW_MAX
        fp_df = df[
            (df["ground_truth_label"].isin(["genuine"])) &
            (df["risk_score"] >= threshold)
        ]
        if not fp_df.empty:
            cols = ["applicant_id","full_name","risk_score","applicant_type"]
            cols = [c for c in cols if c in fp_df.columns]
            st.dataframe(fp_df[cols].head(10).reset_index(drop=True), use_container_width=True)
        else:
            st.success("No false positives at this threshold.")

    with col4:
        st.markdown("### False Negatives (Fraud missed)")
        st.markdown("""
        False negatives are fraud cases the system failed to flag above the threshold.
        New synthetic identities with no prior database matches are the hardest to detect
        using relationship analysis alone.
        """)
        fn_df = df[
            (df["ground_truth_label"].isin(["synthetic_fraud","fraud_ring"])) &
            (df["risk_score"] < threshold)
        ]
        if not fn_df.empty:
            cols = ["applicant_id","full_name","risk_score","applicant_type","ground_truth_label"]
            cols = [c for c in cols if c in fn_df.columns]
            st.dataframe(fn_df[cols].head(10).reset_index(drop=True), use_container_width=True)
        else:
            st.success("No false negatives at this threshold.")
