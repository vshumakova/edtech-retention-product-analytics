from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="Cifrium Retention Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.15);}
      .hero {
          padding: 1.3rem 1.5rem;
          border-radius: 18px;
          background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(14,165,233,.08));
          border: 1px solid rgba(99,102,241,.18);
          margin-bottom: 1rem;
      }
      .hero h1 {margin: 0; font-size: 2rem;}
      .hero p {margin: .4rem 0 0 0; opacity: .78;}
      .metric-card {
          padding: 1rem 1.1rem;
          border-radius: 16px;
          border: 1px solid rgba(128,128,128,.18);
          background: rgba(255,255,255,.02);
          min-height: 116px;
      }
      .metric-label {font-size: .85rem; opacity: .68;}
      .metric-value {font-size: 1.75rem; font-weight: 700; margin-top: .15rem;}
      .metric-sub {font-size: .78rem; opacity: .62; margin-top: .2rem;}
      .section-note {
          padding: .8rem 1rem;
          border-left: 3px solid #6366F1;
          background: rgba(99,102,241,.06);
          border-radius: 0 10px 10px 0;
          margin-bottom: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED = {
    "model_metrics": RESULTS / "model_metrics.csv",
    "temporal_validation": RESULTS / "temporal_validation.csv",
    "behavioral_summary": RESULTS / "behavioral_summary.csv",
    "feature_importance": RESULTS / "feature_importance.csv",
    "funnel_summary": RESULTS / "funnel_summary.csv",
    "capacity_curve": RESULTS / "capacity_curve.csv",
}

missing = [name for name, path in REQUIRED.items() if not path.exists()]
if missing:
    st.error(
        "Missing public result files: "
        + ", ".join(missing)
        + ". Run the analytical pipeline first."
    )
    st.stop()

@st.cache_data
def load_results():
    return {name: pd.read_csv(path) for name, path in REQUIRED.items()}

data = load_results()
models = data["model_metrics"]
temporal = data["temporal_validation"]
behavior = data["behavioral_summary"]
importance = data["feature_importance"]
funnel = data["funnel_summary"]
capacity = data["capacity_curve"]

rf = models.loc[models["model"] == "Random Forest"].iloc[0]
logit = models.loc[models["model"] == "Logistic Regression"].iloc[0]

with st.sidebar:
    st.markdown("## 🎓 Cifrium")
    st.caption("Retention Intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Executive overview",
            "Retention analytics",
            "Model quality",
            "Capacity planning",
            "Experiment framework",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Scoring point")
    st.markdown("**Day 14 of learning**")
    st.caption("North Star")
    st.markdown("**Course Completion Rate**")
    st.caption("Operating model")
    st.markdown("**Random Forest**")
    st.caption("Public data mode")
    st.markdown("**Aggregated results only**")

def card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def note(text: str):
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>Early Churn Detection</h1>
      <p>Product analytics and D14 risk scoring for proactive student retention.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Executive overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Critical funnel loss", "36%", "Loss between Module 1 and Module 2")
    with c2:
        card("Future holdout", "615", "Students from later cohorts")
    with c3:
        card("Recall@Top-20%", f"{rf['recall_at_20']:.1%}", "Share of future churners captured")
    with c4:
        card("Lift@Top-20%", f"{rf['lift_at_20']:.2f}×", "Risk concentration vs cohort baseline")

    st.write("")
    left, right = st.columns([1.15, .85])

    with left:
        st.subheader("Course retention funnel")
        note(
            "The largest product opportunity is concentrated before Module 2, "
            "so the first module is the primary intervention zone."
        )
        fig = px.bar(
            funnel,
            x="stage",
            y="index",
            text=funnel["index"].map(lambda x: f"{x:.1f}"),
            labels={"stage": "", "index": "Student index (M1 = 100)"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Decision system")
        st.markdown(
            """
            **1. Measure** the retention bottleneck  
            **2. Score** each student at D14  
            **3. Prioritize** the highest-risk segment  
            **4. Intervene** using behavioral diagnostics  
            **5. Experiment** to measure causal uplift  
            **6. Monitor** downstream Course Completion
            """
        )
        st.info(
            "This public dashboard uses only aggregated model outputs. "
            "No student-level raw data is loaded by the app."
        )

elif page == "Retention analytics":
    st.subheader("Early behavioral differences")
    note(
        "These are observed D14 associations between later retained and churned students. "
        "They are useful for diagnosis, but they are not causal estimates."
    )

    display = behavior.copy()
    display["retained_label"] = display["retained_mean"]
    display["churn_label"] = display["churn_mean"]

    fig = go.Figure()
    fig.add_bar(
        y=display["metric"],
        x=display["retained_mean"],
        name="Retained",
        orientation="h",
    )
    fig.add_bar(
        y=display["metric"],
        x=display["churn_mean"],
        name="Future churn",
        orientation="h",
    )
    fig.update_layout(
        barmode="group",
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Mean value",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Temporal cohort variation")
    fig = px.bar(
        temporal,
        x="fold",
        y="test_churn_rate",
        text=temporal["test_churn_rate"].map(lambda x: f"{x:.1%}"),
        labels={"fold": "Temporal fold", "test_churn_rate": "Observed churn rate"},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model quality":
    st.subheader("Future-cohort model comparison")
    note(
        "Logistic Regression is strongest on overall ranking and calibration. "
        "Random Forest is selected for operations because it performs best inside the Top-20% curator-capacity constraint."
    )

    comparison = models.copy()
    comparison["ROC-AUC"] = comparison["roc_auc"]
    comparison["PR-AUC"] = comparison["pr_auc"]
    comparison["Precision@20%"] = comparison["precision_at_20"]
    comparison["Recall@20%"] = comparison["recall_at_20"]
    comparison["Lift@20%"] = comparison["lift_at_20"]

    st.dataframe(
        comparison[
            ["model", "ROC-AUC", "PR-AUC", "brier", "Precision@20%", "Recall@20%", "Lift@20%"]
        ].rename(columns={"model": "Model", "brier": "Brier"}),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Random Forest ROC-AUC", f"{rf['roc_auc']:.3f}")
    with c2:
        card("Random Forest PR-AUC", f"{rf['pr_auc']:.3f}")
    with c3:
        card("Precision@20%", f"{rf['precision_at_20']:.1%}")
    with c4:
        card("Recall@20%", f"{rf['recall_at_20']:.1%}")

    left, right = st.columns(2)

    with left:
        st.subheader("Feature importance")
        imp = importance.sort_values("importance").tail(12)
        fig = px.bar(
            imp,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance", "feature": ""},
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Temporal validation")
        temp_long = temporal.melt(
            id_vars=["fold"],
            value_vars=["roc_auc", "pr_auc"],
            var_name="metric",
            value_name="value",
        )
        fig = px.line(
            temp_long,
            x="fold",
            y="value",
            color="metric",
            markers=True,
            labels={"fold": "Temporal fold", "value": "Score", "metric": ""},
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Capacity planning":
    st.subheader("Curator capacity trade-off")
    note(
        "The operating threshold is selected from available curator capacity, "
        "not from a generic probability threshold of 0.5."
    )

    cap = capacity.copy()
    cap["capacity_label"] = cap["capacity"].map(lambda x: f"Top {x:.0%}")

    left, right = st.columns(2)

    with left:
        fig = px.line(
            cap,
            x="capacity_label",
            y=["precision", "recall"],
            markers=True,
            labels={"value": "Rate", "capacity_label": "Contact capacity", "variable": ""},
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        selected = st.select_slider(
            "Available curator capacity",
            options=cap["capacity"].tolist(),
            value=0.20,
            format_func=lambda x: f"Top {x:.0%}",
        )
        row = cap.loc[cap["capacity"] == selected].iloc[0]

        st.metric("Expected precision", f"{row['precision']:.1%}")
        st.metric("Expected recall", f"{row['recall']:.1%}")
        if pd.notna(row.get("lift")):
            st.metric("Lift vs random targeting", f"{row['lift']:.2f}×")

        st.markdown(
            """
            **Interpretation:** increasing the queue size improves recall but reduces precision.  
            The product team can therefore choose the operating point from staffing capacity and intervention cost.
            """
        )

elif page == "Experiment framework":
    st.subheader("How to prove retention impact")
    note(
        "Prediction identifies risk. Only a randomized experiment can establish whether an intervention actually reduces churn."
    )

    st.markdown(
        """
        ### Population
        Students classified as high risk on D14 and eligible for intervention.

        ### Randomization
        - **Control:** business-as-usual
        - **Treatment:** standardized curator outreach

        ### Primary metric
        **Module 1 → Module 2 Transition Rate**

        ### Secondary metric
        **Course Completion Rate**

        ### Guardrails
        - curator workload
        - intervention cost
        - complaints / opt-outs

        ### Impact formula

        ```text
        Incremental retained students
        =
        Eligible contacted students
        ×
        measured causal uplift
        ```

        Model ranking determines **who should be contacted first**.  
        The experiment determines **whether the chosen action works**.
        """
    )

st.divider()
st.caption(
    "Public portfolio dashboard. Source student-level data is intentionally excluded; "
    "all visible values are read from aggregated files in results/."
)
