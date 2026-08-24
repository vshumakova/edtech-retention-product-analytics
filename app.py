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

st.markdown("""
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1450px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.16);}
.hero {padding: 1.35rem 1.5rem; border-radius: 20px;
       background: linear-gradient(135deg, rgba(99,102,241,.13), rgba(14,165,233,.07));
       border: 1px solid rgba(99,102,241,.20); margin-bottom: 1.1rem;}
.hero h1 {margin:0; font-size:2.05rem;}
.hero p {margin:.45rem 0 0; opacity:.75;}
.metric-card {padding:1rem 1.1rem; border-radius:16px;
              border:1px solid rgba(128,128,128,.18); min-height:116px;}
.metric-label {font-size:.78rem; letter-spacing:.03em; opacity:.65; text-transform:uppercase;}
.metric-value {font-size:1.8rem; font-weight:750; margin-top:.18rem;}
.metric-sub {font-size:.78rem; opacity:.62; margin-top:.22rem;}
.note {padding:.8rem 1rem; border-left:3px solid #6366F1;
       background:rgba(99,102,241,.06); border-radius:0 10px 10px 0; margin-bottom:1rem;}
.small-muted {opacity:.65; font-size:.86rem;}
</style>
""", unsafe_allow_html=True)

FILES = {
    "models": "model_metrics_extended.csv",
    "ablation": "feature_source_ablation.csv",
    "temporal": "temporal_validation_extended.csv",
    "behavior": "behavioral_summary.csv",
    "funnel": "funnel_summary.csv",
    "capacity": "capacity_curve.csv",
    "audit": "source_audit.csv",
}
missing = [v for v in FILES.values() if not (RESULTS / v).exists()]
if missing:
    st.error("Missing public result files: " + ", ".join(missing))
    st.stop()

@st.cache_data
def load():
    return {k: pd.read_csv(RESULTS / v) for k, v in FILES.items()}

d = load()
models, ablation, temporal = d["models"], d["ablation"], d["temporal"]
behavior, funnel, capacity, audit = d["behavior"], d["funnel"], d["capacity"], d["audit"]
operating = models.loc[models["model"].str.contains("media \\+ training", regex=True)].iloc[0]

def card(label, value, sub=""):
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">{label}</div><div class="metric-value">{value}</div>
    <div class="metric-sub">{sub}</div></div>""", unsafe_allow_html=True)

def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎓 Cifrium")
    st.caption("Retention Intelligence")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Executive overview", "Retention signals", "Model & feature store",
         "Capacity planning", "Experiment & impact"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Prediction moment")
    st.markdown("**Day 14 of learning**")
    st.caption("North Star")
    st.markdown("**Course Completion Rate**")
    st.caption("Operating model")
    st.markdown("**Logistic Regression**")
    st.caption("Feature store")
    st.markdown("**Media + training**")
    st.caption("Public mode")
    st.markdown("**Aggregated results only**")

st.markdown("""<div class="hero">
<h1>Cifrium Retention Intelligence</h1>
<p>Early-warning system for student churn · D14 scoring · capacity-aware intervention</p>
</div>""", unsafe_allow_html=True)

if page == "Executive overview":
    c1,c2,c3,c4 = st.columns(4)
    with c1: card("Critical funnel loss","36%","Module 1 → Module 2")
    with c2: card("Precision @ Top-20%",f"{operating.precision_at_20:.1%}","High-risk queue accuracy")
    with c3: card("Recall @ Top-20%",f"{operating.recall_at_20:.1%}","Future churners captured")
    with c4: card("Lift @ Top-20%",f"{operating.lift_at_20:.2f}×","vs. random targeting")

    left,right = st.columns([1.15,.85])
    with left:
        st.subheader("Retention funnel")
        note("The largest retention loss occurs before Module 2, making the first module the primary intervention zone.")
        fig = px.bar(funnel, x="stage", y="index", text="index",
                     labels={"stage":"","index":"Student index (M1 = 100)"})
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Decision system")
        st.markdown("""
**1. Measure** the retention bottleneck  
**2. Score** each student at D14  
**3. Prioritize** the highest-risk segment  
**4. Intervene** using behavioral diagnostics  
**5. Experiment** to measure causal uplift  
**6. Monitor** downstream Course Completion
""")
        st.success("Recommended operating point: **Top 20% risk queue**")
        st.caption("The public dashboard never loads student-level raw data.")

    st.subheader("What changed after extending the feature store?")
    media = ablation.loc[ablation.feature_set == "Media only"].iloc[0]
    ext = ablation.loc[ablation.feature_set == "Media + training"].iloc[0]
    a,b,c = st.columns(3)
    with a: st.metric("Precision@20%", f"{ext.precision_at_20:.1%}", f"{(ext.precision_at_20-media.precision_at_20):+.1%}")
    with b: st.metric("Recall@20%", f"{ext.recall_at_20:.1%}", f"{(ext.recall_at_20-media.recall_at_20):+.1%}")
    with c: st.metric("Lift@20%", f"{ext.lift_at_20:.2f}×", f"{(ext.lift_at_20-media.lift_at_20):+.2f}×")

elif page == "Retention signals":
    st.subheader("Early behavioral differences")
    note("Future churn is associated with loss of learning cadence. These are predictive associations, not causal effects.")
    fig = go.Figure()
    fig.add_bar(y=behavior.metric, x=behavior.retained_mean, name="Retained", orientation="h")
    fig.add_bar(y=behavior.metric, x=behavior.churn_mean, name="Future churn", orientation="h")
    fig.update_layout(barmode="group", height=520, margin=dict(l=10,r=10,t=10,b=10),
                      xaxis_title="Mean value", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cohort instability")
    fig = px.bar(temporal, x="fold", y="test_churn_rate",
                 text=temporal.test_churn_rate.map(lambda x:f"{x:.1%}"),
                 labels={"fold":"Temporal fold","test_churn_rate":"Observed churn rate"})
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=330, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.warning("Churn prevalence shifts materially across cohorts. Production monitoring must be cohort-aware.")

elif page == "Model & feature store":
    st.subheader("Feature-source ablation")
    note("More data is not automatically better. Sources are retained only when they improve the actual operating decision.")
    show = ablation.copy()
    show["Precision@20%"] = show.precision_at_20.map(lambda x:f"{x:.1%}")
    show["Recall@20%"] = show.recall_at_20.map(lambda x:f"{x:.1%}")
    show["Lift@20%"] = show.lift_at_20.map(lambda x:f"{x:.2f}×")
    st.dataframe(show[["feature_set","n_features","roc_auc","pr_auc","brier","Precision@20%","Recall@20%","Lift@20%"]]
                 .rename(columns={"feature_set":"Feature set","n_features":"Features","roc_auc":"ROC-AUC","pr_auc":"PR-AUC","brier":"Brier"}),
                 hide_index=True, use_container_width=True)
    st.success("Selected: **Media + training** — strongest Precision, Recall and Lift at Top-20% capacity.")

    left,right = st.columns(2)
    with left:
        st.subheader("Model comparison")
        m = models.copy()
        st.dataframe(m.rename(columns={"model":"Model","roc_auc":"ROC-AUC","pr_auc":"PR-AUC",
                                       "brier":"Brier","precision_at_20":"Precision@20%",
                                       "recall_at_20":"Recall@20%","lift_at_20":"Lift@20%"}),
                     hide_index=True, use_container_width=True)
    with right:
        st.subheader("Temporal validation")
        long = temporal.melt(id_vars=["fold"], value_vars=["roc_auc","pr_auc"],
                             var_name="metric", value_name="value")
        fig = px.line(long, x="fold", y="value", color="metric", markers=True,
                      labels={"fold":"Temporal fold","value":"Score","metric":""})
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Source audit & leakage policy")
    st.dataframe(audit, hide_index=True, use_container_width=True)

elif page == "Capacity planning":
    st.subheader("Curator capacity trade-off")
    note("The threshold is a business decision based on available intervention capacity, not a generic probability cutoff.")
    cap = capacity.copy()
    cap["capacity_label"] = cap.capacity.map(lambda x:f"Top {x:.0%}")
    left,right = st.columns([1.15,.85])
    with left:
        fig = px.line(cap, x="capacity_label", y=["precision","recall"], markers=True,
                      labels={"value":"Rate","capacity_label":"Contact capacity","variable":""})
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=390, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        selected = st.select_slider("Available curator capacity",
                                    options=cap.capacity.tolist(), value=0.20,
                                    format_func=lambda x:f"Top {x:.0%}")
        row = cap.loc[cap.capacity == selected].iloc[0]
        st.metric("Expected precision", f"{row.precision:.1%}")
        st.metric("Expected recall", f"{row.recall:.1%}")
        if pd.notna(row.get("lift")): st.metric("Lift vs random", f"{row.lift:.2f}×")
        st.markdown("**Recommended portfolio scenario:** Top 20% balances queue size and churn coverage.")

elif page == "Experiment & impact":
    st.subheader("From prediction to causal impact")
    note("A high-quality model does not prove that outreach reduces churn. The intervention must be tested.")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
### Experiment design
**Population:** eligible high-risk D14 students  
**Control:** business-as-usual  
**Treatment:** standardized curator outreach  

**Primary metric**  
Module 1 → Module 2 Transition Rate

**Secondary metric**  
Course Completion Rate
""")
    with c2:
        st.markdown("""
### Guardrails
- curator workload
- intervention cost
- complaints / opt-outs
- no deterioration in low-risk experience

### Business impact
```text
Incremental retained students
=
Eligible contacted students
×
Measured causal uplift
```
""")
    st.info("Model ranking decides **who to prioritize**. The experiment decides **whether the action works**.")

st.divider()
st.caption("Portfolio dashboard · all values are loaded from aggregated public-safe files in results/.")
