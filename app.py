from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(
    page_title="Cifrium Retention Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme / CSS ----------
st.markdown(
    """
    <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.15);}
        .hero {
            padding: 1.35rem 1.5rem;
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
        .risk-high {color:#ef4444;font-weight:700;}
        .risk-medium {color:#f59e0b;font-weight:700;}
        .risk-low {color:#22c55e;font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Demo data ----------
@st.cache_data
def demo_data(seed: int = 42):
    rng = np.random.default_rng(seed)

    funnel = pd.DataFrame({
        "module": ["Module 1", "Module 2", "Module 3", "Module 4"],
        "students": [1000, 640, 584, 548],
    })
    funnel["transition_rate"] = [1.00, .64, .9125, .9384]

    cohorts = pd.DataFrame({
        "cohort": pd.period_range("2025-01", periods=10, freq="M").astype(str),
        "retention_rate": [.61, .63, .60, .65, .64, .66, .62, .67, .65, .68],
        "students": [92, 104, 98, 111, 107, 115, 96, 121, 118, 126],
    })

    n = 240
    risk = np.clip(rng.beta(2.1, 3.2, size=n), 0, 1)
    churn = rng.binomial(1, np.clip(.08 + .78 * risk, 0, .95))
    students = pd.DataFrame({
        "student_id": [f"ST-{1000+i}" for i in range(n)],
        "risk_score": risk,
        "observed_churn": churn,
        "active_days_d14": np.clip(np.rint(12 - 8*risk + rng.normal(0, 1.7, n)), 0, 14).astype(int),
        "days_since_last_activity": np.clip(1 + 9*risk + rng.normal(0, 1.5, n), 0, 14),
        "task_attempts_d14": np.clip(np.rint(32 - 24*risk + rng.normal(0, 5, n)), 0, None).astype(int),
        "video_watch_pct": np.clip(.82 - .55*risk + rng.normal(0, .11, n), 0, 1),
    })
    students["risk_segment"] = pd.cut(
        students["risk_score"],
        bins=[-np.inf, .50, .72, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    model_metrics = pd.DataFrame({
        "metric": ["ROC-AUC", "PR-AUC", "Recall@Top20%", "Precision@Top20%", "Lift@Top20%"],
        "value": [.82, .71, .63, .58, 1.86],
    })

    feature_imp = pd.DataFrame({
        "feature": [
            "Days since last activity",
            "Active days in D14",
            "Task attempts in D14",
            "Video watch depth",
            "LMS sessions in D14",
            "Unique lessons in D14",
        ],
        "importance": [.23, .19, .17, .14, .11, .08],
    })

    return funnel, cohorts, students, model_metrics, feature_imp


@st.cache_data
def try_load_real_data():
    """
    The dashboard is portfolio-ready even without private source data.
    If prepared exports are present, use them; otherwise fall back to demo mode.
    """
    prepared = DATA / "dashboard_students.csv"
    cohorts_path = DATA / "dashboard_cohorts.csv"
    funnel_path = DATA / "dashboard_funnel.csv"
    metrics_path = DATA / "dashboard_model_metrics.csv"
    importance_path = DATA / "dashboard_feature_importance.csv"

    required = [prepared, cohorts_path, funnel_path, metrics_path, importance_path]
    if all(p.exists() for p in required):
        return (
            pd.read_csv(funnel_path),
            pd.read_csv(cohorts_path),
            pd.read_csv(prepared),
            pd.read_csv(metrics_path),
            pd.read_csv(importance_path),
            False,
        )

    f, c, s, m, i = demo_data()
    return f, c, s, m, i, True


funnel, cohorts, students, model_metrics, feature_imp, demo_mode = try_load_real_data()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎓 Cifrium")
    st.caption("Retention Intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Executive overview", "Retention analytics", "Risk operations", "Model quality", "Impact simulator"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Scoring point")
    st.markdown("**Day 14 after enrollment**")
    st.caption("North Star")
    st.markdown("**Course Completion Rate**")
    st.caption("Decision metric")
    st.markdown("**M1 → M2 transition**")

    if demo_mode:
        st.divider()
        st.info(
            "Portfolio demo mode\n\n"
            "The UI is populated with illustrative data because private source CSVs are not included in the repository."
        )

# ---------- Helpers ----------
def card(label, value, sub=""):
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

def section_title(title, note=None):
    st.subheader(title)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>Early Churn Detection</h1>
        <p>Product analytics and D14 risk scoring for proactive student retention.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Page 1 ----------
if page == "Executive overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Critical funnel loss", "36%", "Students lost between M1 and M2")
    with c2:
        card("Scoring moment", "D14", "Before the outcome is known")
    with c3:
        high_share = (students["risk_segment"] == "High").mean()
        card("High-risk share", f"{high_share:.0%}", "Students requiring priority attention")
    with c4:
        top20 = students.nlargest(max(1, int(len(students)*.2)), "risk_score")
        card("Observed churn in top 20%", f"{top20['observed_churn'].mean():.0%}", "Risk concentration in priority queue")

    st.write("")
    left, right = st.columns([1.15, .85])

    with left:
        section_title(
            "Course funnel",
            "The largest loss is concentrated before Module 2, which makes the first module the highest-value intervention point.",
        )
        fig = px.bar(
            funnel,
            x="module",
            y="students",
            text="students",
            labels={"module":"", "students":"Students"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section_title("Operating logic")
        st.markdown(
            """
            **1. Diagnose** where retention breaks  
            **2. Score** every student on D14  
            **3. Prioritize** limited curator capacity  
            **4. Intervene** using behavior-specific playbooks  
            **5. Experiment** to estimate causal uplift  
            **6. Scale** only if Course Completion improves
            """
        )

        st.markdown("#### Risk mix")
        mix = students["risk_segment"].value_counts().reindex(["High","Medium","Low"]).fillna(0)
        fig = go.Figure(go.Pie(
            labels=mix.index,
            values=mix.values,
            hole=.64,
            sort=False,
        ))
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# ---------- Page 2 ----------
elif page == "Retention analytics":
    section_title(
        "Cohort retention",
        "Cohort tracking prevents a single blended average from hiding deterioration in newer student groups.",
    )

    fig = px.line(
        cohorts,
        x="cohort",
        y="retention_rate",
        markers=True,
        labels={"cohort":"Enrollment cohort","retention_rate":"M1 retention"},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Early behavioral signals")
    signal_cols = {
        "active_days_d14": "Active days in first 14 days",
        "days_since_last_activity": "Days since last activity",
        "task_attempts_d14": "Task attempts by D14",
        "video_watch_pct": "Average video watch depth",
    }
    selected = st.selectbox("Behavior", list(signal_cols.keys()), format_func=signal_cols.get)

    tmp = students[[selected, "observed_churn"]].copy()
    try:
        tmp["bucket"] = pd.qcut(tmp[selected], 5, duplicates="drop")
    except Exception:
        tmp["bucket"] = pd.cut(tmp[selected], 5)

    risk_curve = (
        tmp.groupby("bucket", observed=True)
        .agg(students=("observed_churn","size"), churn_rate=("observed_churn","mean"))
        .reset_index()
    )
    risk_curve["bucket_label"] = risk_curve["bucket"].astype(str)

    fig = px.bar(
        risk_curve,
        x="bucket_label",
        y="churn_rate",
        text=risk_curve["churn_rate"].map(lambda x: f"{x:.0%}"),
        labels={"bucket_label":"Behavior bucket","churn_rate":"Observed churn"},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------- Page 3 ----------
elif page == "Risk operations":
    section_title(
        "D14 intervention queue",
        "The score ranks students by priority. The intervention itself is chosen from the underlying behavioral signal.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        segment = st.multiselect("Risk segment", ["High","Medium","Low"], default=["High","Medium"])
    with col2:
        min_risk = st.slider("Minimum risk score", 0.0, 1.0, 0.50, 0.05)
    with col3:
        max_rows = st.selectbox("Rows to show", [10, 25, 50, 100], index=1)

    queue = students[
        students["risk_segment"].isin(segment) &
        (students["risk_score"] >= min_risk)
    ].copy()

    queue["recommended_action"] = np.select(
        [
            queue["days_since_last_activity"] >= 6,
            queue["task_attempts_d14"] <= 8,
            queue["video_watch_pct"] <= .35,
        ],
        [
            "Personal reactivation outreach",
            "Study-plan + academic support",
            "Alternative content / learning-path nudge",
        ],
        default="Curator check-in",
    )

    queue = queue.sort_values("risk_score", ascending=False)

    st.dataframe(
        queue[
            [
                "student_id",
                "risk_score",
                "risk_segment",
                "active_days_d14",
                "days_since_last_activity",
                "task_attempts_d14",
                "video_watch_pct",
                "recommended_action",
            ]
        ].head(max_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.2f"),
            "video_watch_pct": st.column_config.ProgressColumn("Video depth", min_value=0, max_value=1, format="%.0%%"),
        },
    )

    st.caption(
        f"{len(queue):,} students match the current filters. "
        "In production, this queue can be exported to CRM / curator workflow."
    )

# ---------- Page 4 ----------
elif page == "Model quality":
    section_title(
        "Model quality",
        "For retention operations, ranking quality at the available contact capacity is more useful than accuracy alone.",
    )

    cols = st.columns(len(model_metrics))
    for c, (_, row) in zip(cols, model_metrics.iterrows()):
        with c:
            metric = row["metric"]
            val = row["value"]
            if "Lift" in metric:
                shown = f"{val:.2f}×"
            else:
                shown = f"{val:.2f}"
            card(metric, shown)

    left, right = st.columns(2)

    with left:
        st.subheader("Top D14 signals")
        imp = feature_imp.sort_values("importance")
        fig = px.bar(
            imp,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance":"Importance","feature":""},
        )
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Risk concentration")
        tmp = students.sort_values("risk_score", ascending=False).reset_index(drop=True)
        tmp["decile"] = pd.qcut(tmp.index + 1, 10, labels=False) + 1
        dec = tmp.groupby("decile").agg(
            churn_rate=("observed_churn","mean"),
            students=("student_id","size"),
        ).reset_index()
        base = tmp["observed_churn"].mean()
        dec["lift"] = dec["churn_rate"] / base

        fig = px.bar(
            dec,
            x="decile",
            y="lift",
            labels={"decile":"Risk decile (1 = highest risk)","lift":"Lift vs baseline"},
        )
        fig.add_hline(y=1, line_dash="dash")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Portfolio note: values shown in demo mode are illustrative. "
        "When prepared model outputs are added to data/, the dashboard switches to real project metrics automatically."
    )

# ---------- Page 5 ----------
elif page == "Impact simulator":
    section_title(
        "Intervention impact simulator",
        "This separates model targeting quality from causal intervention effectiveness.",
    )

    c1, c2 = st.columns(2)
    with c1:
        cohort_size = st.number_input("Students in cohort", min_value=100, max_value=100000, value=1000, step=100)
        contact_share = st.slider("Share we can contact", .05, .50, .20, .05)
        precision = st.slider("Precision among contacted high-risk students", .05, .95, .58, .01)
    with c2:
        intervention_uplift = st.slider(
            "Intervention uplift among true-risk students",
            .00, .50, .10, .01,
            help="Causal uplift should come from an A/B test.",
        )
        completion_conversion = st.slider(
            "Share of additionally retained students who ultimately complete the course",
            .10, 1.00, .55, .05,
        )

    contacted = cohort_size * contact_share
    true_risk = contacted * precision
    retained = true_risk * intervention_uplift
    incremental_completers = retained * completion_conversion

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        card("Contacted", f"{contacted:,.0f}", f"{contact_share:.0%} of cohort")
    with r2:
        card("True-risk reached", f"{true_risk:,.0f}", f"Precision {precision:.0%}")
    with r3:
        card("Incrementally retained", f"{retained:,.1f}", f"Assumed uplift {intervention_uplift:.0%}")
    with r4:
        card("Incremental completers", f"{incremental_completers:,.1f}", "Expected North Star contribution")

    st.write("")
    st.markdown(
        """
        **How to use this simulator**

        - Replace precision with the observed Precision@Top-K from the model.
        - Replace intervention uplift with the causal estimate from an experiment.
        - Replace completion conversion with observed downstream conversion.

        This prevents the ML model from being credited for business impact that actually depends on the intervention.
        """
    )
