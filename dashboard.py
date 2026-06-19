"""
dashboard.py  —  Streamlit Dashboard
Run:  streamlit run dashboard.py
Open: http://localhost:8501
"""

import json, os, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance ML",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── MINIMAL CSS — only targets custom HTML elements, never native Streamlit ──
st.markdown("""
<style>
/* Grade badge — only used inside st.markdown() custom HTML */
.grade-badge {
    display: inline-block;
    font-size: 52px;
    font-weight: 800;
    line-height: 1;
    padding: 8px 26px;
    border-radius: 8px;
    margin-bottom: 4px;
}
.g-A { background: #d4f0e2; color: #1a6640; }
.g-B { background: #d4edd8; color: #2a5e40; }
.g-C { background: #fdefc3; color: #8a6000; }
.g-D { background: #ffe0b2; color: #8a4000; }
.g-F { background: #fce4e1; color: #8a2020; }

/* Probability bar — only inside st.markdown() */
.prob-row { display:flex; align-items:center; gap:10px; margin-bottom:9px; }
.prob-lbl { width:18px; font-weight:800; font-size:13px; font-family:monospace; }
.prob-track { flex:1; background:#e0ddd8; border-radius:3px; height:10px; overflow:hidden; }
.prob-fill  { height:100%; border-radius:3px; }
.prob-val   { width:44px; text-align:right; font-size:12px; font-family:monospace; opacity:.75; }

/* Section divider label */
.sec-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    opacity: .55;
    margin: 18px 0 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MDL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CLASS_NAMES  = ["A", "B", "C", "D", "F"]
GRADE_COLORS = {"A":"#1a6640","B":"#2a5e40","C":"#c9852b","D":"#b5651d","F":"#a33b2c"}
GRADE_BG     = {"A":"#d4f0e2","B":"#d4edd8","C":"#fdefc3","D":"#ffe0b2","F":"#fce4e1"}
GRADE_DESC   = {
    "A": "Excellent  —  GPA >= 3.5",
    "B": "Good  —  3.0 <= GPA < 3.5",
    "C": "Satisfactory  —  2.5 <= GPA < 3.0",
    "D": "Needs Improvement  —  2.0 <= GPA < 2.5",
    "F": "At Risk  —  GPA < 2.0",
}
PALETTE = {"Logistic Regression": "#14213d", "Random Forest": "#c9852b"}
FEATURE_COLS = [
    "Age","Gender","Ethnicity","ParentalEducation",
    "StudyTimeWeekly","Absences","Tutoring","ParentalSupport",
    "Extracurricular","Sports","Music","Volunteering",
]

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return (
        joblib.load(f"{MDL}/logistic_regression.pkl"),
        joblib.load(f"{MDL}/random_forest.pkl"),
        joblib.load(f"{MDL}/scaler.pkl"),
    )

@st.cache_data
def load_json(name):
    with open(f"{MDL}/{name}") as f:
        return json.load(f)

lr_model, rf_model, scaler = load_models()
metrics    = load_json("metrics.json")
feat_imp   = load_json("feature_importance.json")
roc_data   = load_json("roc_data.json")
ds_summary = load_json("dataset_summary.json")

# ── Prediction helper ──────────────────────────────────────────────────────────
def predict_both(feat_dict):
    row = np.array([[float(feat_dict[c]) for c in FEATURE_COLS]])
    lr_idx  = int(lr_model.predict(scaler.transform(row))[0])
    lr_prob = lr_model.predict_proba(scaler.transform(row))[0]
    rf_idx  = int(rf_model.predict(row)[0])
    rf_prob = rf_model.predict_proba(row)[0]
    return {
        "Logistic Regression": {"grade": CLASS_NAMES[lr_idx], "probs": lr_prob},
        "Random Forest":       {"grade": CLASS_NAMES[rf_idx], "probs": rf_prob},
    }

def parse_code(s):
    return int(s.split("(")[1].rstrip(")"))

# ── Plotly default theme (no reliance on Streamlit theme) ─────────────────────
PLOTLY_LAYOUT = dict(
    plot_bgcolor  = "#ffffff",
    paper_bgcolor = "#f6f3ec",
    font          = dict(family="Arial, sans-serif", size=13, color="#14213d"),
    margin        = dict(l=0, r=0, t=36, b=0),
)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("Student Profile")
    st.caption("Fill in the student details. Both models predict instantly.")
    st.divider()

    # --- Demographics ---
    st.markdown('<div class="sec-label">Demographics</div>', unsafe_allow_html=True)
    age    = st.selectbox("Age (years)", [15, 16, 17, 18], index=1)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    eth_opts = ["Caucasian", "African American", "Asian", "Other"]
    ethnicity = st.selectbox("Ethnicity", eth_opts)
    par_edu_opts = ["None", "High School", "Some College", "Bachelor's", "Higher"]
    par_edu = st.selectbox("Parental Education", par_edu_opts, index=2)

    # --- Academic ---
    st.markdown('<div class="sec-label">Academic Behaviour</div>', unsafe_allow_html=True)
    study_time = st.slider("Weekly Study Time (hrs)", 0.0, 20.0, 10.0, 0.5)
    absences   = st.slider("Absences (days/year)", 0, 30, 5)
    tutoring   = st.radio("Tutoring", ["No", "Yes"], horizontal=True)

    # --- Support ---
    st.markdown('<div class="sec-label">Support & Activities</div>', unsafe_allow_html=True)
    par_sup_opts = ["None", "Low", "Moderate", "High", "Very High"]
    par_sup = st.selectbox("Parental Support", par_sup_opts, index=2)
    extra    = st.radio("Extracurricular", ["No", "Yes"], horizontal=True)
    sports   = st.radio("Sports",          ["No", "Yes"], horizontal=True)
    music    = st.radio("Music",           ["No", "Yes"], horizontal=True)
    volunt   = st.radio("Volunteering",    ["No", "Yes"], horizontal=True)

features = {
    "Age":               age,
    "Gender":            0 if gender == "Male" else 1,
    "Ethnicity":         eth_opts.index(ethnicity),
    "ParentalEducation": par_edu_opts.index(par_edu),
    "StudyTimeWeekly":   study_time,
    "Absences":          absences,
    "Tutoring":          0 if tutoring == "No" else 1,
    "ParentalSupport":   par_sup_opts.index(par_sup),
    "Extracurricular":   0 if extra == "No" else 1,
    "Sports":            0 if sports == "No" else 1,
    "Music":             0 if music == "No" else 1,
    "Volunteering":      0 if volunt == "No" else 1,
}

preds = predict_both(features)

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# Student Performance ML Dashboard")
c1, c2, c3 = st.columns(3)
c1.info("Logistic Regression")
c2.info("Random Forest")
c3.info("Local Inference — No API")

# ═════════════════════════════════════════════════════════════════════════════
# TABS  (plain text labels — no icons, avoids Material Icon raw-text bug)
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Live Prediction",
    "Model Comparison",
    "Confusion Matrices",
    "ROC Curves",
    "Feature Importance",
    "Dataset Overview",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  —  LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    lr_g = preds["Logistic Regression"]["grade"]
    rf_g = preds["Random Forest"]["grade"]
    lr_p = preds["Logistic Regression"]["probs"]
    rf_p = preds["Random Forest"]["probs"]

    # Agreement banner
    if lr_g == rf_g:
        st.success(f"Both models agree  —  Predicted Grade: **{lr_g}**  ({GRADE_DESC[lr_g]})")
    else:
        st.warning(f"Models disagree  —  Logistic Regression: **{lr_g}**  |  Random Forest: **{rf_g}**")

    col_lr, col_rf = st.columns(2)

    for col, mname, grade, probs in [
        (col_lr, "Logistic Regression", lr_g, lr_p),
        (col_rf, "Random Forest",       rf_g, rf_p),
    ]:
        with col:
            st.subheader(mname)
            # Grade badge using safe HTML
            st.markdown(
                f'<div class="grade-badge g-{grade}">{grade}</div>'
                f'<p style="margin:6px 0 18px;font-size:13px;opacity:.7;">{GRADE_DESC[grade]}</p>',
                unsafe_allow_html=True,
            )
            # Probability bars — pure HTML, no native Streamlit widget
            bars_html = ""
            for cls, prob in zip(CLASS_NAMES, probs):
                pct  = round(prob * 100, 1)
                fill = GRADE_COLORS[cls]
                bars_html += f"""
                <div class="prob-row">
                  <span class="prob-lbl">{cls}</span>
                  <div class="prob-track">
                    <div class="prob-fill" style="width:{pct}%;background:{fill};"></div>
                  </div>
                  <span class="prob-val">{pct}%</span>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

    # Comparison chart
    st.markdown("---")
    st.subheader("Probability Comparison — Both Models")
    fig_cmp = go.Figure()
    for mname, probs, color in [
        ("Logistic Regression", lr_p, PALETTE["Logistic Regression"]),
        ("Random Forest",       rf_p, PALETTE["Random Forest"]),
    ]:
        fig_cmp.add_trace(go.Bar(
            name=mname,
            x=CLASS_NAMES,
            y=[round(p * 100, 1) for p in probs],
            marker_color=color,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
            width=0.35,
        ))
    fig_cmp.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        height=320,
        yaxis=dict(title="Probability (%)", gridcolor="#e0ddd8", range=[0, max(max(lr_p), max(rf_p)) * 140]),
        xaxis=dict(title="Grade Class"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

    # Student profile expander
    with st.expander("View Current Student Profile"):
        labels = feat_imp.get("labels", {})
        rows = [{"Feature": labels.get(k, k), "Value": v} for k, v in features.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  —  MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Performance Metrics")
    lr_m = metrics["Logistic Regression"]
    rf_m = metrics["Random Forest"]

    # KPI row
    cols = st.columns(6)
    kpi_items = [
        ("Accuracy",   lr_m["accuracy"],      rf_m["accuracy"],      True,  "%.1f%%", 100),
        ("Precision",  lr_m["precision"],     rf_m["precision"],     True,  "%.1f%%", 100),
        ("Recall",     lr_m["recall"],        rf_m["recall"],        True,  "%.1f%%", 100),
        ("F1-Score",   lr_m["f1_score"],      rf_m["f1_score"],      True,  "%.1f%%", 100),
        ("ROC-AUC",    lr_m["roc_auc"],       rf_m["roc_auc"],       True,  "%.4f",   1),
        ("Train Time", lr_m["train_time_sec"],rf_m["train_time_sec"],False, "%.3fs",  1),
    ]
    for col, (label, lv, rv, higher_better, fmt, mult) in zip(cols, kpi_items):
        d = round((rv - lv) * mult * (1 if higher_better else -1), 3)
        col.metric(
            f"LR {label}",
            fmt % (lv * mult),
            f"RF {'▲' if d > 0 else '▼'} {abs(d):.2f}",
        )

    # Table
    st.subheader("Side-by-Side Comparison")
    rows = []
    for label, lv, rv, higher_better, fmt, mult in kpi_items:
        lr_better = (lv >= rv) if higher_better else (lv <= rv)
        rows.append({
            "Metric":               label,
            "Logistic Regression":  fmt % (lv * mult),
            "Random Forest":        fmt % (rv * mult),
            "Winner":               "LR" if lr_better else "RF",
        })
    df_cmp = pd.DataFrame(rows)
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    # Bar chart
    st.subheader("Metric Comparison Chart")
    metric_lbls = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    lr_vals = [lr_m["accuracy"], lr_m["precision"], lr_m["recall"], lr_m["f1_score"], lr_m["roc_auc"]]
    rf_vals = [rf_m["accuracy"], rf_m["precision"], rf_m["recall"], rf_m["f1_score"], rf_m["roc_auc"]]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Logistic Regression", x=metric_lbls, y=lr_vals,
        marker_color=PALETTE["Logistic Regression"],
        text=[f"{v:.3f}" for v in lr_vals], textposition="outside",
    ))
    fig_bar.add_trace(go.Bar(
        name="Random Forest", x=metric_lbls, y=rf_vals,
        marker_color=PALETTE["Random Forest"],
        text=[f"{v:.3f}" for v in rf_vals], textposition="outside",
    ))
    fig_bar.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group", height=360,
        yaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="#e0ddd8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # Training time
    st.subheader("Training Time")
    fig_time = go.Figure(go.Bar(
        x=["Logistic Regression", "Random Forest"],
        y=[lr_m["train_time_sec"], rf_m["train_time_sec"]],
        marker_color=[PALETTE["Logistic Regression"], PALETTE["Random Forest"]],
        text=[f"{lr_m['train_time_sec']:.4f}s", f"{rf_m['train_time_sec']:.4f}s"],
        textposition="outside",
    ))
    fig_time.update_layout(
        **PLOTLY_LAYOUT, height=260,
        yaxis=dict(title="Seconds", gridcolor="#e0ddd8"),
        showlegend=False,
    )
    st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  —  CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Confusion Matrices")
    st.caption("Rows = Actual class   |   Columns = Predicted class   |   Diagonal = correct predictions")

    col_l, col_r = st.columns(2)
    for col, mname in [(col_l, "Logistic Regression"), (col_r, "Random Forest")]:
        cm = np.array(metrics[mname]["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=CLASS_NAMES, y=CLASS_NAMES,
            text_auto=True,
            color_continuous_scale=[[0, "#f0ece4"], [0.5, "#7fa8c9"], [1, "#14213d"]],
        )
        fig_cm.update_layout(
            title=dict(text=f"<b>{mname}</b>", font=dict(size=14, color="#14213d")),
            height=380,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Arial, sans-serif", size=14, color="#14213d"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig_cm.update_traces(textfont=dict(size=16, color="#14213d"))
        with col:
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
            # Per-class recall
            recall_cols = st.columns(5)
            for i, cls in enumerate(CLASS_NAMES):
                total   = int(cm[i].sum())
                correct = int(cm[i][i])
                val     = f"{correct/total*100:.0f}%" if total > 0 else "N/A"
                recall_cols[i].metric(f"Grade {cls}", val, "recall")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  —  ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("ROC Curves  (One-vs-Rest)")
    st.caption("Higher AUC = better ability to separate that grade class from the rest.")

    sel_cls = st.radio("Show curve for grade:", CLASS_NAMES, horizontal=True)

    fig_roc = go.Figure()
    fig_roc.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="#aaaaaa", width=1),
    )
    for mname in ["Logistic Regression", "Random Forest"]:
        d = roc_data[mname][sel_cls]
        r, g, b = int(PALETTE[mname][1:3], 16), int(PALETTE[mname][3:5], 16), int(PALETTE[mname][5:], 16)
        fig_roc.add_trace(go.Scatter(
            x=d["fpr"], y=d["tpr"], mode="lines",
            name=f"{mname}  (AUC = {d['auc']:.4f})",
            line=dict(color=PALETTE[mname], width=2.5),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.07)",
        ))
    fig_roc.update_layout(
        **PLOTLY_LAYOUT, height=420,
        xaxis=dict(title="False Positive Rate", range=[0, 1], gridcolor="#e0ddd8"),
        yaxis=dict(title="True Positive Rate",  range=[0, 1.02], gridcolor="#e0ddd8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})

    # AUC summary table
    st.subheader("AUC Summary — All Grades")
    auc_rows = []
    for cls in CLASS_NAMES:
        la = roc_data["Logistic Regression"][cls]["auc"]
        ra = roc_data["Random Forest"][cls]["auc"]
        auc_rows.append({
            "Grade": cls,
            "LR AUC":  round(la, 4),
            "RF AUC":  round(ra, 4),
            "Better":  "LR" if la >= ra else "RF",
        })
    st.dataframe(pd.DataFrame(auc_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5  —  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Feature Importance")
    labels_map = feat_imp.get("labels", {})

    col_l, col_r = st.columns(2)
    specs = [
        (col_l, "Random Forest",       "Gini impurity reduction (native RF importance)"),
        (col_r, "Logistic Regression", "Mean |coefficient| across all 5 grade classes"),
    ]
    for col, mname, caption in specs:
        imp = feat_imp[mname]
        sorted_items = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        ylabels = [labels_map.get(k, k) for k, _ in sorted_items]
        vals    = [v for _, v in sorted_items]

        fig_fi = go.Figure(go.Bar(
            x=vals[::-1], y=ylabels[::-1],
            orientation="h",
            marker_color=PALETTE[mname],
            text=[f"{v:.4f}" for v in vals[::-1]],
            textposition="outside",
        ))
        fig_fi.update_layout(
            **{**PLOTLY_LAYOUT, "margin": dict(l=0, r=90, t=60, b=0)},
            title=dict(text=f"<b>{mname}</b><br><sup style='font-size:11px'>{caption}</sup>",
                       font=dict(size=13, color="#14213d")),
            height=430,
            xaxis=dict(gridcolor="#e0ddd8"),
            yaxis=dict(tickfont=dict(size=11)),
        )
        with col:
            st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    # Scatter comparison
    st.subheader("Feature Agreement Between Models")
    all_feats = list(feat_imp["Logistic Regression"].keys())
    sdf = pd.DataFrame({
        "Feature":            [labels_map.get(f, f) for f in all_feats],
        "Logistic Regression":[feat_imp["Logistic Regression"][f] for f in all_feats],
        "Random Forest":      [feat_imp["Random Forest"][f] for f in all_feats],
    })
    fig_sc = px.scatter(sdf, x="Logistic Regression", y="Random Forest", text="Feature")
    fig_sc.update_traces(
        textposition="top center",
        marker=dict(size=10, color="#c9852b", line=dict(color="#14213d", width=1.5)),
        textfont=dict(color="#14213d", size=11),
    )
    fig_sc.update_layout(
        **PLOTLY_LAYOUT, height=340,
        xaxis=dict(gridcolor="#e0ddd8"),
        yaxis=dict(gridcolor="#e0ddd8"),
    )
    st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6  —  DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Dataset Overview")
    st.markdown(
        "**Source:** Kaggle — *Students Performance Dataset* · Rabie El Kharoua · CC BY 4.0  \n"
        "https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset"
    )
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{ds_summary['n_rows']:,}")
    c2.metric("Features Used",    str(ds_summary["n_features"]))
    c3.metric("Training Samples", f"{ds_summary['train_size']:,}")
    c4.metric("Test Samples",     f"{ds_summary['test_size']:,}")

    # Class distribution
    st.subheader("Class Distribution")
    cd    = ds_summary["class_counts"]
    total = sum(cd.values())
    fig_cd = go.Figure(go.Bar(
        x=list(cd.keys()),
        y=list(cd.values()),
        marker_color=[GRADE_COLORS[g] for g in cd],
        text=[f"{v}  ({v/total*100:.1f}%)" for v in cd.values()],
        textposition="outside",
    ))
    fig_cd.update_layout(
        **PLOTLY_LAYOUT, height=300,
        xaxis=dict(title="Grade Class"),
        yaxis=dict(title="Number of Students", gridcolor="#e0ddd8"),
        showlegend=False,
    )
    st.plotly_chart(fig_cd, use_container_width=True, config={"displayModeBar": False})

    # Feature reference
    st.subheader("Feature Reference")
    feat_ref = [
        ("Age",               "Numerical",   "15 – 18 years"),
        ("Gender",            "Categorical", "0 = Male,  1 = Female"),
        ("Ethnicity",         "Categorical", "0 = Caucasian,  1 = African American,  2 = Asian,  3 = Other"),
        ("ParentalEducation", "Categorical", "0 = None,  1 = High School,  2 = Some College,  3 = Bachelor's,  4 = Higher"),
        ("StudyTimeWeekly",   "Numerical",   "0 – 20 hours/week"),
        ("Absences",          "Numerical",   "0 – 30 days/year"),
        ("Tutoring",          "Categorical", "0 = No,  1 = Yes"),
        ("ParentalSupport",   "Categorical", "0 = None  …  4 = Very High"),
        ("Extracurricular",   "Categorical", "0 = No,  1 = Yes"),
        ("Sports",            "Categorical", "0 = No,  1 = Yes"),
        ("Music",             "Categorical", "0 = No,  1 = Yes"),
        ("Volunteering",      "Categorical", "0 = No,  1 = Yes"),
        ("GPA  (EXCLUDED)",   "Numerical",   "0.0 – 4.0 — dropped to prevent data leakage"),
        ("GradeClass (TARGET)","Integer",    "0 = A,  1 = B,  2 = C,  3 = D,  4 = F"),
    ]
    st.dataframe(
        pd.DataFrame(feat_ref, columns=["Feature", "Type", "Description"]),
        use_container_width=True, hide_index=True,
    )

    # Preprocessing steps
    st.subheader("Preprocessing Steps (train_models.py)")
    steps = [
        ("1. Load CSV",          "Read Students_Performance_Dataset.csv with pandas"),
        ("2. Impute missing",    "SimpleImputer(strategy='median') for StudyTimeWeekly and Absences"),
        ("3. Remove duplicates", "df.drop_duplicates()"),
        ("4. Drop leakage cols", "Remove StudentID (identifier) and GPA (leaks target)"),
        ("5. Train/test split",  "80/20 stratified split  |  random_state=42"),
        ("6. StandardScaler",    "Fit on training set only — applied to LR input; RF does not need scaling"),
    ]
    st.dataframe(
        pd.DataFrame(steps, columns=["Step", "Detail"]),
        use_container_width=True, hide_index=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "ML Course Project  |  Logistic Regression + Random Forest  |  "
    "Kaggle Students Performance Dataset  |  All inference runs locally — no external APIs"
)