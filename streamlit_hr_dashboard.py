# ── UI labels (avoid duplicated literals; fixes Sonar S1192)
LABEL_TOTAL               = "Total Employees"
LABEL_ATTRITION_RATE      = "Attrition Rate"
LABEL_EARLY_ATTRITION     = "Early (≤1y) Attrition"
LABEL_AVG_TENURE          = "Avg Tenure (yrs)"
LABEL_AVG_SAT             = "Avg Job Satisfaction (1-4)"
LABEL_AVG_INCOME          = "Avg Monthly Income"
LABEL_DELTA_VS_OVERALL    = "Δ_vs_overall"
LABEL_EMPLOYEES_ANALYZED  = "Employees analyzed"
LABEL_REPORTING_EFF       = "Reporting efficiency"

DEFAULT_EMPLOYEE_COUNT    = 1470       # numeric fallback (not pre-formatted string)
DEFAULT_ATTRITION_RATE    = 0.161      # 16.1% fallback

# ---- Project website / case study metadata ----
REPORTING_EFFICIENCY_IMPROVEMENT = 40  # % vs Excel
TOP_TURNOVER_FACTORS = ["OverTime", "JobRole", "YearsAtCompany", "BusinessTravel", "MonthlyIncome"]
REPO_URL = "https://github.com/rvuppala-sketch/hr-attrition-analytics"
DEMO_URL = "https://<your-heroku-or-streamlit-url>"

import os
from math import sqrt
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st

# ML (driver model)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Optional survival analysis
try:
    from lifelines import KaplanMeierFitter
    HAVE_LIFELINES = True
except Exception:
    HAVE_LIFELINES = False

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

# -----------------------------
# Helpers
# -----------------------------

def fmt_num(x: float | int | None) -> str:
    return f"{x:,.0f}" if x is not None and pd.notnull(x) else "—"

def fmt_pct(x: float | None) -> str:
    return f"{x*100:.1f}%" if x is not None and pd.notnull(x) else "—"

def fmt_float(x: float | None) -> str:
    return f"{x:.2f}" if x is not None and pd.notnull(x) else "—"

def tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = tidy_columns(df)
    if "Attrition" in df.columns:
        df["Attrition_Flag"] = (df["Attrition"].astype(str).str.strip().str.upper() == "YES").astype(int)
    else:
        st.warning("Column 'Attrition' was not found; 'Attrition_Flag' cannot be created.")
        df["Attrition_Flag"] = 0

    # Age bands
    if "Age" in df.columns:
        bins_age = [17, 25, 30, 35, 40, 45, 50, 60]
        labels_age = ["18-25","26-30","31-35","36-40","41-45","46-50","51-60"]
        df["Age_Band"] = pd.cut(df["Age"], bins=bins_age, labels=labels_age, include_lowest=True)
    # Tenure bands
    if "YearsAtCompany" in df.columns:
        bins_tenure = [-1, 1, 3, 5, 10, 20, 40]
        labels_tenure = ["<1","1-3","3-5","5-10","10-20","20+"]
        df["YearsAtCompany_Band"] = pd.cut(df["YearsAtCompany"], bins=bins_tenure, labels=labels_tenure)
    # Income band
    if "MonthlyIncome" in df.columns and df["MonthlyIncome"].notna().sum() > 0:
        try:
            df["MonthlyIncome_Band"] = pd.qcut(df["MonthlyIncome"], q=5, duplicates="drop")
        except Exception:
            pass
    return df

def kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    attr_rate = df["Attrition_Flag"].mean() if total else np.nan
    avg_tenure = df["YearsAtCompany"].mean() if "YearsAtCompany" in df.columns else np.nan
    early_tenure_rate = np.nan
    if "YearsAtCompany" in df.columns:
        early = df[df["YearsAtCompany"] <= 1]
        early_tenure_rate = early["Attrition_Flag"].mean() if len(early) else np.nan
    avg_sat = df["JobSatisfaction"].mean() if "JobSatisfaction" in df.columns else np.nan
    avg_income = df["MonthlyIncome"].mean() if "MonthlyIncome" in df.columns else np.nan
    return {
        LABEL_TOTAL:           total,
        LABEL_ATTRITION_RATE:  attr_rate,
        LABEL_EARLY_ATTRITION: early_tenure_rate,
        LABEL_AVG_TENURE:      avg_tenure,
        LABEL_AVG_SAT:         avg_sat,
        LABEL_AVG_INCOME:      avg_income,
    }

def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0 or pd.isna(p):
        return (np.nan, np.nan)
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    margin = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - margin, center + margin

def attrition_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Slice attrition by a categorical column and attach Wilson CIs."""
    if col not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby(col)["Attrition_Flag"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Attrition_Rate", "count": "Employees"})
        .reset_index()
    )
    ci = g.apply(lambda r: _wilson_ci(r["Attrition_Rate"], int(r["Employees"])), axis=1)
    g["CI_low"], g["CI_high"] = zip(*ci)
    g = g.sort_values("Attrition_Rate", ascending=False).reset_index(drop=True)
    return g

def plot_bar(df_cat: pd.DataFrame, x: str, y: str, title: str, y_label: str | None = None) -> None:
    fig, ax = plt.subplots()
    df_cat.plot(kind="bar", x=x, y=y, legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y_label if y_label else y)
    # If y looks like a rate, show percentage ticks and labels
    if y.lower().endswith("rate") or (y_label and "rate" in y_label.lower()):
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        for p in ax.patches:
            h = p.get_height()
            if pd.notnull(h):
                ax.annotate(f"{h:.1%}", (p.get_x() + p.get_width()/2, h),
                            ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

def build_driver_model(df: pd.DataFrame, use_cols: list[str]) -> dict:
    """Train a simple logistic regression to estimate drivers. Returns model, feature importances."""
    out: dict[str, object] = {"model": None, "importances": None, "details": ""}
    if "Attrition_Flag" not in df.columns:
        out["details"] = "Attrition_Flag missing."
        return out

    cols_present = [c for c in use_cols if c in df.columns]
    if not cols_present:
        out["details"] = "No selected feature columns are present in the data."
        return out

    data = df[cols_present + ["Attrition_Flag"]].dropna()
    if data["Attrition_Flag"].nunique() < 2:
        out["details"] = "Need both attrition classes to train the model."
        return out

    # Separate categorical vs numeric
    cat_cols = [c for c in cols_present if data[c].dtype == "object" or str(data[c].dtype).startswith("category")]
    for c in cols_present:  # treat small-cardinality ints as categorical
        if c not in cat_cols and pd.api.types.is_integer_dtype(data[c]) and data[c].nunique() <= 8:
            cat_cols.append(c)
    num_cols = [c for c in cols_present if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=400, class_weight="balanced")
    pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])
    X = data[cols_present]
    y = data["Attrition_Flag"].values
    pipe.fit(X, y)

    # Feature names and coefficients
    try:
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(len(pipe.named_steps["clf"].coef_.ravel()))]

    coefs = pipe.named_steps["clf"].coef_.ravel()
    imp = pd.DataFrame({"feature": feature_names, "coef": coefs})
    imp["abs_coef"] = imp["coef"].abs()

    # Group importances back to root feature (before one-hot)
    def root_name(name: str) -> str:
        parts = name.split("__")
        last = parts[-1] if parts and parts[0] in ("cat", "num", "prep") else name
        root = last.split("_")[0]
        return root

    imp["root"] = [root_name(f) for f in imp["feature"]]
    grouped = imp.groupby("root")["abs_coef"].sum().sort_values(ascending=False).reset_index()
    grouped = grouped.rename(columns={"abs_coef": "importance"})

    out["model"] = pipe
    out["importances"] = grouped
    out["details"] = f"Fitted on {len(data)} rows, {len(cols_present)} features ({len(cat_cols)} categorical, {len(num_cols)} numeric)."
    return out

def equity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Read-only disparities vs overall for protected attributes."""
    rows: list[pd.DataFrame] = []
    overall = df["Attrition_Flag"].mean() if len(df) else np.nan
    for col in ["Gender", "Age_Band"]:
        if col in df.columns:
            g = df.groupby(col)["Attrition_Flag"].mean().reset_index().rename(columns={"Attrition_Flag": "Attrition_Rate"})
            g["Group"] = col
            g["Delta_vs_Overall"] = g["Attrition_Rate"] - overall
            rows.append(g[["Group", col, "Attrition_Rate", "Delta_vs_Overall"]].rename(columns={col: "Value"}))
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()

# --- Case Study / Project Website view ---
def render_case_study(df: pd.DataFrame | None) -> None:
    st.title("HR Analytics Dashboard — Case Study")
    st.caption("Python • Pandas • scikit-learn • Streamlit • Docker • CI/CD • Heroku")

    # Top metrics
    cols = st.columns(3)
    cols[0].metric(LABEL_REPORTING_EFF, f"{REPORTING_EFFICIENCY_IMPROVEMENT}%", "vs Excel")
    if df is not None and "Attrition_Flag" in df.columns:
        cols[1].metric(LABEL_EMPLOYEES_ANALYZED, fmt_num(len(df)))
        cols[2].metric(LABEL_ATTRITION_RATE, fmt_pct(df["Attrition_Flag"].mean()))
    else:
        cols[1].metric(LABEL_EMPLOYEES_ANALYZED, fmt_num(DEFAULT_EMPLOYEE_COUNT))
        cols[2].metric(LABEL_ATTRITION_RATE, fmt_pct(DEFAULT_ATTRITION_RATE))

    st.divider()
    st.subheader("What I built")
    st.markdown(
        """
- **Interactive dashboard (Streamlit)** with KPI tiles, hotspot analysis by department/role/tenure, and Excel export.
- **Python + Pandas** data processing; engineered tenure/age bands and a clean attrition target.
- **Scikit-learn logistic model** to rank drivers; surfaced **five turnover factors** (overtime, job role, tenure, travel, pay).
- **Deployment**: containerized with `requirements.txt` and deployed to **Heroku**; produced a shareable URL for stakeholder reviews.
        """
    )

    # Simple drivers bar (illustrative values—replace with your export if desired)
    st.subheader("Top Turnover Drivers (feature importance)")
    drivers = pd.DataFrame(
        {"factor": TOP_TURNOVER_FACTORS, "importance": [0.62, 0.38, 0.35, 0.19, 0.14][:len(TOP_TURNOVER_FACTORS)]}
    )
    fig, ax = plt.subplots()
    drivers.plot(kind="bar", x="factor", y="importance", legend=False, ax=ax)
    ax.set_xlabel("Factor"); ax.set_ylabel("Relative importance"); ax.set_title("Model-identified drivers")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h*100:.0f}%", (p.get_x()+p.get_width()/2, h), ha="center", va="bottom", fontsize=8)
    st.pyplot(fig)
    st.caption("Note: Above importances are illustrative; replace with your logistic model export if you want exact numbers.")

    st.divider()
    st.subheader("Links")
    st.markdown(f"- **Live demo:** {DEMO_URL}")
    st.markdown(f"- **Source code:** {REPO_URL}")

    st.divider()
    if df is not None and {"Department","Attrition_Flag"}.issubset(df.columns):
        st.subheader("Live snapshot")
        g = (
            df.groupby("Department")["Attrition_Flag"]
              .agg(["mean","count"]).rename(columns={"mean":"Attrition_Rate","count":"Employees"})
              .sort_values("Attrition_Rate", ascending=False).reset_index()
        )
        st.dataframe(g, use_container_width=True)
    else:
        st.info("Drop the IBM CSV next to this file as **WA_Fn-UseC_-HR-Employee-Attrition.csv** to show a live snapshot here.")

# -----------------------------
# Sidebar: Data input & filters
# -----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload IBM HR Analytics CSV", type=["csv"])

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    # Local fallback: look for a CSV in the same folder as this script
    local_csv = Path(__file__).with_name("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if local_csv.exists():
        df = pd.read_csv(local_csv)
        st.sidebar.info(f"Using sample dataset: {local_csv.name}")
    else:
        st.sidebar.warning("Please upload the IBM HR CSV to proceed.")

if df is not None:
    df = add_derived_columns(df)

# Filters
if df is not None:
    st.sidebar.header("Filters")
    filt_cols = ["Department", "JobRole", "OverTime", "JobLevel", "BusinessTravel", "YearsAtCompany_Band", "Age_Band"]
    active_filters: dict[str, list[str]] = {}
    for c in filt_cols:
        if c in df.columns:
            vals = [str(v) for v in sorted(df[c].dropna().unique().tolist())]
            sel = st.sidebar.multiselect(c, vals, default=vals)
            active_filters[c] = sel

    # Apply filters
    mask = pd.Series(True, index=df.index)
    for c, sel in active_filters.items():
        if c in df.columns and sel and len(sel) != len(df[c].dropna().unique()):
            mask &= df[c].astype(str).isin(sel)
    df_filt = df[mask].copy()
else:
    df_filt = None

# -----------------------------
# View switch (Dashboard / Case Study)
# -----------------------------
view = st.sidebar.radio("View", ["Dashboard", "Case Study"], index=0)

if view == "Case Study":
    render_case_study(df)
    st.stop()

# -----------------------------
# Dashboard layout
# -----------------------------
st.title("HR Attrition Dashboard")

if df_filt is None or len(df_filt) == 0:
    st.info("Upload a CSV and/or adjust filters to view analytics.")
    st.stop()

# KPIs
st.subheader("Key Metrics")
k = kpis(df_filt)
kpi_cols = st.columns(6)
kpi_cols[0].metric(LABEL_TOTAL,           fmt_num(k[LABEL_TOTAL]))
kpi_cols[1].metric(LABEL_ATTRITION_RATE,  fmt_pct(k[LABEL_ATTRITION_RATE]))
kpi_cols[2].metric(LABEL_EARLY_ATTRITION, fmt_pct(k[LABEL_EARLY_ATTRITION]))
kpi_cols[3].metric(LABEL_AVG_TENURE,      fmt_float(k[LABEL_AVG_TENURE]))
kpi_cols[4].metric(LABEL_AVG_SAT,         fmt_float(k[LABEL_AVG_SAT]))
kpi_cols[5].metric(LABEL_AVG_INCOME,      fmt_num(k[LABEL_AVG_INCOME]))

st.caption("Tip: Use the sidebar to focus on a department/job role or tenure window.")

# Hotspots
st.header("Hotspots")
hot_cols = ["Department", "JobRole", "JobLevel", "OverTime", "BusinessTravel", "YearsAtCompany_Band", "Age_Band"]
hot_tabs = st.tabs([f"By {c}" for c in hot_cols])
for tab, c in zip(hot_tabs, hot_cols):
    with tab:
        if c in df_filt.columns:
            tbl = attrition_by(df_filt, c)
            st.dataframe(tbl, use_container_width=True)
            if not tbl.empty:
                plot_bar(tbl, c, "Attrition_Rate", f"{LABEL_ATTRITION_RATE} by {c}", y_label=LABEL_ATTRITION_RATE)

def _as_percent(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: f"{v:.1%}" if pd.notnull(v) else "—")
    return df

# inside the loop where you render each tab:
tbl = attrition_by(df_filt, c)
if not tbl.empty:
    tbl_fmt = _as_percent(tbl, ["Attrition_Rate", "CI_low", "CI_high"])
    st.dataframe(tbl_fmt, use_container_width=True)
    plot_bar(tbl, c, "Attrition_Rate", f"{LABEL_ATTRITION_RATE} by {c}", y_label=LABEL_ATTRITION_RATE)

# Largest deltas
st.subheader("Largest Attrition Gaps (vs overall)")
overall = df_filt["Attrition_Flag"].mean()

def top_deltas(col: str) -> pd.DataFrame:
    t = attrition_by(df_filt, col)
    if t.empty:
        return t
    t[LABEL_DELTA_VS_OVERALL] = t["Attrition_Rate"] - overall
    return t.sort_values(LABEL_DELTA_VS_OVERALL, ascending=False)[[col, "Attrition_Rate", "Employees", LABEL_DELTA_VS_OVERALL]].head(5)

cols = st.columns(3)
with cols[0]:
    st.caption("By Department"); st.dataframe(top_deltas("Department"), use_container_width=True)
with cols[1]:
    st.caption("By JobRole"); st.dataframe(top_deltas("JobRole"), use_container_width=True)
with cols[2]:
    st.caption("By OverTime"); st.dataframe(top_deltas("OverTime"), use_container_width=True)

# Driver model
st.header("Drivers (Logistic Regression)")
st.caption("Shows which factors move attrition odds up/down within the current filtered population.")

default_features = [
    "OverTime", "BusinessTravel", "DistanceFromHome",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
    "JobRole", "Department", "JobLevel",
    "MonthlyIncome", "StockOptionLevel",
    "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance",
    "TrainingTimesLastYear"
]
selectable = [c for c in default_features if c in df_filt.columns]
chosen = st.multiselect("Driver features to include", options=selectable, default=selectable)

if chosen:
    res = build_driver_model(df_filt, chosen)
    if res["model"] is None:
        st.warning(f"Driver model not available: {res['details']}")
    else:
        st.success(res["details"])
        imp = res["importances"]
        st.dataframe(imp, use_container_width=True)
        fig, ax = plt.subplots()
        imp.plot(kind="bar", x="root", y="importance", legend=False, ax=ax)
        ax.set_title("Feature Group Importances (|coef| aggregated)")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        fig.tight_layout()
        st.pyplot(fig)
else:
    st.info("Select at least one feature to fit the driver model.")

# Survival analysis (optional)
st.header("Tenure Survival (Kaplan–Meier)")
if HAVE_LIFELINES and "YearsAtCompany" in df_filt.columns:
    kmf = KaplanMeierFitter()
    T = df_filt["YearsAtCompany"].astype(float)
    E = df_filt["Attrition_Flag"].astype(int)
    try:
        kmf.fit(T, event_observed=E, label="All Employees")
        fig, ax = plt.subplots()
        kmf.plot(ax=ax)
        ax.set_title("Survival (1 - Cumulative Attrition) by Tenure")
        ax.set_xlabel("Years at Company")
        ax.set_ylabel("Survival Probability")
        fig.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not compute survival curve: {e}")
else:
    st.info("Install 'lifelines' to enable tenure survival curves (pip install lifelines).")

# Equity monitor
st.header("Equity Monitor (Read-only)")
eq = equity_table(df_filt)
if not eq.empty:
    st.dataframe(eq, use_container_width=True)
else:
    st.info("No protected attribute columns available for equity view.")

# Export
st.header("Export")
export_btn = st.button("Download filtered summary as Excel")

if export_btn:
    summary_tables: dict[str, pd.DataFrame] = {}
    summary_tables["KPIs"] = pd.DataFrame([k])
    for c in hot_cols:
        if c in df_filt.columns:
            summary_tables[f"by_{c}"] = attrition_by(df_filt, c)
    if chosen and 'res' in locals() and res.get("importances") is not None:
        summary_tables["driver_importances"] = res["importances"]

    xls_path = "hr_dashboard_export.xlsx"
    with pd.ExcelWriter(xls_path, engine="xlsxwriter") as writer:
        for sheet, data in summary_tables.items():
            data.to_excel(writer, sheet_name=sheet[:31], index=False)
        df_filt.to_excel(writer, sheet_name="filtered_rows", index=False)
    with open(xls_path, "rb") as f:
        st.download_button("Download Excel", data=f, file_name="hr_dashboard_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Built with Streamlit • Logistic Regression for drivers • Optional lifelines for survival.")
