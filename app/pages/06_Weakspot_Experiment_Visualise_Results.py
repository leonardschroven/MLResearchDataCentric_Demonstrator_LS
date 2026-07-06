"""
Sweep Results — visual comparison of the 8 detection methods across all
parameter-sweep runs saved to `data/experiment_results/sweep_results.csv`.

Each CSV row is one (config × method) result. This page groups by method so
you can see, across many configs, which methods locate the induced weakspot
most accurately (IoU), most precisely (distance), and how their performance
varies with radius / sample size / training iterations / weakspot location.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Weakspot Experiment — Visualise Results", layout="wide")
st.title("📈 Weakspot Experiment — Visualise Results")
st.markdown(
    "Aggregated view of the parameter-sweep CSV. Each method is evaluated on "
    "many configurations (weakspot location, radius, training iterations, sample size); "
    "this page shows how they compare on **IoU vs the induced weakspot**, "
    "**centroid distance**, and **sensitivity to sweep parameters**."
)

CSV_PATH = Path("data/experiment_results/sweep_results.csv")

if not CSV_PATH.exists():
    st.warning(
        f"No sweep CSV found at `{CSV_PATH}`. "
        "Open **Weakspot Experiment** and run the parameter sweep first."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_results(path: Path, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_results(CSV_PATH, CSV_PATH.stat().st_mtime)

IOU_COLS = ["iou_q80", "iou_q85", "iou_q90", "iou_q95"]
SWEEP_DIMS = [
    "center_x", "center_y", "radius", "iterations", "n_samples",
    "n_bumps", "noise_std",
]

# All numeric per-run metrics we let the user analyse + whether higher is better.
METRIC_DIRECTION = {
    "iou_q80":   "higher",
    "iou_q85":   "higher",
    "iou_q90":   "higher",
    "iou_q95":   "higher",
    "distance":  "lower",
    "area_frac": "neither",
    "sigma_x1":  "neither",
    "sigma_x2":  "neither",
}
METRIC_OPTIONS = list(METRIC_DIRECTION.keys())

# ─────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Filters")
st.sidebar.caption(
    "Restrict the rows used in every chart below. "
    "Empty selection = keep all values for that dim."
)

filters = {}
for dim in SWEEP_DIMS:
    vals = sorted(df[dim].dropna().unique().tolist())
    filters[dim] = st.sidebar.multiselect(dim, vals, default=vals)

methods_all = sorted(df["method"].dropna().unique().tolist())
selected_methods = st.sidebar.multiselect(
    "method", methods_all, default=methods_all
)

st.sidebar.markdown("---")
primary_metric = st.sidebar.selectbox(
    "Primary metric (drives ranking, distributions, parameter-effect charts)",
    METRIC_OPTIONS,
    index=METRIC_OPTIONS.index("iou_q90"),
)
metric_dir = METRIC_DIRECTION[primary_metric]
HIGHER_IS_BETTER = metric_dir == "higher"
LOWER_IS_BETTER  = metric_dir == "lower"
dir_hint = {"higher": " ↑ higher = better",
            "lower":  " ↓ lower = better",
            "neither": ""}[metric_dir]
# Kept for the legacy "IoU at each quantile" bar chart on Tab 1.
iou_q_label = primary_metric if primary_metric in IOU_COLS else "iou_q90"

# Apply filters
mask = pd.Series(True, index=df.index)
for dim, vals in filters.items():
    if vals:
        mask &= df[dim].isin(vals)
mask &= df["method"].isin(selected_methods)
df_f = df.loc[mask].copy()

# Summary strip
n_configs = df_f["param_key"].nunique() if "param_key" in df_f else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Methods", df_f["method"].nunique())
c2.metric("Unique configs", n_configs)
c3.metric("Rows after filter", len(df_f))
c4.metric("Total CSV rows", len(df))

if df_f.empty:
    st.error("No rows match the current filters.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Method Ranking",
    "📊 Metric Distributions",
    "📈 Parameter Effects",
    "📍 Distance & Size",
    "🗂️ Raw Data",
])


# ─────────────────────────────────────────────────────────────
# Tab 1 — Ranking
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Average performance per method, across every metric")
    st.caption(
        "Each cell is the **mean of that metric across all filtered configs** for "
        "the given method. Sorted by the sidebar's primary metric. Distance is "
        "lower-is-better; IoU is higher-is-better; σ / area columns are not "
        "directional — compare them to the induced radius."
    )

    # Mean of every numeric metric per method
    summary = (
        df_f.groupby("method")[METRIC_OPTIONS].mean().round(4)
        .assign(n_runs=df_f.groupby("method").size())
    )
    summary = summary.sort_values(
        primary_metric, ascending=LOWER_IS_BETTER
    )
    st.dataframe(summary, width='stretch')

    # Bar chart — mean of the primary metric per method
    bar_df = (
        df_f.groupby("method")[primary_metric].agg(["mean", "std"]).reset_index()
        .sort_values("mean", ascending=LOWER_IS_BETTER)
    )
    fig_primary = px.bar(
        bar_df, x="method", y="mean", error_y="std",
        title=f"Mean **{primary_metric}** per method{dir_hint}  ·  error bars = ±1 std",
        labels={"mean": f"mean {primary_metric}"},
    )
    fig_primary.update_layout(xaxis_tickangle=-30, height=480)
    st.plotly_chart(fig_primary, width='stretch')

    # Grouped bar — mean IoU at each quantile (always useful regardless of primary)
    mean_long = (
        df_f.groupby("method")[IOU_COLS].mean().reset_index()
        .melt(id_vars="method", var_name="quantile", value_name="mean_iou")
    )
    mean_long["quantile"] = mean_long["quantile"].str.replace("iou_", "")
    order_iou = (
        df_f.groupby("method")["iou_q90"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    fig_bar = px.bar(
        mean_long, x="method", y="mean_iou", color="quantile", barmode="group",
        category_orders={"method": order_iou},
        title="Mean IoU per method, broken out by quantile threshold",
        labels={"mean_iou": "Mean IoU"},
    )
    fig_bar.update_layout(xaxis_tickangle=-30, height=480)
    st.plotly_chart(fig_bar, width='stretch')

    # Mean distance — always shown (lower is better)
    dist_df = (
        df_f.groupby("method")["distance"].mean().reset_index()
        .sort_values("distance")
    )
    fig_d = px.bar(
        dist_df, x="method", y="distance",
        title="Mean centroid distance to induced center ↓ lower = better",
    )
    fig_d.update_layout(xaxis_tickangle=-30, height=420)
    st.plotly_chart(fig_d, width='stretch')


# ─────────────────────────────────────────────────────────────
# Tab 2 — IoU distributions
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Distribution of **{primary_metric}** per method{dir_hint}")
    st.caption(
        "Box: median (line), IQR (box), 1.5×IQR whiskers. "
        "Violin: full density. Long tails in the better direction mean "
        "the method *can* nail it on favourable configs."
    )

    order = (
        df_f.groupby("method")[primary_metric].median()
        .sort_values(ascending=LOWER_IS_BETTER).index.tolist()
    )

    fig_box = px.box(
        df_f, x="method", y=primary_metric, color="method",
        category_orders={"method": order},
        points="outliers",
        title=f"Per-method distribution — {primary_metric}",
    )
    fig_box.update_layout(showlegend=False, height=520, xaxis_tickangle=-30)
    st.plotly_chart(fig_box, width='stretch')

    fig_violin = px.violin(
        df_f, x="method", y=primary_metric, color="method",
        category_orders={"method": order}, box=True, points=False,
        title=f"Density of {primary_metric} per method",
    )
    fig_violin.update_layout(showlegend=False, height=520, xaxis_tickangle=-30)
    st.plotly_chart(fig_violin, width='stretch')

    # CDF — fraction of configs achieving metric ≥/≤ threshold
    if HIGHER_IS_BETTER or LOWER_IS_BETTER:
        st.subheader("Survival curve — fraction of configs reaching a threshold")
        comp = "≥" if HIGHER_IS_BETTER else "≤"
        st.caption(
            f"For each method: **fraction of configs with {primary_metric} {comp} x**. "
            "Higher curves (or further along the better direction) = better."
        )
        vals_all = df_f[primary_metric].dropna()
        if len(vals_all):
            lo, hi = float(vals_all.min()), float(vals_all.max())
            thresholds = np.linspace(lo, hi, 51)
            cdf_rows = []
            for m, sub in df_f.groupby("method"):
                vals = sub[primary_metric].dropna().values
                if len(vals) == 0:
                    continue
                for t in thresholds:
                    frac = float((vals >= t).mean()) if HIGHER_IS_BETTER \
                        else float((vals <= t).mean())
                    cdf_rows.append({
                        "method": m, "threshold": t, "frac": frac,
                    })
            fig_cdf = px.line(
                pd.DataFrame(cdf_rows), x="threshold", y="frac", color="method",
                title=f"Fraction of runs with {primary_metric} {comp} threshold",
                labels={"frac": "Fraction of configs"},
            )
            fig_cdf.update_layout(height=480, yaxis_tickformat=".0%")
            st.plotly_chart(fig_cdf, width='stretch')


# ─────────────────────────────────────────────────────────────
# Tab 3 — Parameter effects (avg metric vs each sweep dim)
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"Effect of each sweep parameter on **{primary_metric}**{dir_hint}")
    st.caption(
        "For every sweep parameter: x = the parameter value, "
        f"y = **mean of {primary_metric}** across all filtered configs that share "
        "that value, one line per method. "
        "Shaded band = ±1 std across the configs that contributed to the mean. "
        "If a method's line trends with the parameter, the parameter affects it; "
        "if it stays flat, the method is insensitive to that dim."
    )

    # Order legend / colour by overall ranking on the primary metric so the
    # best method is the same colour across every plot.
    method_order = (
        df_f.groupby("method")[primary_metric].mean()
        .sort_values(ascending=LOWER_IS_BETTER).index.tolist()
    )

    for dim in SWEEP_DIMS:
        sub = (
            df_f.groupby(["method", dim])[primary_metric]
            .agg(["mean", "std", "count"]).reset_index()
        )
        if sub.empty:
            continue
        sub["std"] = sub["std"].fillna(0.0)
        fig = px.line(
            sub, x=dim, y="mean", color="method", error_y="std",
            markers=True, category_orders={"method": method_order},
            title=f"Mean {primary_metric} vs {dim}",
            labels={"mean": f"mean {primary_metric}"},
        )
        fig.update_layout(height=460)
        st.plotly_chart(fig, width='stretch')

    # Spatial 2D heatmap — weakspot location effect, per method
    st.subheader(f"2D location effect — mean {primary_metric} by (center_x, center_y)")
    st.caption(
        "Per-method 3×3 grid of induced weakspot location. "
        "Tells you whether a method is biased toward central or corner placements."
    )
    loc_piv = (
        df_f.groupby(["method", "center_x", "center_y"])[primary_metric]
        .mean().reset_index()
    )
    if not loc_piv.empty:
        fig_loc = px.density_heatmap(
            loc_piv, x="center_x", y="center_y", z=primary_metric,
            facet_col="method", facet_col_wrap=4, histfunc="avg",
            nbinsx=3, nbinsy=3,
            color_continuous_scale="Viridis" if HIGHER_IS_BETTER else "Viridis_r",
            title=f"Mean {primary_metric} by induced weakspot location",
        )
        fig_loc.update_layout(height=620)
        st.plotly_chart(fig_loc, width='stretch')


# ─────────────────────────────────────────────────────────────
# Tab 4 — Distance & size
# ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Centroid distance — how close is the extracted weakspot?")
    st.caption(
        "Euclidean distance between the extracted centroid (at q=0.85) and the "
        "induced center. Lower = better. Many high-IoU methods also have low distance, "
        "but a method can be close in centroid yet still miss the region (wrong size)."
    )

    order_d = (
        df_f.groupby("method")["distance"].median()
        .sort_values().index.tolist()
    )
    fig_d = px.box(
        df_f, x="method", y="distance", color="method",
        category_orders={"method": order_d}, points="outliers",
        title="Centroid distance to induced center — per method",
    )
    fig_d.update_layout(showlegend=False, height=480, xaxis_tickangle=-30)
    st.plotly_chart(fig_d, width='stretch')

    st.subheader("Extracted σ — does method size match induced radius?")
    st.caption(
        "Scatter of extracted σ₁/σ₂ vs the induced radius. "
        "An ideal method tracks the diagonal: bigger induced gap → bigger σ. "
        "Flat / inverted clouds indicate the method ignores induced size."
    )
    df_size = df_f.dropna(subset=["sigma_x1", "sigma_x2", "radius"]).copy()
    df_size["sigma_mean"] = (df_size["sigma_x1"] + df_size["sigma_x2"]) / 2.0
    if not df_size.empty:
        fig_sz = px.scatter(
            df_size, x="radius", y="sigma_mean", color="method",
            facet_col="method", facet_col_wrap=4, opacity=0.55,
            title="Extracted σ (mean of σ₁, σ₂) vs induced radius",
        )
        fig_sz.update_layout(height=620, showlegend=False)
        st.plotly_chart(fig_sz, width='stretch')

    st.subheader(f"{primary_metric} vs distance — trade-off scatter")
    st.caption(
        "Each point is one (config × method) run. "
        + ("Top-left corner = ideal regime (low distance, high metric)."
           if HIGHER_IS_BETTER else
           "Bottom-left corner = ideal regime (low distance, low metric)."
           if LOWER_IS_BETTER else
           "Compare clusters per method.")
    )
    fig_td = px.scatter(
        df_f, x="distance", y=primary_metric, color="method",
        opacity=0.55,
        title=f"{primary_metric} vs centroid distance — coloured by method",
    )
    fig_td.update_layout(height=520)
    st.plotly_chart(fig_td, width='stretch')


# ─────────────────────────────────────────────────────────────
# Tab 5 — Raw
# ─────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Filtered raw rows")
    st.caption(
        f"{len(df_f)} rows after sidebar filters. "
        "Sort by any column; download via the toolbar in the top-right of the table."
    )
    show_cols = (
        ["method"] + SWEEP_DIMS +
        ["distance", "sigma_x1", "sigma_x2", "area_frac"] + IOU_COLS +
        ["model_rmse", "model_mae", "model_r2"]
    )
    show_cols = [c for c in show_cols if c in df_f.columns]
    st.dataframe(df_f[show_cols], width='stretch', hide_index=True)

    st.download_button(
        "⬇ Download filtered CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="sweep_results_filtered.csv",
        mime="text/csv",
    )
