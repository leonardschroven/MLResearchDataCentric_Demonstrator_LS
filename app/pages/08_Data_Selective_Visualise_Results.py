"""
Data-Selective Training — Visualise Results.

Aggregated view of the data-selective sweep CSV. Each row is one
(config × detector) result: the detector drives a weakspot-guided data
selection, a model is retrained on those new points only, and its outcome is
compared against the model before selection and against a random baseline
(same #points, no focus).

Tells you which detector, driving selection, most improves the model — and how
that depends on the selection σ, the amount of data added, the model state, and
the dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Data-Selective — Visualise Results", layout="wide")
st.title("📊 Data-Selective Training — Visualise Results")
st.markdown(
    "Aggregated comparison across the sweep. **Guided** = model retrained on new "
    "points selected around the detected weakspot; **Baseline** = same number of "
    "uniformly random new points; **Initial** = model before any selection."
)

RESULTS_DIR = Path("data/experiment_results/data_selective_training")
# One results file per sweep config: sweep__<config>.csv (newest first). The legacy
# single-file name is included if it still exists.
_csv_files = sorted(RESULTS_DIR.glob("sweep__*.csv"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
_legacy = RESULTS_DIR / "data_selective_training_sweep.csv"
if _legacy.exists():
    _csv_files.append(_legacy)

if not _csv_files:
    st.warning(
        "No sweep results found in `data/experiment_results/data_selective_training/` (expected "
        "`sweep__<config>.csv`). Open **Data-Selective Training**, choose a sweep "
        "configuration, and run the parameter sweep first."
    )
    st.stop()

_pick = st.sidebar.selectbox(
    "Results file (per sweep config)", [p.name for p in _csv_files], index=0,
    help="One file per sweep configuration (`sweep__<config>.csv`). Pick which "
         "sweep's results to explore — e.g. the broad sweep or a focused isolate.")
CSV_PATH = RESULTS_DIR / _pick


@st.cache_data(show_spinner=False)
def load_results(path: Path, mtime: float) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        # tolerate a partially-written or ragged line (e.g. a sweep in progress)
        return pd.read_csv(path, engine="python", on_bad_lines="skip",
                           low_memory=False)

df = load_results(CSV_PATH, CSV_PATH.stat().st_mtime)

# Rows swept before the selection-method axis existed all used the legacy strategy.
if "sel_method" not in df.columns:
    df["sel_method"] = "Weakpoint distance"
else:
    df["sel_method"] = df["sel_method"].fillna("Weakpoint distance")

SWEEP_DIMS_ALL = [
    "n_bumps", "noise_std", "n_pool_total", "shift_strength", "radius",
    "iters_initial", "n_train", "sel_sigma", "n_select", "n_candidate",
    "iters_retrain",
]
SWEEP_DIMS = [d for d in SWEEP_DIMS_ALL if d in df.columns]

METRIC_DIRECTION = {
    "gap_mae_guided_minus_base": "lower",   # guided − baseline MAE (negative = guided wins)
    "d_mae_guided":              "lower",   # guided MAE − initial MAE
    "d_errin_guided":            "lower",   # guided in-weakspot err − initial
    "guided_mae":                "lower",
    "guided_err_in":             "lower",
    "guided_r2":                 "higher",
    "iou_q90":                   "higher",  # detection quality
    "distance":                  "lower",   # detection quality
}
METRIC_OPTIONS = [m for m in METRIC_DIRECTION if m in df.columns]

# ─────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Filters")
st.sidebar.caption("Restrict the rows used in every chart. Empty = keep all.")

filters = {}
for dim in SWEEP_DIMS:
    vals = sorted(df[dim].dropna().unique().tolist())
    filters[dim] = st.sidebar.multiselect(dim, vals, default=vals)

methods_all = sorted(df["method"].dropna().unique().tolist())
selected_methods = st.sidebar.multiselect("detector (method)", methods_all,
                                           default=methods_all)

sel_methods_all = sorted(df["sel_method"].dropna().unique().tolist())
selected_sel_methods = st.sidebar.multiselect(
    "selection strategy (sel_method)", sel_methods_all, default=sel_methods_all,
    help="Data-selection strategy. Rows swept before this axis existed are shown "
         "as 'Weakpoint distance' (the original strategy).")

st.sidebar.markdown("---")
primary_metric = st.sidebar.selectbox(
    "Primary metric",
    METRIC_OPTIONS,
    index=METRIC_OPTIONS.index("gap_mae_guided_minus_base")
    if "gap_mae_guided_minus_base" in METRIC_OPTIONS else 0,
)
metric_dir = METRIC_DIRECTION[primary_metric]
HIGHER_IS_BETTER = metric_dir == "higher"
LOWER_IS_BETTER = metric_dir == "lower"
dir_hint = {"higher": " ↑ higher = better", "lower": " ↓ lower = better"}[metric_dir]

mask = pd.Series(True, index=df.index)
for dim, vals in filters.items():
    if vals:
        mask &= df[dim].isin(vals)
mask &= df["method"].isin(selected_methods)
mask &= df["sel_method"].isin(selected_sel_methods)
df_f = df.loc[mask].copy()

n_configs = df_f["param_key"].nunique() if "param_key" in df_f else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Detectors", df_f["method"].nunique())
c2.metric("Unique configs", n_configs)
c3.metric("Rows after filter", len(df_f))
c4.metric("Total CSV rows", len(df))

if df_f.empty:
    st.error("No rows match the current filters.")
    st.stop()

# ── Best configuration by the sidebar's primary metric ──
_bm = df_f[primary_metric].dropna()
if not _bm.empty:
    _bidx = _bm.idxmin() if LOWER_IS_BETTER else _bm.idxmax()
    br = df_f.loc[_bidx]
    _varying = [d for d in SWEEP_DIMS if d in df_f.columns and df_f[d].nunique() > 1]
    _dims = ", ".join(f"`{d}`={br[d]}" for d in (_varying or SWEEP_DIMS))
    _extra = ""
    if {"guided_mae", "base_mae"}.issubset(df_f.columns):
        _extra = (f"  ·  guided MAE **{br['guided_mae']:.4f}** vs baseline "
                  f"**{br['base_mae']:.4f}** (Δ {br['guided_mae'] - br['base_mae']:+.4f})")
    st.success(
        f"🥇 **Best configuration by `{primary_metric}`{dir_hint}:** detector "
        f"**{br['method']}** with {_dims}  →  **{primary_metric} = "
        f"{br[primary_metric]:.4f}**{_extra}"
    )
    st.caption(
        "The single best (configuration × detector) among the currently filtered "
        "rows, ranked by the sidebar's **primary metric**. Change that metric — or "
        "filter to one detector — in the sidebar to ask a different question."
    )

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Detector Ranking",
    "⚔️ Guided vs Baseline",
    "📈 Parameter Effects",
    "🎯 Detection Quality",
    "🔎 Method Explorer",
    "🗂️ Raw Data",
])

# ─────────────────────────────────────────────────────────────
# Tab 1 — Detector ranking
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Which detector, driving selection, most improves the model?")
    st.caption(
        "Mean over all filtered configs. `d_mae_guided` < 0 means guided selection "
        "reduced MAE vs the initial model; `gap_mae_guided_minus_base` < 0 means it "
        "beat the random baseline. `d_errin_guided` is the change in in-weakspot error."
    )
    rank_cols = [c for c in [
        "gap_mae_guided_minus_base", "d_mae_guided", "d_errin_guided",
        "guided_mae", "guided_err_in", "iou_q90", "distance",
    ] if c in df_f.columns]
    summary = (
        df_f.groupby("method")[rank_cols].mean().round(4)
        .assign(n_runs=df_f.groupby("method").size())
        .sort_values(primary_metric, ascending=LOWER_IS_BETTER)
    )
    st.dataframe(summary, width='stretch')

    bar_df = (
        df_f.groupby("method")[primary_metric].agg(["mean", "std"]).reset_index()
        .sort_values("mean", ascending=LOWER_IS_BETTER)
    )
    fig = px.bar(bar_df, x="method", y="mean", error_y="std",
                 title=f"Mean **{primary_metric}** per detector{dir_hint} · ±1 std",
                 labels={"mean": f"mean {primary_metric}"})
    fig.update_layout(xaxis_tickangle=-30, height=480)
    st.plotly_chart(fig, width='stretch')

    # Initial vs guided vs baseline mean MAE per detector
    mae_cols = [c for c in ["init_mae", "guided_mae", "base_mae"] if c in df_f.columns]
    if mae_cols:
        long = (df_f.groupby("method")[mae_cols].mean().reset_index()
                .melt(id_vars="method", var_name="model", value_name="mae"))
        order = (df_f.groupby("method")["guided_mae"].mean()
                 .sort_values().index.tolist()) if "guided_mae" in df_f else None
        fig2 = px.bar(long, x="method", y="mae", color="model", barmode="group",
                      category_orders={"method": order} if order else None,
                      title="Mean evaluation MAE — initial vs guided vs baseline")
        fig2.update_layout(xaxis_tickangle=-30, height=480)
        st.plotly_chart(fig2, width='stretch')

# ─────────────────────────────────────────────────────────────
# Tab 2 — Guided vs baseline head-to-head
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Does weakspot-guided selection beat the random baseline?")

    # ── Per-method: which detectors beat the baseline, and by how much ──
    if {"guided_mae", "base_mae"}.issubset(df_f.columns):
        agg = df_f.groupby("method").agg(
            guided_mae=("guided_mae", "mean"),
            base_mae=("base_mae", "mean"),
        )
        if "init_mae" in df_f.columns:
            agg["init_mae"] = df_f.groupby("method")["init_mae"].mean()
        agg["diff_vs_base"] = agg["guided_mae"] - agg["base_mae"]
        agg["win_rate"] = (df_f.assign(w=df_f["guided_mae"] < df_f["base_mae"])
                           .groupby("method")["w"].mean())
        agg = agg.reset_index()
        agg["result"] = np.where(agg["diff_vs_base"] < 0,
                                 "guided better", "baseline better")
        agg = agg.sort_values("diff_vs_base")

        st.caption(
            "`diff_vs_base` = mean guided MAE − mean baseline MAE. **Negative (green) "
            "= guided beat the baseline on average; positive (red) = worse.** "
            "`win_rate` = share of configs where guided MAE < baseline MAE."
        )
        figdiff = px.bar(
            agg, x="diff_vs_base", y="method", orientation="h", color="result",
            color_discrete_map={"guided better": "#2ca02c", "baseline better": "#d62728"},
            title="Mean guided − baseline MAE per detector (negative = guided better)",
            labels={"diff_vs_base": "guided − baseline MAE", "method": ""},
        )
        figdiff.add_vline(x=0, line_dash="dash", line_color="black")
        figdiff.update_layout(height=520, legend_title_text="",
                              yaxis={"categoryorder": "array",
                                     "categoryarray": agg["method"].tolist()[::-1]})
        st.plotly_chart(figdiff, width='stretch')

        show_cols = [c for c in ["method", "init_mae", "guided_mae", "base_mae",
                                 "diff_vs_base", "win_rate", "result"] if c in agg.columns]
        table = agg[show_cols].copy()
        for c in ("init_mae", "guided_mae", "base_mae", "diff_vs_base"):
            if c in table:
                table[c] = table[c].round(4)
        table["win_rate"] = (table["win_rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(table, width='stretch', hide_index=True)
        st.markdown("---")

    if "gap_mae_guided_minus_base" in df_f.columns:
        st.caption(
            "`gap = guided_mae − base_mae`. Negative = guided wins. Bars show the "
            "**fraction of configs** where guided beats the baseline, per detector."
        )
        win = (df_f.assign(win=df_f["gap_mae_guided_minus_base"] < 0)
               .groupby("method")["win"].mean().reset_index()
               .sort_values("win", ascending=False))
        figw = px.bar(win, x="method", y="win",
                      title="Fraction of configs where guided MAE < baseline MAE",
                      labels={"win": "win rate"})
        figw.update_layout(xaxis_tickangle=-30, height=440, yaxis_tickformat=".0%")
        st.plotly_chart(figw, width='stretch')

        figb = px.box(df_f, x="method", y="gap_mae_guided_minus_base", color="method",
                      points="outliers",
                      title="Distribution of (guided − baseline) MAE per detector")
        figb.add_hline(y=0, line_dash="dash", line_color="black")
        figb.update_layout(showlegend=False, height=500, xaxis_tickangle=-30)
        st.plotly_chart(figb, width='stretch')

    if {"guided_mae", "base_mae"}.issubset(df_f.columns):
        st.caption("Each point = one (config × detector). Below the diagonal = "
                   "guided better than baseline.")
        figs = px.scatter(df_f, x="base_mae", y="guided_mae", color="method",
                          opacity=0.5, title="Guided MAE vs baseline MAE")
        lo = float(min(df_f["base_mae"].min(), df_f["guided_mae"].min()))
        hi = float(max(df_f["base_mae"].max(), df_f["guided_mae"].max()))
        figs.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                       line=dict(dash="dash", color="black"))
        figs.update_layout(height=520)
        st.plotly_chart(figs, width='stretch')

# ─────────────────────────────────────────────────────────────
# Tab 3 — Parameter effects
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"Effect of each swept parameter on **{primary_metric}**{dir_hint}")

    # ── Best value per parameter (marginal) ──
    best_rows = []
    for dim in SWEEP_DIMS:
        if df_f[dim].nunique() < 2:
            continue
        means = df_f.groupby(dim)[primary_metric].mean()
        best_val = means.idxmin() if LOWER_IS_BETTER else means.idxmax()
        best_rows.append({
            "parameter": dim,
            "best value": best_val,
            f"mean {primary_metric}": round(float(means.loc[best_val]), 4),
        })
    if best_rows:
        st.markdown(f"**Best value for each swept parameter** (by mean "
                    f"`{primary_metric}`{dir_hint}):")
        st.dataframe(pd.DataFrame(best_rows), width='stretch', hide_index=True)
        st.caption(
            "Marginal best: for each parameter, the single value with the best mean "
            "metric, averaged over all other settings. Read together with the "
            "single best configuration shown at the top of the page."
        )
        st.markdown("---")

    st.caption(
        "x = parameter value, y = mean of the primary metric across configs sharing "
        "that value, one line per detector. Flat line = insensitive to that axis."
    )
    method_order = (df_f.groupby("method")[primary_metric].mean()
                    .sort_values(ascending=LOWER_IS_BETTER).index.tolist())
    for dim in SWEEP_DIMS:
        if df_f[dim].nunique() < 2:
            continue
        sub = (df_f.groupby(["method", dim])[primary_metric]
               .agg(["mean", "std"]).reset_index())
        sub["std"] = sub["std"].fillna(0.0)
        fig = px.line(sub, x=dim, y="mean", color="method", error_y="std",
                      markers=True, category_orders={"method": method_order},
                      title=f"Mean {primary_metric} vs {dim}",
                      labels={"mean": f"mean {primary_metric}"})
        fig.update_layout(height=440)
        st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────
# Tab 4 — Detection quality vs outcome
# ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Weakspot-identification quality per detector")
    st.caption("Detection metrics (before retraining): centroid distance to the "
               "induced centre (lower better) and ellipse-IoU@0.90 (higher better).")
    q1, q2 = st.columns(2)
    if "distance" in df_f.columns:
        order_d = df_f.groupby("method")["distance"].median().sort_values().index.tolist()
        figd = px.box(df_f, x="method", y="distance", color="method",
                      category_orders={"method": order_d}, points=False,
                      title="Centroid distance per detector")
        figd.update_layout(showlegend=False, height=460, xaxis_tickangle=-30)
        q1.plotly_chart(figd, width='stretch')
    if "iou_q90" in df_f.columns:
        order_i = (df_f.groupby("method")["iou_q90"].median()
                   .sort_values(ascending=False).index.tolist())
        figi = px.box(df_f, x="method", y="iou_q90", color="method",
                      category_orders={"method": order_i}, points=False,
                      title="IoU@0.90 per detector")
        figi.update_layout(showlegend=False, height=460, xaxis_tickangle=-30)
        q2.plotly_chart(figi, width='stretch')

    if {"iou_q90", "d_errin_guided"}.issubset(df_f.columns):
        st.subheader("Does better detection give a better guided outcome?")
        st.caption("Each point = one (config × detector). Ideal: high IoU (right) and "
                   "large in-weakspot error reduction (bottom, negative d_errin_guided).")
        figc = px.scatter(df_f, x="iou_q90", y="d_errin_guided", color="method",
                          opacity=0.5,
                          title="In-weakspot error change vs detection IoU")
        figc.add_hline(y=0, line_dash="dash", line_color="black")
        figc.update_layout(height=520)
        st.plotly_chart(figc, width='stretch')

# ─────────────────────────────────────────────────────────────
# Tab 5 — Method explorer: MAE vs a chosen parameter for one detector
# ─────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Method explorer — MAE vs a chosen parameter")
    st.caption(
        "Pick one detector and one swept parameter to see how MAE varies with that "
        "parameter for the selected method. Guided, baseline and initial MAE are "
        "shown together. The chart honours the sidebar filters, so you can deselect "
        "any parameter values there to restrict which configurations are included."
    )
    ex_dims = [d for d in SWEEP_DIMS if df_f[d].nunique() > 1]
    if not ex_dims:
        st.info("No swept parameter has more than one value under the current filters.")
    else:
        e1, e2, e3 = st.columns([1.4, 1, 1.2])
        ex_method = e1.selectbox("Detector", sorted(df_f["method"].unique()))
        ex_param = e2.selectbox("Parameter (x-axis)", ex_dims)
        ex_kind = e3.radio("Plot style", ["Mean ± std (line)", "Distribution (box)"])
        mae_opts = [c for c in ["guided_mae", "base_mae", "init_mae"] if c in df_f.columns]
        which = st.multiselect("MAE series to show", mae_opts,
                               default=[c for c in ["guided_mae", "base_mae"] if c in mae_opts])

        sub = df_f[df_f["method"] == ex_method]
        if sub.empty or not which:
            st.info("Nothing to plot — pick a detector and at least one MAE series.")
        else:
            if ex_kind.startswith("Mean"):
                frames = []
                for col in which:
                    g = sub.groupby(ex_param)[col].agg(["mean", "std"]).reset_index()
                    g["series"] = col
                    frames.append(g)
                long = pd.concat(frames, ignore_index=True)
                long["std"] = long["std"].fillna(0.0)
                fig = px.line(long, x=ex_param, y="mean", color="series",
                              error_y="std", markers=True,
                              title=f"MAE vs {ex_param} — {ex_method}",
                              labels={"mean": "MAE"})
            else:
                long = sub.melt(id_vars=[ex_param], value_vars=which,
                                var_name="series", value_name="mae")
                fig = px.box(long, x=ex_param, y="mae", color="series",
                             title=f"MAE distribution vs {ex_param} — {ex_method}")
            fig.update_layout(height=520)
            st.plotly_chart(fig, width='stretch')

            summary = (sub.groupby(ex_param)[which].mean().round(4)
                       .assign(configs=sub.groupby(ex_param).size()))
            st.caption(f"Mean MAE for **{ex_method}** at each value of `{ex_param}` "
                       f"(over the filtered configurations).")
            st.dataframe(summary, width='stretch')

# ─────────────────────────────────────────────────────────────
# Tab 6 — Raw
# ─────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Filtered raw rows")
    st.caption(f"{len(df_f)} rows after filters. Sort any column; download below.")
    st.dataframe(df_f, width='stretch', hide_index=True)
    st.download_button(
        "⬇ Download filtered CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="data_selective_sweep_filtered.csv", mime="text/csv",
    )
