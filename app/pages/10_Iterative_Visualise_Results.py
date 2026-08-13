"""Iterative Data-Selective Training — Visualise Results.

Aggregated view of the iterative sweep CSVs. Every row is one
(configuration × detector × **iteration**), so unlike the single-round
visualisation this page works with *trajectories*: the questions it answers are
about shape over the loop — when guidance leads, whether the lead survives, how
fast the model forgets what it already knew, and which of the axes the loop adds
(architecture, learning-rate schedule, coverage schedules, staging) actually
move those curves.

Read alongside the companion paper's single-round findings: the rehearsal mix
and kernel width dominated there, the detector was second-order, and the
whole-area advantage was conditional. This page is where those claims are
re-tested under repetition.
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

from scripts.iterative.sweep import PARAM_FIELDS, RESULTS_DIR

st.set_page_config(page_title="Iterative — Visualise Results", layout="wide")
st.title("📊 Iterative Data-Selective Training — Visualise Results")
st.markdown(
    "Trajectory-level comparison across the sweep. **Guided** tracks select around "
    "the model's current weakspot; **random** tracks draw uniformly from the same "
    "candidate pool; both advance under identical conditions from the same initial "
    "model."
)

# ─────────────────────────────────────────────────────────────
# FILE PICKER + SAFE LOADING (same contract as the single-round page)
# ─────────────────────────────────────────────────────────────
csv_files = sorted(RESULTS_DIR.glob("sweep__*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
if not csv_files:
    st.warning(
        f"No results found in `{RESULTS_DIR}` (expected `sweep__<config>.csv`). Open "
        "**Iterative Data-Selective Training**, choose a sweep configuration in the "
        "🧪 Parameter Sweep tab, and run it first.")
    st.stop()

_PLACEHOLDER = "— choose a results file —"
pick = st.sidebar.selectbox(
    "Results file (per sweep config)", [_PLACEHOLDER] + [p.name for p in csv_files],
    index=0,
    help="One file per sweep configuration. Nothing is read until a file is picked, "
         "so opening this page never touches a large results CSV.")
if pick == _PLACEHOLDER:
    st.info("⬅ **Pick a results file in the sidebar** to load and explore it.")
    st.stop()
CSV_PATH = RESULTS_DIR / pick

SAFE_TRIGGER_MB = 250
size_mb = CSV_PATH.stat().st_size / 1e6
st.sidebar.markdown("---")
safe_mode = st.sidebar.checkbox(
    "⚡ Safe mode (subsample huge files)", value=True,
    help=f"This file is {size_mb:,.0f} MB. Safe mode reads it in chunks and keeps a "
         "uniform random sample of whole **configurations** (never a partial "
         "trajectory), so learning curves stay complete and every group mean remains "
         "an unbiased estimate of the full sweep.")
row_budget = st.sidebar.select_slider(
    "Row budget (safe mode)",
    options=[50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000],
    value=250_000, disabled=not safe_mode)


def _estimate_rows(path: Path, size_bytes: int) -> int:
    with open(path, "rb") as f:
        f.readline()
        sample = f.read(5_000_000)
    nl = sample.count(b"\n")
    return int(size_bytes / (len(sample) / nl)) if nl else size_bytes // 300


@st.cache_data(show_spinner="Loading results…")
def load_results(path: Path, mtime: float, safe: bool,
                 budget: int) -> tuple[pd.DataFrame, int, int]:
    """Load the results, subsampling by *configuration* when the file is large.

    Sampling whole ``param_key`` groups rather than individual rows is what keeps
    every trajectory intact — a randomly thinned set of rows would leave curves
    with holes and bias any per-iteration mean towards whichever iterations
    happened to survive.
    """
    size = path.stat().st_size

    def _read(**kw):
        try:
            return pd.read_csv(path, low_memory=False, **kw)
        except Exception:
            return pd.read_csv(path, engine="python", on_bad_lines="skip",
                               low_memory=False, **kw)

    if not safe or size < SAFE_TRIGGER_MB * 1e6:
        d = _read()
        return d, len(d), len(d)

    est = _estimate_rows(path, size)
    if budget >= est:
        d = _read()
        return d, len(d), len(d)

    keys = _read(usecols=["param_key"])["param_key"].dropna().unique()
    frac = min(1.0, budget / max(est, 1))
    keep = set(pd.Series(keys).sample(frac=frac, random_state=0))
    frames = [ch[ch["param_key"].isin(keep)]
              for ch in pd.read_csv(path, chunksize=250_000, low_memory=False)]
    d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return d, est, len(d)


df, est_rows, loaded_rows = load_results(
    CSV_PATH, CSV_PATH.stat().st_mtime, safe_mode, int(row_budget))
if df.empty:
    st.error("The results file could not be read, or contains no rows.")
    st.stop()
if loaded_rows < est_rows:
    st.sidebar.caption(f"Loaded **{loaded_rows:,}** of ~**{est_rows:,}** rows "
                       f"(whole configurations, {size_mb:,.0f} MB file).")

df = df[df["iteration"] >= 0].copy()          # −1 marks a failed configuration
if df.empty:
    st.error("Every configuration in this file recorded an error.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# TRACKS / METRICS
# ─────────────────────────────────────────────────────────────
REGIME_TRACKS = {
    "accumulative": ("gacc", "racc"),
    "new-only": ("gnew", "rnew"),
    "scaled-accum": ("gsca", "rsca"),
}
METRIC_FAMILY = {
    "MAE (whole area)": ("mae", "MAE", "lower"),
    "Error inside the weakspot": ("err_in", "error in gap", "lower"),
    "Error outside the weakspot": ("err_out", "error outside gap", "lower"),
    "R²": ("r2", "R²", "higher"),
}
available_regimes = [r for r, (g, _) in REGIME_TRACKS.items() if f"{g}_mae" in df.columns
                     and df[f"{g}_mae"].notna().any()]

# Parameters that actually vary in this file — everything else is a constant of
# the study and would only clutter the filters.
param_cols = [c for c in PARAM_FIELDS if c in df.columns]
varying = [c for c in param_cols if df[c].nunique(dropna=False) > 1]

st.sidebar.header("🎛️ Filters")
st.sidebar.caption("Restrict the rows behind every chart. Empty = keep all.")
mask = pd.Series(True, index=df.index)
for c in varying:
    vals = sorted(df[c].dropna().unique().tolist(), key=str)
    chosen = st.sidebar.multiselect(c, vals, default=vals, key=f"f_{c}")
    if chosen:
        mask &= df[c].isin(chosen)
methods = sorted(df["method"].dropna().unique().tolist())
chosen_methods = st.sidebar.multiselect("detector (method)", methods, default=methods)
mask &= df["method"].isin(chosen_methods)
d = df.loc[mask].copy()
if d.empty:
    st.error("No rows match the current filters.")
    st.stop()

st.sidebar.markdown("---")
regime = st.sidebar.selectbox(
    "Regime", available_regimes or ["accumulative"], index=0,
    help="**Accumulative** retrains on the whole growing set; **new-only** on just "
         "each round's points (the companion paper's protocol); **scaled-accum** is "
         "the size-matched control.")
G, Rr = REGIME_TRACKS[regime]
family = st.sidebar.selectbox("Metric family", list(METRIC_FAMILY), index=0)
suffix, ylab, direction = METRIC_FAMILY[family]
LOWER = direction == "lower"
compare_by = st.sidebar.selectbox(
    "Compare by", ["— none —"] + varying, index=0,
    help="Splits every trajectory chart into one line per value of this parameter, "
         "which is how a single-axis sweep is meant to be read.")
compare = None if compare_by == "— none —" else compare_by

c1, c2, c3, c4 = st.columns(4)
c1.metric("Configurations", d["param_key"].nunique())
c2.metric("Detectors", d["method"].nunique())
c3.metric("Iterations", int(d["iteration"].max()))
c4.metric("Rows after filter", f"{len(d):,}")


# ─────────────────────────────────────────────────────────────
# Per-configuration summary — one row per (configuration × detector)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Summarising trajectories…")
def summarise(frame: pd.DataFrame, g: str, r: str, params: tuple) -> pd.DataFrame:
    """Collapse each trajectory to the numbers a sweep is judged on.

    ``mean_gap`` (guided − random averaged over iterations 1…K) is preferred to
    the final-iteration gap whenever the curves cross, and ``best_iter`` exposes
    overshoot: a best iteration well before K means the loop should have stopped.
    """
    f = frame.sort_values("iteration")
    gap = f[f["iteration"] > 0].copy()
    gap["win"] = gap[f"{g}_mae"] < gap[f"{r}_mae"]
    gap["gap"] = gap[f"{g}_mae"] - gap[f"{r}_mae"]
    agg = gap.groupby(["param_key", "method"]).agg(
        mean_gap_mae=("gap", "mean"), win_rate=("win", "mean"),
        n_iter=("iteration", "max")).reset_index()
    last = f.groupby(["param_key", "method"], as_index=False).tail(1)
    keep = ["param_key", "method", f"{g}_mae", f"{r}_mae", f"{g}_err_in",
            f"{r}_err_in", f"{g}_err_out", f"{r}_err_out", "init_mae",
            "forget_guided", "forget_random", "ss_guided_mae", "ss_random_mae",
            "n_model_params"]
    keep = [k for k in keep if k in last.columns] + [p for p in params
                                                     if p in last.columns]
    out = agg.merge(last[keep], on=["param_key", "method"], how="left")
    out["final_gap_mae"] = out[f"{g}_mae"] - out[f"{r}_mae"]
    if "ss_guided_mae" in out.columns:
        out["staging_gain"] = out["ss_guided_mae"] - out[f"{g}_mae"]
    best = (f.loc[f.groupby(["param_key", "method"])[f"{g}_mae"].idxmin(),
                  ["param_key", "method", "iteration"]]
            .rename(columns={"iteration": "best_iter"}))
    return out.merge(best, on=["param_key", "method"], how="left")


summary = summarise(d, G, Rr, tuple(param_cols))

SUMMARY_METRICS = {
    "mean_gap_mae": ("Across-loop mean gap (guided − random)", "lower"),
    "final_gap_mae": ("Final-iteration gap (guided − random)", "lower"),
    "win_rate": ("Share of iterations guided led", "higher"),
    f"{G}_mae": ("Final guided MAE", "lower"),
    "forget_guided": ("Forgetting: Δ error outside the gap", "lower"),
    "best_iter": ("Iteration of the guided minimum", "higher"),
}
SUMMARY_METRICS = {k: v for k, v in SUMMARY_METRICS.items() if k in summary.columns}
if "staging_gain" in summary.columns:
    SUMMARY_METRICS["staging_gain"] = (
        "Staging gain (single-shot − loop; positive = looping won)", "higher")


def _ci_band(frame: pd.DataFrame, col: str, by: str | None):
    """Mean with a 95% CI across configurations, per iteration (and per group)."""
    keys = ["iteration"] + ([by] if by else [])
    g = frame.groupby(keys)[col].agg(["mean", "std", "count"]).reset_index()
    g["hw"] = 1.96 * g["std"].fillna(0.0) / np.sqrt(g["count"].clip(lower=1))
    return g


def _curve_figure(frame, cols_colours, by, title, y_title):
    """Per-iteration mean curves with CI bands. ``cols_colours`` maps a column to
    (label, colour); ``by`` adds one line style per group value."""
    fig = go.Figure()
    dashes = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    groups = sorted(frame[by].dropna().unique(), key=str) if by else [None]
    for gi, gv in enumerate(groups):
        sub = frame if gv is None else frame[frame[by] == gv]
        dash = dashes[gi % len(dashes)]
        for col, (label, colour) in cols_colours.items():
            if col not in sub.columns:
                continue
            b = _ci_band(sub, col, None)
            name = label if gv is None else f"{label} · {by}={gv}"
            fig.add_trace(go.Scatter(
                x=list(b["iteration"]) + list(b["iteration"])[::-1],
                y=list(b["mean"] + b["hw"]) + list(b["mean"] - b["hw"])[::-1],
                fill="toself", fillcolor=colour, opacity=0.12, line=dict(width=0),
                hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=b["iteration"], y=b["mean"], mode="lines+markers", name=name,
                line=dict(color=colour, width=2.4, dash=dash),
                marker=dict(size=5)))
    fig.update_layout(height=480, title=title, template="plotly_white",
                      xaxis_title="iteration", yaxis_title=y_title,
                      legend_title_text="")
    return fig


tabs = st.tabs([
    "📉 Learning Curves", "⚔️ Advantage", "🧯 Forgetting", "📈 Parameter Effects",
    "🧠 Architecture & LR", "🎛️ Schedules", "⏱️ Staging", "🎯 Detection", "🗂️ Raw",
])

# ─────────────────────────────────────────────────────────────
# 📉 LEARNING CURVES
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"{family} per iteration — {regime} regime")
    st.caption(
        "Mean across every filtered configuration, shaded with a 95% confidence "
        "interval. Iteration 0 is the shared initial model, so both strategies start "
        "identically by construction and any separation afterwards is the selection "
        "rule alone. Use **Compare by** in the sidebar to split the curves along the "
        "swept axis.")
    st.plotly_chart(_curve_figure(
        d, {f"{G}_{suffix}": ("guided", "#2ca02c"),
            f"{Rr}_{suffix}": ("random", "#d62728")},
        compare, f"{family} — guided vs random ({regime})", ylab), width='stretch')

    if len(available_regimes) > 1:
        st.markdown("---")
        st.subheader("All regimes at once")
        st.caption(
            "The companion paper retrained on the newly selected points alone and "
            "called that protocol conservative, expecting accumulation to soften the "
            "collapse. Here the protocols run side by side: if **accumulative** stays "
            "flat where **new-only** degrades, that expectation is confirmed, and if "
            "**scaled-accum** tracks accumulative despite matching new-only's point "
            "count, the difference is the training distribution rather than the amount "
            "of data.")
        cols = {}
        shades = {"accumulative": ("#2ca02c", "#d62728"),
                  "new-only": ("#98df8a", "#ff9896"),
                  "scaled-accum": ("#006d2c", "#a50f15")}
        for reg in available_regimes:
            g_, r_ = REGIME_TRACKS[reg]
            cg, cr = shades[reg]
            cols[f"{g_}_{suffix}"] = (f"guided · {reg}", cg)
            cols[f"{r_}_{suffix}"] = (f"random · {reg}", cr)
        st.plotly_chart(_curve_figure(d, cols, None,
                                      f"{family} — every regime", ylab),
                        width='stretch')

# ─────────────────────────────────────────────────────────────
# ⚔️ ADVANTAGE
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Does guidance stay ahead as the loop runs?")
    st.caption(
        "`gap = guided − random` at each iteration; **below zero means guidance is "
        "ahead**. The single-round study could only report one such number. A gap that "
        "opens early and closes later is the random baseline catching up as it "
        "accumulates coverage — the honest reading is then that guidance bought "
        "*earlier* convergence, not a better endpoint.")
    gcol = f"{G}_{suffix}"
    rcol = f"{Rr}_{suffix}"
    dd = d.copy()
    dd["gap"] = dd[gcol] - dd[rcol]
    dd["win"] = dd["gap"] < 0
    st.plotly_chart(_curve_figure(dd, {"gap": ("guided − random", "#1f77b4")},
                                  compare, f"{family}: advantage per iteration",
                                  "guided − random"), width='stretch')

    wr = (dd[dd["iteration"] > 0]
          .groupby(["iteration"] + ([compare] if compare else []))["win"]
          .mean().reset_index())
    figw = px.line(wr, x="iteration", y="win", color=compare, markers=True,
                   title="Share of configurations where guided leads, per iteration")
    figw.add_hline(y=0.5, line_dash="dash", line_color="black")
    figw.update_layout(height=420, yaxis_tickformat=".0%", template="plotly_white")
    st.plotly_chart(figw, width='stretch')
    st.caption(
        "A win rate hovering at 50% is a coin flip, not a method. The companion paper "
        "found guidance beat its matched baseline in only 31.7% of configurations over "
        "its full grid, and the median configuration was marginally worse — this chart "
        "is the iterative restatement of that caution.")

    st.markdown("---")
    st.subheader("Detector ranking across the loop")
    st.caption(
        "Mean across-loop gap per detector. At a sound operating point the paper found "
        "every one of its seventeen detectors beat random, with margins differing "
        "almost tenfold, and concluded the detector is second-order — settle coverage "
        "first, choose the detector last.")
    rank = (summary.groupby("method")
            .agg(mean_gap=("mean_gap_mae", "mean"), win_rate=("win_rate", "mean"),
                 runs=("mean_gap_mae", "size")).reset_index()
            .sort_values("mean_gap"))
    figr = px.bar(rank, x="mean_gap", y="method", orientation="h",
                  color=np.where(rank["mean_gap"] < 0, "guided better", "random better"),
                  color_discrete_map={"guided better": "#2ca02c",
                                      "random better": "#d62728"},
                  title="Mean across-loop gap per detector (negative = guided better)")
    figr.add_vline(x=0, line_dash="dash", line_color="black")
    figr.update_layout(height=max(320, 46 * len(rank)), template="plotly_white",
                       legend_title_text="")
    st.plotly_chart(figr, width='stretch')
    st.dataframe(rank.round(4), width='stretch', hide_index=True)

# ─────────────────────────────────────────────────────────────
# 🧯 FORGETTING
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Catastrophic forgetting, measured directly")
    st.caption(
        "`forget = error outside the induced gap − the same error at iteration 0`. "
        "**Positive means the model got worse where it was already competent.** The "
        "companion paper identified over-concentration as the dominant risk and the "
        "binding constraint on the whole method, but could only infer it from the "
        "whole-area metric after a single round; a loop lets it be watched "
        "accumulating.")
    if {"forget_guided", "forget_random"}.issubset(d.columns) and d["forget_guided"].notna().any():
        st.plotly_chart(_curve_figure(
            d, {"forget_guided": ("guided", "#2ca02c"),
                "forget_random": ("random", "#d62728")},
            compare, "Error outside the gap, relative to iteration 0",
            "Δ error outside the gap"), width='stretch')
        st.markdown("---")
        st.subheader("Repair against damage")
        st.caption(
            "Each point is one configuration at its final iteration: the horizontal "
            "axis is how much the gap was repaired (negative = better), the vertical "
            "how much was lost elsewhere. The bottom-left quadrant is the only place a "
            "guided run is unambiguously worth running.")
        figq = px.scatter(
            summary, x=f"{G}_err_in", y="forget_guided", color=compare or "method",
            hover_data=[c for c in varying if c in summary.columns],
            title="Final in-gap error vs forgetting outside it",
            labels={f"{G}_err_in": "final error inside the gap",
                    "forget_guided": "Δ error outside the gap"})
        figq.add_hline(y=0, line_dash="dash", line_color="black")
        figq.update_layout(height=520, template="plotly_white")
        st.plotly_chart(figq, width='stretch')
    else:
        st.info("This sweep induced no weakspot (radius 0), so in/out-of-gap errors "
                "are not defined. Use the whole-area MAE curves instead.")

# ─────────────────────────────────────────────────────────────
# 📈 PARAMETER EFFECTS
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Which parameters move the result?")
    metric = st.selectbox(
        "Summary metric", list(SUMMARY_METRICS),
        format_func=lambda k: f"{k} — {SUMMARY_METRICS[k][0]}")
    better_low = SUMMARY_METRICS[metric][1] == "lower"
    st.caption(
        "Each trajectory is collapsed to one number, then averaged over every other "
        "setting. This is the sensitivity view the companion paper used to rank its "
        "factors, where the rehearsal mix and kernel width dominated everything else "
        "and the task-describing factors were near-inert.")

    if not varying:
        st.info("Only one configuration is present in this file, so there is no "
                "parameter to compare. Run a sweep with more than one grid value.")
    else:
        rows = []
        for c in varying:
            if c not in summary.columns:
                continue
            m = summary.groupby(c)[metric].mean()
            if len(m) < 2:
                continue
            best = m.idxmin() if better_low else m.idxmax()
            worst = m.idxmax() if better_low else m.idxmin()
            # Levels are stringified because a sweep mixes numeric and categorical
            # axes in one table, and a mixed-type column is not Arrow-serialisable.
            rows.append({"parameter": c, "best value": str(best),
                         f"best mean {metric}": round(float(m.loc[best]), 4),
                         "worst value": str(worst),
                         f"worst mean {metric}": round(float(m.loc[worst]), 4),
                         "span": round(float(m.max() - m.min()), 4)})
        if rows:
            sens = pd.DataFrame(rows).sort_values("span", ascending=False)
            st.dataframe(sens, width='stretch', hide_index=True)
            figs = px.bar(sens, x="span", y="parameter", orientation="h",
                          title=f"Sensitivity: range of mean `{metric}` across each "
                                f"parameter's levels")
            figs.update_layout(height=max(300, 44 * len(sens)), template="plotly_white",
                               yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(figs, width='stretch')

            st.markdown("---")
            for c in sens["parameter"]:
                g = summary.groupby(c)[metric].agg(["mean", "std", "count"]).reset_index()
                g["hw"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
                fig = px.line(g, x=c, y="mean", error_y="hw", markers=True,
                              title=f"Mean {metric} vs {c}")
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(height=380, template="plotly_white")
                st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────────────────────
# 🧠 ARCHITECTURE & LEARNING RATE
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Architecture and learning rate")
    st.caption(
        "Two axes the single-round study had no reason to vary. Under repetition the "
        "architecture sets how much capacity is free to absorb the weak region without "
        "overwriting what is already learnt, and the per-iteration learning rate sets "
        "how far each round may move the weights at all — the optimiser-side "
        "counterpart of the data-side coverage controls.")
    metric2 = st.selectbox("Metric", list(SUMMARY_METRICS), key="arch_metric",
                           format_func=lambda k: f"{k} — {SUMMARY_METRICS[k][0]}")
    if "arch" in summary.columns and summary["arch"].nunique() > 1:
        a = (summary.groupby(["arch", "n_model_params"])[metric2]
             .agg(["mean", "std", "count"]).reset_index()
             .sort_values("n_model_params"))
        a["hw"] = 1.96 * a["std"].fillna(0) / np.sqrt(a["count"].clip(lower=1))
        figa = px.bar(a, x="arch", y="mean", error_y="hw",
                      title=f"{metric2} by architecture (ordered by parameter count)",
                      hover_data=["n_model_params"])
        figa.add_hline(y=0, line_dash="dash", line_color="black")
        figa.update_layout(height=460, xaxis_tickangle=-25, template="plotly_white")
        st.plotly_chart(figa, width='stretch')

        figc = px.scatter(a, x="n_model_params", y="mean", error_y="hw", text="arch",
                          log_x=True, title=f"{metric2} vs model capacity")
        figc.update_traces(textposition="top center")
        figc.add_hline(y=0, line_dash="dash", line_color="black")
        figc.update_layout(height=460, template="plotly_white")
        st.plotly_chart(figc, width='stretch')
    else:
        st.info("Architecture was not swept in this file.")

    lr_axes = [c for c in ("lr_schedule", "lr_init", "lr_gamma", "solver", "alpha",
                           "batch_size", "activation")
               if c in summary.columns and summary[c].nunique() > 1]
    if lr_axes:
        st.markdown("---")
        if {"lr_schedule", "lr_init"}.issubset(lr_axes):
            piv = summary.pivot_table(index="lr_schedule", columns="lr_init",
                                      values=metric2, aggfunc="mean")
            figh = px.imshow(piv, text_auto=".3f", aspect="auto",
                             color_continuous_scale="RdYlGn_r" if
                             SUMMARY_METRICS[metric2][1] == "lower" else "RdYlGn",
                             title=f"{metric2}: learning-rate schedule × base rate")
            figh.update_layout(height=420)
            st.plotly_chart(figh, width='stretch')
        for c in lr_axes:
            g = summary.groupby(c)[metric2].agg(["mean", "std", "count"]).reset_index()
            g["hw"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
            fig = px.bar(g, x=c, y="mean", error_y="hw", title=f"{metric2} by {c}")
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.update_layout(height=380, template="plotly_white", xaxis_tickangle=-20)
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("No optimiser axis was swept in this file.")

# ─────────────────────────────────────────────────────────────
# 🎛️ SCHEDULES
# ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Scheduled coverage: relaxing the focus as the gap heals")
    st.caption(
        "The companion paper's Discussion names this extension outright — *\"iterating "
        "the loop with a schedule that relaxes focus as the weak region shrinks is the "
        "obvious extension, expressible through either coverage control\"*. The "
        "adaptive schedules do exactly that, reading the weakspot's severity from the "
        "model's own error surface, so nothing here needs ground truth.")
    applied = [c for c in ("mix_used", "sigma_used", "severity", "lr_used")
               if c in d.columns and d[c].notna().any()]
    if applied:
        which = st.multiselect("Applied controls to plot", applied,
                               default=[c for c in applied if c != "lr_used"])
        cmap = {"mix_used": ("rehearsal mix α", "#2ca02c"),
                "sigma_used": ("kernel width σ", "#9467bd"),
                "severity": ("weakspot severity", "#8c564b"),
                "lr_used": ("learning rate", "#ff7f0e")}
        if which:
            st.plotly_chart(_curve_figure(
                d[d["iteration"] > 0], {c: cmap[c] for c in which}, compare,
                "What each round actually applied", "value"), width='stretch')
            st.caption(
                "A flat line is a constant schedule; a curve that bends *towards* "
                "coverage as the loop runs is the relaxation working. Read the "
                "severity trace alongside it — an adaptive schedule that never moves "
                "is one whose weakspot never healed.")
    sched_axes = [c for c in ("mix_schedule", "sigma_schedule", "mix_ratio",
                              "sel_sigma", "mix_rate", "sigma_rate")
                  if c in summary.columns and summary[c].nunique() > 1]
    if sched_axes:
        metric3 = st.selectbox("Metric", list(SUMMARY_METRICS), key="sched_metric",
                               format_func=lambda k: f"{k} — {SUMMARY_METRICS[k][0]}")
        if {"mix_schedule", "mix_ratio"}.issubset(sched_axes):
            piv = summary.pivot_table(index="mix_schedule", columns="mix_ratio",
                                      values=metric3, aggfunc="mean")
            fig = px.imshow(piv, text_auto=".3f", aspect="auto",
                            color_continuous_scale="RdYlGn_r" if
                            SUMMARY_METRICS[metric3][1] == "lower" else "RdYlGn",
                            title=f"{metric3}: α schedule × starting α")
            fig.update_layout(height=420)
            st.plotly_chart(fig, width='stretch')
        if {"sigma_schedule", "sel_sigma"}.issubset(sched_axes):
            piv = summary.pivot_table(index="sigma_schedule", columns="sel_sigma",
                                      values=metric3, aggfunc="mean")
            fig = px.imshow(piv, text_auto=".3f", aspect="auto",
                            color_continuous_scale="RdYlGn_r" if
                            SUMMARY_METRICS[metric3][1] == "lower" else "RdYlGn",
                            title=f"{metric3}: σ schedule × starting σ")
            fig.update_layout(height=420)
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("No schedule axis was swept in this file.")

# ─────────────────────────────────────────────────────────────
# ⏱️ STAGING
# ─────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Is a budget better staged or spent at once?")
    st.caption(
        "Every run also trains a **budget- and compute-matched single-shot** model: "
        "the same K × n_select points added in one round with K × the retraining "
        "budget. `staging_gain = single-shot MAE − final loop MAE`, so **positive "
        "means iterating won**. This is the question the single-round study defined "
        "out of existence by design.")
    if "staging_gain" in summary.columns and summary["staging_gain"].notna().any():
        s = summary.dropna(subset=["staging_gain"])
        k1, k2, k3 = st.columns(3)
        k1.metric("Mean staging gain", f"{s['staging_gain'].mean():+.4f}")
        k2.metric("Configurations where looping won",
                  f"{(s['staging_gain'] > 0).mean() * 100:.0f}%")
        k3.metric("Configurations", f"{len(s):,}")
        figs = px.histogram(s, x="staging_gain", color=compare, nbins=40,
                            title="Distribution of the staging gain "
                                  "(positive = looping beat the single shot)")
        figs.add_vline(x=0, line_dash="dash", line_color="black")
        figs.update_layout(height=430, template="plotly_white")
        st.plotly_chart(figs, width='stretch')

        if {"n_iterations", "n_select"}.issubset(summary.columns) and \
                summary["n_iterations"].nunique() > 1:
            summary["total_budget"] = summary["n_iterations"] * summary["n_select"]
            piv = summary.pivot_table(index="n_iterations", columns="n_select",
                                      values=f"{G}_mae", aggfunc="mean")
            figh = px.imshow(piv, text_auto=".3f", aspect="auto",
                             color_continuous_scale="RdYlGn_r",
                             title="Final guided MAE: rounds × points per round "
                                   "(constant-product cells add the same total data)")
            figh.update_layout(height=440)
            st.plotly_chart(figh, width='stretch')
            st.caption(
                "Read along a constant-product diagonal — 1×800, 2×400, 4×200, 8×100, "
                "16×50 all add 800 curated points, so any difference between those "
                "cells is the staging alone.")
    else:
        st.info("The single-shot control was disabled for this sweep, so the staging "
                "comparison is unavailable. Re-run with it enabled in the sidebar.")

# ─────────────────────────────────────────────────────────────
# 🎯 DETECTION
# ─────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Identification quality across the loop")
    st.caption(
        "As the gap heals the detector should legitimately stop finding it — the "
        "weakest region is genuinely elsewhere by then — so a falling IoU late in the "
        "loop is not necessarily failure. What matters is whether it held the target "
        "while the target still existed.")
    det_cols = [c for c in ("det_distance", "det_iou", "det_drift") if c in d.columns]
    if det_cols and d[det_cols].notna().any().any():
        st.plotly_chart(_curve_figure(
            d[d["iteration"] > 0],
            {"det_distance": ("centroid distance ↓", "#d62728"),
             "det_drift": ("drift from previous round", "#7f7f7f")},
            compare, "Where the detector aimed, round by round", "distance"),
            width='stretch')
        st.plotly_chart(_curve_figure(
            d[d["iteration"] > 0], {"det_iou": ("IoU @ q=0.90 ↑", "#1f77b4")},
            compare, "Overlap with the induced gap", "IoU"), width='stretch')

        st.markdown("---")
        st.caption(
            "Does better identification buy a better outcome? The paper's strongest "
            "detector achieved its advantage with only a moderate overlap score, "
            "suggesting that placing the selection mass approximately right matters "
            "more than recovering the region's exact extent.")
        j = d[d["iteration"] > 0].groupby(["param_key", "method"], as_index=False).agg(
            iou=("det_iou", "mean"), dist=("det_distance", "mean"))
        j = j.merge(summary[["param_key", "method", "mean_gap_mae"]],
                    on=["param_key", "method"], how="left")
        figj = px.scatter(j, x="iou", y="mean_gap_mae", color="method", opacity=0.6,
                          title="Across-loop gap vs mean detection IoU")
        figj.add_hline(y=0, line_dash="dash", line_color="black")
        figj.update_layout(height=500, template="plotly_white")
        st.plotly_chart(figj, width='stretch')
    else:
        st.info("No detection metrics in this file (they require an induced gap).")

# ─────────────────────────────────────────────────────────────
# 🗂️ RAW
# ─────────────────────────────────────────────────────────────
with tabs[8]:
    st.subheader("Per-configuration summary")
    st.caption("One row per (configuration × detector) — the collapsed trajectories "
               "every chart above is built from.")
    st.dataframe(summary, width='stretch', hide_index=True)
    st.download_button("⬇ Download summary CSV",
                       data=summary.to_csv(index=False).encode("utf-8"),
                       file_name=f"{CSV_PATH.stem}_summary.csv", mime="text/csv")
    st.markdown("---")
    st.subheader("Filtered raw rows")
    st.caption(f"{len(d):,} rows after filters (one per configuration × detector × "
               f"iteration).")
    st.dataframe(d.head(5000), width='stretch', hide_index=True)
    st.download_button("⬇ Download filtered rows CSV",
                       data=d.to_csv(index=False).encode("utf-8"),
                       file_name=f"{CSV_PATH.stem}_filtered.csv", mime="text/csv")
