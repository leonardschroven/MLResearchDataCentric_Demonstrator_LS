"""
Analyse the parameter-sweep results in ``data/experiment_results/sweep_results.csv``
and produce a self-contained set of PNG figures + a summary CSV that
describe how each of the 17 detection methods performs.

Output (relative to repo root):
    figures/Experiment_WeakspotAnalyzer/
        01_method_ranking.png            mean IoU per method (sorted, by tier)
        02_iou_distribution.png          box+strip of IoU per method
        03_base_methods_iou.png          same, restricted to the 8 base methods
        04_sensitivity_noise.png         mean IoU vs noise_std
        05_sensitivity_complexity.png    mean IoU vs n_bumps
        06_sensitivity_radius.png        mean IoU vs radius
        07_sensitivity_samples.png       mean IoU vs n_samples
        08_sensitivity_location.png      mean IoU per (center_x, center_y) faceted per method
        09_distance_vs_iou.png           centroid distance vs IoU per (method, config)
        10_method_ranking_distance.png   mean centroid distance per method (sorted, by tier)
        11_sensitivity_noise_distance.png        mean distance vs noise_std
        12_sensitivity_complexity_distance.png   mean distance vs n_bumps
        13_sensitivity_radius_distance.png       mean distance vs radius
        14_sensitivity_samples_distance.png      mean distance vs n_samples
        method_summary.csv               per-method mean / median IoU, distance, n_runs

Usage:
    python scripts/AnalyzeExperimentWeakspotMethods.py

Re-run any time the sweep CSV is updated to refresh every figure.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────
# Paths (relative — run from repo root)
# ─────────────────────────────────────────────────────────────
CSV_PATH = Path("data/experiment_results/sweep_results.csv")
OUT_DIR  = Path("figures/Experiment_WeakspotAnalyzer")

PRIMARY = "iou_q90"
WIDTH, HEIGHT, SCALE = 1100, 600, 2   # 2× scale → ~300 DPI in print

# ─────────────────────────────────────────────────────────────
# Method tiers — match scripts/weakspot/detection.py registry keys
# ─────────────────────────────────────────────────────────────
BASE_METHODS = [
    "Gaussian Process Regression",
    "Polynomial Response Surface",
    "RBF Interpolation",
    "kNN Performance Mapping",
    "LOESS Local Regression",
    "Quantile Regression",
    "Peaks over Threshold (EVT)",
    "Bayesian Optimization (EI)",
]
SIMPLE_ENSEMBLES = [
    "kNN + GPR (mean)",
    "kNN + Quantile (mean)",
]
ADVANCED_ENSEMBLES = [
    "Anchored GPR (QR prior)",
    "Gated GPR (QR anchor)",
    "QR × GPR (geometric)",
    "EVT × GPR (geometric)",
    "GPR/kNN (variance-weighted)",
    "GPR/kNN (disagreement-amplified)",
    "Local GPR (QR-localised)",
]
TIER_OF = (
    {m: "Base"     for m in BASE_METHODS}
    | {m: "Simple"   for m in SIMPLE_ENSEMBLES}
    | {m: "Advanced" for m in ADVANCED_ENSEMBLES}
)
TIER_COLORS = {"Base": "#ff7f0e", "Simple": "#888888", "Advanced": "#1f77b4"}


# ─────────────────────────────────────────────────────────────
# Loading + saving helpers
# ─────────────────────────────────────────────────────────────
def _load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"  CSV not found: {CSV_PATH.resolve()}")
        print("  Run the sweep first (Streamlit page or scripts/run_sweep.py).")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    if PRIMARY not in df.columns:
        print(f"  CSV missing column {PRIMARY}.")
        sys.exit(1)
    df = df[df["method"].isin(TIER_OF)].copy()
    df["tier"] = df["method"].map(TIER_OF)
    print(f"  Loaded {len(df):,} rows, "
          f"{df['param_key'].nunique() if 'param_key' in df else '?'} configs, "
          f"{df['method'].nunique()} methods.")
    return df


def _save(fig: go.Figure, name: str, height: int = HEIGHT) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.write_image(str(path), width=WIDTH, height=height, scale=SCALE)
    print(f"  wrote {path}")


def _style(fig: go.Figure, *, title: str, xtitle: str, ytitle: str,
           legend: bool = False) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title=dict(text=xtitle, font=dict(size=15)),
                   tickfont=dict(size=12)),
        yaxis=dict(title=dict(text=ytitle, font=dict(size=15)),
                   tickfont=dict(size=12), gridcolor="lightgray"),
        font=dict(family="Times New Roman, serif"),
        plot_bgcolor="white",
        margin=dict(l=70, r=30, t=70, b=130 if legend else 80),
        showlegend=legend,
    )
    fig.update_xaxes(showgrid=False, linecolor="black", mirror=True, tickangle=-30)
    fig.update_yaxes(linecolor="black", mirror=True)


# ─────────────────────────────────────────────────────────────
# 01 / 10 — method ranking bar (either metric)
# ─────────────────────────────────────────────────────────────
def fig_method_ranking(df: pd.DataFrame, *,
                       metric: str, filename: str,
                       title: str, ytitle: str,
                       lower_is_better: bool = False) -> None:
    agg = (
        df.groupby("method")
        .agg(mean_val=(metric, "mean"),
             std_val=(metric, "std"),
             tier=("tier", "first"))
        .reset_index()
        .sort_values("mean_val", ascending=lower_is_better)
    )
    fig = px.bar(
        agg, x="method", y="mean_val", color="tier",
        error_y="std_val",
        color_discrete_map=TIER_COLORS,
        category_orders={"tier": ["Advanced", "Simple", "Base"]},
    )
    direction = "ascending" if lower_is_better else "descending"
    _style(
        fig,
        title=f"{title}  ·  bars ordered {direction}  ·  error bars = ±1 σ",
        xtitle="Method",
        ytitle=ytitle,
        legend=True,
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.45, x=0.5,
                                  xanchor="center", title_text=""))
    _save(fig, filename)


# ─────────────────────────────────────────────────────────────
# 02 — IoU distribution (all 17 methods)
# ─────────────────────────────────────────────────────────────
def fig_iou_distribution(df: pd.DataFrame) -> None:
    order = (
        df.groupby("method")[PRIMARY].mean()
        .sort_values(ascending=False).index.tolist()
    )
    fig = px.box(
        df, x="method", y=PRIMARY, color="tier",
        category_orders={"method": order, "tier": ["Advanced", "Simple", "Base"]},
        color_discrete_map=TIER_COLORS, points="outliers",
    )
    _style(
        fig,
        title="IoU@q=0.90 distribution per method  ·  ordered by mean",
        xtitle="Method",
        ytitle="IoU vs induced weakspot",
        legend=True,
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.45, x=0.5,
                                  xanchor="center", title_text=""))
    _save(fig, "02_iou_distribution.png")


# ─────────────────────────────────────────────────────────────
# 03 — base methods only
# ─────────────────────────────────────────────────────────────
def fig_base_methods(df: pd.DataFrame) -> None:
    sub = df[df["method"].isin(BASE_METHODS)]
    order = (
        sub.groupby("method")[PRIMARY].median()
        .sort_values(ascending=False).index.tolist()
    )
    fig = px.box(
        sub, x="method", y=PRIMARY, color="method",
        category_orders={"method": order}, points="outliers",
    )
    _style(
        fig,
        title="IoU@q=0.90 — eight base methods only",
        xtitle="Method",
        ytitle="IoU vs induced weakspot",
        legend=False,
    )
    _save(fig, "03_base_methods_iou.png")


# ─────────────────────────────────────────────────────────────
# 04–07 — Sensitivity to a single sweep dim
# ─────────────────────────────────────────────────────────────
def fig_sensitivity(df: pd.DataFrame, dim: str, filename: str,
                    xtitle: str, *,
                    metric: str = PRIMARY,
                    title_metric: str = "Mean IoU@q=0.90",
                    ytitle: str = "Mean IoU",
                    lower_is_better: bool = False) -> None:
    if dim not in df.columns:
        print(f"  skip {filename} — column {dim} missing from CSV.")
        return
    sub = (
        df.groupby(["method", dim])[metric]
        .agg(["mean", "std"]).reset_index()
    )
    sub["std"] = sub["std"].fillna(0.0)
    method_order = (
        df.groupby("method")[metric].mean()
        .sort_values(ascending=lower_is_better).index.tolist()
    )
    fig = px.line(
        sub, x=dim, y="mean", color="method", error_y="std",
        markers=True, category_orders={"method": method_order},
    )
    _style(
        fig,
        title=f"{title_metric} vs {xtitle}",
        xtitle=xtitle,
        ytitle=ytitle,
        legend=True,
    )
    fig.update_xaxes(tickangle=0)
    # Enlarged fonts: the sensitivity line charts (Figs 7-9 in the paper) were
    # hard to read at the previous sizes.
    fig.update_layout(
        title=dict(font=dict(size=24)),
        xaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=17)),
        yaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=17)),
        legend=dict(orientation="h", y=-0.38, x=0.5,
                    xanchor="center", title_text="",
                    font=dict(size=15)),
    )
    _save(fig, filename, height=640)


# ─────────────────────────────────────────────────────────────
# 08 — 2D location heatmap, faceted per method
# ─────────────────────────────────────────────────────────────
def fig_location_heatmap(df: pd.DataFrame) -> None:
    if not {"center_x", "center_y"}.issubset(df.columns):
        print("  skip 08_sensitivity_location.png — center_x / center_y missing.")
        return
    # Sort by mean IoU so the best methods appear first
    method_order = (
        df.groupby("method")[PRIMARY].mean()
        .sort_values(ascending=False).index.tolist()
    )
    sub = (
        df.groupby(["method", "center_x", "center_y"])[PRIMARY]
        .mean().reset_index()
    )
    fig = px.density_heatmap(
        sub, x="center_x", y="center_y", z=PRIMARY,
        facet_col="method", facet_col_wrap=4,
        category_orders={"method": method_order},
        histfunc="avg", nbinsx=3, nbinsy=3,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        title=dict(text="Mean IoU@q=0.90 per weakspot location, faceted per method",
                   font=dict(size=18)),
        font=dict(family="Times New Roman, serif"),
        plot_bgcolor="white",
        margin=dict(l=50, r=30, t=70, b=50),
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1],
                                                font=dict(size=11)))
    _save(fig, "08_sensitivity_location.png", height=900)


# ─────────────────────────────────────────────────────────────
# 09 — IoU vs distance scatter
# ─────────────────────────────────────────────────────────────
def fig_distance_vs_iou(df: pd.DataFrame) -> None:
    if "distance" not in df.columns:
        print("  skip 09_distance_vs_iou.png — distance column missing.")
        return
    method_order = (
        df.groupby("method")[PRIMARY].mean()
        .sort_values(ascending=False).index.tolist()
    )
    fig = px.scatter(
        df.sample(min(len(df), 4000), random_state=0),
        x="distance", y=PRIMARY, color="method", opacity=0.45,
        category_orders={"method": method_order},
    )
    _style(
        fig,
        title="IoU@q=0.90 vs centroid distance — top-left = ideal",
        xtitle="Centroid distance to induced centre",
        ytitle="IoU vs induced weakspot",
        legend=True,
    )
    fig.update_xaxes(tickangle=0)
    fig.update_layout(legend=dict(orientation="h", y=-0.35, x=0.5,
                                  xanchor="center", title_text="",
                                  font=dict(size=9)))
    _save(fig, "09_distance_vs_iou.png", height=620)


# ─────────────────────────────────────────────────────────────
# Method summary CSV (for pasting into the paper)
# ─────────────────────────────────────────────────────────────
def write_summary(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    iou_cols = [c for c in df.columns if c.startswith("iou_q")]
    cols = iou_cols + ["distance", "sigma_x1", "sigma_x2", "area_frac"]
    cols = [c for c in cols if c in df.columns]

    rows = []
    for m, sub in df.groupby("method"):
        row = {"method": m, "tier": TIER_OF.get(m, "?"), "n_runs": len(sub)}
        for c in cols:
            row[f"mean_{c}"]   = round(sub[c].mean(), 4)
            row[f"median_{c}"] = round(sub[c].median(), 4)
        rows.append(row)
    out = (
        pd.DataFrame(rows)
        .sort_values(f"mean_{PRIMARY}", ascending=False)
        .reset_index(drop=True)
    )
    path = OUT_DIR / "method_summary.csv"
    out.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(out)} rows × {len(out.columns)} cols)")


# ─────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Reading {CSV_PATH} …")
    df = _load()
    print(f"Writing figures + summary to {OUT_DIR.resolve()} …")

    # ── IoU perspective (how well the area is described) ───────────
    fig_method_ranking(
        df, metric=PRIMARY, filename="01_method_ranking.png",
        title="Mean IoU@q=0.90 per method",
        ytitle="Mean IoU vs induced weakspot",
        lower_is_better=False,
    )
    fig_iou_distribution(df)
    fig_base_methods(df)
    fig_sensitivity(df, "noise_std", "04_sensitivity_noise.png",
                    "noise σ on the underlying function")
    fig_sensitivity(df, "n_bumps", "05_sensitivity_complexity.png",
                    "function complexity (n_bumps)")
    fig_sensitivity(df, "radius", "06_sensitivity_radius.png",
                    "exclusion radius")
    fig_sensitivity(df, "n_samples", "07_sensitivity_samples.png",
                    "training-set size (n_samples)")
    fig_location_heatmap(df)
    fig_distance_vs_iou(df)

    # ── Distance perspective (how well the centre is found) ────────
    fig_method_ranking(
        df, metric="distance", filename="10_method_ranking_distance.png",
        title="Mean centroid distance per method",
        ytitle="Mean distance to induced centre",
        lower_is_better=True,
    )
    fig_sensitivity(df, "noise_std", "11_sensitivity_noise_distance.png",
                    "noise σ on the underlying function",
                    metric="distance",
                    title_metric="Mean centroid distance",
                    ytitle="Mean distance to induced centre",
                    lower_is_better=True)
    fig_sensitivity(df, "n_bumps", "12_sensitivity_complexity_distance.png",
                    "function complexity (n_bumps)",
                    metric="distance",
                    title_metric="Mean centroid distance",
                    ytitle="Mean distance to induced centre",
                    lower_is_better=True)
    fig_sensitivity(df, "radius", "13_sensitivity_radius_distance.png",
                    "exclusion radius",
                    metric="distance",
                    title_metric="Mean centroid distance",
                    ytitle="Mean distance to induced centre",
                    lower_is_better=True)
    fig_sensitivity(df, "n_samples", "14_sensitivity_samples_distance.png",
                    "training-set size (n_samples)",
                    metric="distance",
                    title_metric="Mean centroid distance",
                    ytitle="Mean distance to induced centre",
                    lower_is_better=True)

    write_summary(df)
    print("Done.")


if __name__ == "__main__":
    main()
