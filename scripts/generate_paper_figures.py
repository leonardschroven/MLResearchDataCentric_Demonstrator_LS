"""
Generate the five PNG figures referenced by ``Documents/Paper_Advanced.../main.tex``:

    base_methods_iou.png
    ensemble_iou_distribution.png
    sensitivity_noise.png
    sensitivity_complexity.png
    sensitivity_radius.png

Source: ``data/experiment_results/weakspot_experiment/sweep_results.csv`` (the live sweep CSV).
Output: ``Documents/Paper_Advanced EnsembleMethods/Paper_Advanced-Ensembles-for-Weakspot-Identification/figures/``.

Runs gracefully when the sweep is partial — uses whatever rows are present.
Skips a figure (with a warning) if any required column is missing.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


CSV_PATH = Path("data/experiment_results/weakspot_experiment/sweep_results.csv")
OUT_DIR  = Path(
    "Documents/Paper_Advanced EnsembleMethods/"
    "Paper_Advanced-Ensembles-for-Weakspot-Identification/figures"
)

# Method groupings — must match scripts/weakspot/detection.py registry keys.
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

PRIMARY = "iou_q90"   # paper's headline metric
WIDTH, HEIGHT, SCALE = 1100, 600, 2   # 2× scale → ~300 DPI in print
SENSITIVITY_HEIGHT = 520


def _load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"  CSV not found: {CSV_PATH}.  Run the sweep first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    if PRIMARY not in df.columns:
        print(f"  CSV missing column {PRIMARY}.")
        sys.exit(1)
    print(f"  Loaded {len(df):,} rows, "
          f"{df['param_key'].nunique() if 'param_key' in df else '?'} configs, "
          f"{df['method'].nunique()} methods.")
    return df


def _save(fig: go.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.write_image(str(path), width=WIDTH, height=HEIGHT, scale=SCALE)
    print(f"  wrote {path}")


def _layout(fig: go.Figure, *, title: str, yaxis: str, xaxis: str = "Method") -> None:
    """Common paper-friendly styling: large fonts, no chrome."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title=dict(text=xaxis, font=dict(size=15)),
                   tickfont=dict(size=12), tickangle=-30),
        yaxis=dict(title=dict(text=yaxis, font=dict(size=15)),
                   tickfont=dict(size=12)),
        font=dict(family="Times New Roman, serif"),
        plot_bgcolor="white",
        margin=dict(l=70, r=30, t=70, b=130),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor="black", mirror=True)
    fig.update_yaxes(gridcolor="lightgray", linecolor="black", mirror=True)


# ─────────────────────────────────────────────────────────────
# Figure 1 — Base methods, IoU distribution
# ─────────────────────────────────────────────────────────────
def fig_base_methods_iou(df: pd.DataFrame) -> None:
    sub = df[df["method"].isin(BASE_METHODS)].copy()
    if sub.empty:
        print("  skip base_methods_iou.png — no rows for base methods.")
        return
    order = (
        sub.groupby("method")[PRIMARY].median()
        .sort_values(ascending=False).index.tolist()
    )
    fig = px.box(
        sub, x="method", y=PRIMARY, color="method",
        category_orders={"method": order}, points="outliers",
    )
    _layout(
        fig,
        title=f"Distribution of IoU@q=0.90 — 8 base methods",
        yaxis="IoU vs induced weakspot",
    )
    _save(fig, "base_methods_iou.png")


# ─────────────────────────────────────────────────────────────
# Figure 2 — All methods, IoU distribution (base + simple + advanced)
# ─────────────────────────────────────────────────────────────
def fig_ensemble_iou(df: pd.DataFrame) -> None:
    all_methods = BASE_METHODS + SIMPLE_ENSEMBLES + ADVANCED_ENSEMBLES
    sub = df[df["method"].isin(all_methods)].copy()
    if sub.empty:
        print("  skip ensemble_iou_distribution.png — no rows.")
        return
    order = (
        sub.groupby("method")[PRIMARY].mean()
        .sort_values(ascending=False).index.tolist()
    )
    # Colour methods by tier so the visual separates groups
    tier_of = {m: "Base" for m in BASE_METHODS}
    tier_of.update({m: "Simple"   for m in SIMPLE_ENSEMBLES})
    tier_of.update({m: "Advanced" for m in ADVANCED_ENSEMBLES})
    sub["tier"] = sub["method"].map(tier_of)

    fig = px.box(
        sub, x="method", y=PRIMARY, color="tier",
        category_orders={"method": order,
                         "tier": ["Advanced", "Simple", "Base"]},
        color_discrete_map={"Advanced": "#1f77b4",
                            "Simple":   "#888888",
                            "Base":     "#ff7f0e"},
        points="outliers",
    )
    _layout(
        fig,
        title=f"IoU@q=0.90 — base, simple, and advanced ensembles",
        yaxis="IoU vs induced weakspot",
    )
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", y=-0.35, x=0.5,
                                  xanchor="center", title_text=""))
    _save(fig, "ensemble_iou_distribution.png")


# ─────────────────────────────────────────────────────────────
# Figures 3–5 — Sensitivity to noise / complexity / radius
# ─────────────────────────────────────────────────────────────
def fig_sensitivity(df: pd.DataFrame, dim: str, filename: str,
                    xtitle: str) -> None:
    if dim not in df.columns:
        print(f"  skip {filename} — column {dim} missing (sweep may not have run with it).")
        return
    sub = (
        df.groupby(["method", dim])[PRIMARY]
        .agg(["mean", "std", "count"]).reset_index()
    )
    if sub.empty:
        print(f"  skip {filename} — empty after groupby.")
        return
    sub["std"] = sub["std"].fillna(0.0)
    method_order = (
        df.groupby("method")[PRIMARY].mean()
        .sort_values(ascending=False).index.tolist()
    )
    fig = px.line(
        sub, x=dim, y="mean", color="method", error_y="std",
        markers=True, category_orders={"method": method_order},
    )
    fig.update_layout(
        title=dict(text=f"Mean IoU@q=0.90 vs {xtitle}", font=dict(size=18)),
        xaxis=dict(title=dict(text=xtitle, font=dict(size=15)),
                   tickfont=dict(size=12)),
        yaxis=dict(title=dict(text="Mean IoU", font=dict(size=15)),
                   tickfont=dict(size=12), gridcolor="lightgray"),
        font=dict(family="Times New Roman, serif"),
        plot_bgcolor="white",
        margin=dict(l=70, r=30, t=70, b=80),
        showlegend=True,
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center",
                    title_text="", font=dict(size=10)),
    )
    fig.update_xaxes(showgrid=False, linecolor="black", mirror=True)
    fig.update_yaxes(linecolor="black", mirror=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.write_image(str(path), width=WIDTH, height=SENSITIVITY_HEIGHT, scale=SCALE)
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Reading {CSV_PATH} …")
    df = _load()
    print(f"Writing figures to {OUT_DIR} …")

    fig_base_methods_iou(df)
    fig_ensemble_iou(df)
    fig_sensitivity(df, "noise_std",  "sensitivity_noise.png",
                    "noise σ on the underlying function")
    fig_sensitivity(df, "n_bumps",    "sensitivity_complexity.png",
                    "function complexity (n_bumps)")
    fig_sensitivity(df, "radius",     "sensitivity_radius.png",
                    "exclusion radius")
    print("Done.")


if __name__ == "__main__":
    main()
