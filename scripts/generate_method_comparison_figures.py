"""
Generate the method-comparison bar charts referenced by ``main.tex``:

    base_distance.png       mean centroid distance, 8 base methods (sorted, best first)
    base_iou.png            mean IoU@q=0.90, 8 base methods (sorted, best first)
    advanced_distance.png   mean centroid distance, all 17 methods, coloured by tier
    advanced_iou.png        mean IoU@q=0.90, all 17 methods, coloured by tier

Source : data/experiment_results/sweep_results.csv (the live sweep CSV).
Output : straight into the paper's figures/ folder, so no manual copy is needed.

Design goals (per paper feedback):
  * Two simple comparative plots per stage: one over distance, one over IoU.
  * Horizontal bars so the (long) method names read left-to-right.
  * Large fonts (~50% bigger than the earlier plots) for print legibility.

Re-run whenever the sweep CSV changes:
    python -m scripts.generate_method_comparison_figures
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("data/experiment_results/sweep_results.csv")
OUT_DIR = Path(
    "Documents/Paper_Advanced EnsembleMethods/"
    "Paper_Advanced-Ensembles-for-Weakspot-Identification/figures"
)

# Method tiers — must match scripts/weakspot/detection.py registry keys.
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
TIER = {m: "Base" for m in BASE_METHODS}
TIER.update({m: "Simple" for m in SIMPLE_ENSEMBLES})
TIER.update({m: "Advanced" for m in ADVANCED_ENSEMBLES})
TIER_COLOR = {"Base": "#ff7f0e", "Simple": "#888888", "Advanced": "#1f77b4"}

# Fonts ~50% larger than the previous figures (which used ~10-12 pt).
FS_TITLE, FS_LABEL, FS_TICK, FS_VALUE, FS_LEGEND = 17, 16, 15, 13, 14
PRIMARY = "iou_q90"


def _load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found: {CSV_PATH}. Run the sweep first.")
    df = pd.read_csv(CSV_PATH)
    for col in (PRIMARY, "distance", "method"):
        if col not in df.columns:
            sys.exit(f"CSV missing required column: {col}")
    return df


def _means(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    sub = df[df["method"].isin(methods)]
    g = sub.groupby("method").agg(iou=(PRIMARY, "mean"),
                                  dist=("distance", "mean")).reset_index()
    return g


def _barh(stats: pd.DataFrame, value: str, *, ascending_is_better: bool,
          xlabel: str, title: str, filename: str, height: float,
          show_legend: bool) -> None:
    """One horizontal bar chart. Best method is placed at the top."""
    # Sort so the best method ends up on top of the chart (after the y-axis is
    # inverted, index 0 is drawn at the top).
    ordered = stats.sort_values(value, ascending=ascending_is_better)
    methods = ordered["method"].tolist()
    values = ordered[value].tolist()
    colors = [TIER_COLOR[TIER[m]] for m in methods]

    fig, ax = plt.subplots(figsize=(8.4, height), constrained_layout=True)
    ypos = range(len(methods))
    ax.barh(list(ypos), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(methods, fontsize=FS_TICK)
    ax.invert_yaxis()  # best (first row) on top
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.grid(axis="x", color="lightgray", linewidth=0.7)
    ax.set_axisbelow(True)

    span = max(values) - min(0, min(values)) or 1.0
    for y, v in zip(ypos, values):
        ax.text(v + 0.01 * span, y, f"{v:.3f}", va="center", ha="left",
                fontsize=FS_VALUE)
    ax.set_xlim(0, max(values) * 1.16)

    if show_legend:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=TIER_COLOR[t], edgecolor="black", label=t)
                   for t in ("Base", "Simple", "Advanced")]
        # Below the axes, horizontal — never overlaps the bars.
        ax.legend(handles=handles, fontsize=FS_LEGEND, ncol=3, frameon=False,
                  loc="upper center", bbox_to_anchor=(0.5, -0.12))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# Parameter axes for the paper's sensitivity figures (Figs 7-9 in main.tex).
# (swept column, x-axis label, output filename)
SENSITIVITY_AXES = [
    ("noise_std", "noise sigma on the underlying function",
     "04_sensitivity_noise.png"),
    ("n_bumps", "function complexity (number of Gaussian bumps)",
     "05_sensitivity_complexity.png"),
    ("radius", "exclusion radius r", "06_sensitivity_radius.png"),
]


def _sensitivity_line(df: pd.DataFrame, dim: str, xlabel: str,
                      filename: str) -> None:
    """Mean IoU@q=0.90 against one swept parameter, one line per method.

    Rendered with matplotlib because the Plotly/Kaleido image-export path is
    unavailable in this environment. Large fonts for print legibility; the
    legend is sorted so the strongest methods come first.
    """
    import matplotlib.cm as cm

    if dim not in df.columns:
        print(f"  skip {filename}: column {dim} missing from CSV.")
        return
    methods = BASE_METHODS + SIMPLE_ENSEMBLES + ADVANCED_ENSEMBLES
    sub = df[df["method"].isin(methods)]
    order = (sub.groupby("method")[PRIMARY].mean()
             .sort_values(ascending=False).index.tolist())
    grp = sub.groupby(["method", dim])[PRIMARY].mean().reset_index()

    colors = cm.get_cmap("tab20")(range(len(order)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]

    fig, ax = plt.subplots(figsize=(8.6, 6.6), constrained_layout=True)
    for i, m in enumerate(order):
        d = grp[grp["method"] == m].sort_values(dim)
        ax.plot(d[dim], d[PRIMARY], color=colors[i],
                marker=markers[i % len(markers)], markersize=8,
                linewidth=2.2, alpha=0.9, label=m)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel("mean IoU @ q=0.90", fontsize=FS_LABEL)
    ax.set_title(f"Mean IoU vs {xlabel} (higher is better)", fontsize=FS_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(color="lightgray", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FS_LEGEND, ncol=2, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.13))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> int:
    df = _load()
    print(f"Loaded {len(df):,} rows, {df['method'].nunique()} methods.")

    base = _means(df, BASE_METHODS)
    allm = _means(df, BASE_METHODS + SIMPLE_ENSEMBLES + ADVANCED_ENSEMBLES)

    print(f"Writing figures to {OUT_DIR} ...")
    # --- Base methods: two simple comparative plots ---
    _barh(base, "dist", ascending_is_better=True,
          xlabel="mean centroid distance (lower is better)",
          title="Base methods: localisation accuracy",
          filename="base_distance.png", height=4.2, show_legend=False)
    _barh(base, "iou", ascending_is_better=False,
          xlabel="mean IoU @ q=0.90 (higher is better)",
          title="Base methods: region overlap",
          filename="base_iou.png", height=4.2, show_legend=False)

    # --- Advanced methods: same two metrics, all methods, coloured by tier ---
    _barh(allm, "dist", ascending_is_better=True,
          xlabel="mean centroid distance (lower is better)",
          title="All methods: localisation accuracy",
          filename="advanced_distance.png", height=6.6, show_legend=True)
    _barh(allm, "iou", ascending_is_better=False,
          xlabel="mean IoU @ q=0.90 (higher is better)",
          title="All methods: region overlap",
          filename="advanced_iou.png", height=6.6, show_legend=True)

    # --- Sensitivity line charts (Figs 7-9): mean IoU vs each swept parameter ---
    for dim, xlabel, filename in SENSITIVITY_AXES:
        _sensitivity_line(df, dim, xlabel, filename)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
