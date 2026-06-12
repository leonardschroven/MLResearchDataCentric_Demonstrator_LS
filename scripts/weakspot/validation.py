"""
Ground-truth validation for the weakspot experiment.

The weakspot experiment treats the *induced* data gap (the region from which
training points were withheld) as the ground-truth weakspot. This module
provides the evidence that this is a faithful proxy: it varies the induced
weakspot location across the same 3x3 centre grid used by the parameter
sweep, trains the regression model exactly as the experiment does, and shows
that the model's prediction error concentrates inside the induced gap at
every location.

Everything here reuses the experiment's own building blocks
(``datasets`` -> ``models``) so the validation cannot drift away from the
experiment it is meant to validate. ``run_alignment_grid`` does the
computation only; ``render_alignment_figure`` turns its output into a
publication-quality Matplotlib figure. Both the standalone figure script
(``scripts/generate_alignment_figure.py``) and the Streamlit app call the
same two functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .datasets import (
    load_dataset, normalize, apply_exclusion_zone, make_train_test_split,
)
from .models import AVAILABLE_MODELS, build_model, train_and_evaluate
from .detection import create_grid

# The 3x3 grid of induced-weakspot centres used by the parameter sweep
# (scripts/weakspot/sweep.py: SWEEP_GRID["center_x"] x ["center_y"]).
SWEEP_CENTRES: tuple[float, ...] = (0.25, 0.50, 0.75)


@dataclass
class AlignmentPanel:
    """Per-location result of the alignment study."""
    center: tuple[float, float]
    radius: float
    xx: np.ndarray
    yy: np.ndarray
    err_surface: np.ndarray          # model |error| interpolated on the grid, [0,1]
    X_excl: np.ndarray               # the withheld (ground-truth) points
    mean_err_in: float               # mean model error inside the induced gap
    mean_err_out: float              # mean model error outside the induced gap
    ratio: float                     # mean_err_in / mean_err_out
    err_surface_raw: np.ndarray | None = None   # un-normalised error (for differencing)
    metrics: dict = field(default_factory=dict)


def _interpolate_error_surface_raw(X_norm, errors, grid_flat, shape):
    """Interpolate per-point errors onto the grid (linear, nearest fallback).

    Returns the raw (un-normalised) error magnitude so that surfaces from
    different runs can be subtracted from one another. No detection model is
    involved, only the model's raw errors, identical in spirit to the
    ground-truth surface built in the Streamlit experiment page.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    surf = LinearNDInterpolator(X_norm, errors)(grid_flat)
    nan = np.isnan(surf)
    if nan.any():
        surf[nan] = NearestNDInterpolator(X_norm, errors)(grid_flat[nan])
    return surf.reshape(shape)


def _normalise01(surf):
    """Scale a surface to [0, 1] for display (NaN-safe)."""
    lo, hi = float(np.nanmin(surf)), float(np.nanmax(surf))
    return (surf - lo) / (hi - lo) if hi > lo else surf


def run_alignment_grid(
    *,
    dataset_key: str = "gaussian_bumps",
    centres: list[tuple[float, float]] | None = None,
    radius: float = 0.15,
    n_bumps: int = 5,
    noise_std: float = 0.10,
    model_name: str = "MLP Neural Network",
    complexity: float = 0.5,
    iterations: int = 100,
    n_samples: int = 1500,
    test_size: float = 0.20,
    grid_res: int = 60,
    random_state: int = 42,
) -> list[AlignmentPanel]:
    """Train the experiment's model at several induced-weakspot locations.

    Returns one :class:`AlignmentPanel` per centre. The model, dataset, and
    exclusion logic are exactly those used by the single-run experiment and
    the sweep, so the result reflects the real experimental pipeline.
    """
    if centres is None:
        centres = [(cx, cy) for cy in SWEEP_CENTRES for cx in SWEEP_CENTRES]
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name!r}")

    X, y, *_ = load_dataset(
        dataset_key, n_samples=n_samples,
        n_bumps=int(n_bumps), noise_std=float(noise_std),
        random_state=random_state,
    )
    X_norm, _ = normalize(X)
    xx, yy, grid_flat = create_grid(resolution=grid_res)

    panels: list[AlignmentPanel] = []
    for cx, cy in centres:
        center = np.array([cx, cy])

        # Withhold the induced region, then train exactly as the experiment does:
        # split the *surviving* points into train/test, fit on the train split,
        # and evaluate errors on the full dataset (including the withheld gap).
        X_tr, y_tr, X_excl, _y_excl, _ = apply_exclusion_zone(X_norm, y, center, radius)
        X_train, _, y_train, _ = make_train_test_split(
            X_tr, y_tr, excl_mask=None, test_size=test_size, random_state=random_state
        )
        model = build_model(
            AVAILABLE_MODELS[model_name], complexity=complexity,
            iterations=iterations, random_state=random_state,
        )
        _y_pred, errors, metrics = train_and_evaluate(
            model, X_train, y_train, X_norm, y
        )

        err_raw = _interpolate_error_surface_raw(X_norm, errors, grid_flat, xx.shape)
        err_surface = _normalise01(err_raw)

        # Quantitative alignment: how much larger is the error inside the gap?
        dist_pts = np.hypot(X_norm[:, 0] - cx, X_norm[:, 1] - cy)
        inside = dist_pts <= radius
        mean_in = float(errors[inside].mean()) if inside.any() else float("nan")
        mean_out = float(errors[~inside].mean()) if (~inside).any() else float("nan")
        ratio = mean_in / mean_out if mean_out and mean_out > 0 else float("nan")

        panels.append(AlignmentPanel(
            center=(float(cx), float(cy)), radius=float(radius),
            xx=xx, yy=yy, err_surface=err_surface, X_excl=X_excl,
            mean_err_in=mean_in, mean_err_out=mean_out, ratio=ratio,
            err_surface_raw=err_raw, metrics=metrics,
        ))
    return panels


def run_baseline_error_surface(
    *,
    dataset_key: str = "gaussian_bumps",
    n_bumps: int = 5,
    noise_std: float = 0.10,
    model_name: str = "MLP Neural Network",
    complexity: float = 0.5,
    iterations: int = 100,
    n_samples: int = 1500,
    test_size: float = 0.20,
    grid_res: int = 60,
    random_state: int = 42,
):
    """Train the model on the *complete* data (no induced weakspot) and return
    its raw error surface on the grid: ``(xx, yy, baseline_raw)``.

    This baseline captures the error the model makes from noise and landscape
    difficulty alone. Subtracting it from a with-weakspot surface isolates the
    error caused specifically by withholding the data.
    """
    X, y, *_ = load_dataset(
        dataset_key, n_samples=n_samples, n_bumps=int(n_bumps),
        noise_std=float(noise_std), random_state=random_state,
    )
    X_norm, _ = normalize(X)
    xx, yy, grid_flat = create_grid(resolution=grid_res)
    X_train, _, y_train, _ = make_train_test_split(
        X_norm, y, excl_mask=None, test_size=test_size, random_state=random_state
    )
    model = build_model(
        AVAILABLE_MODELS[model_name], complexity=complexity,
        iterations=iterations, random_state=random_state,
    )
    _pred, errors, _metrics = train_and_evaluate(model, X_train, y_train, X_norm, y)
    baseline_raw = _interpolate_error_surface_raw(X_norm, errors, grid_flat, xx.shape)
    return xx, yy, baseline_raw


def render_alignment_figure(
    panels: list[AlignmentPanel],
    *,
    model_name: str = "MLP Neural Network",
    n_bumps: int = 5,
    noise_std: float = 0.10,
    title: str | None = None,
):
    """Render the alignment panels as a Matplotlib figure and return it.

    Each panel shows the model's (per-panel normalised) absolute-error surface
    with the induced exclusion disk overlaid as a dashed outline, annotated
    with the inside/outside error ratio. A figure object is returned so the
    caller decides whether to ``savefig`` (script) or ``st.pyplot`` (app).
    """
    import matplotlib
    matplotlib.use("Agg")  # safe headless default; Streamlit handles its own
    import matplotlib.pyplot as plt
    from matplotlib import cm

    n = len(panels)
    ncols = 3 if n % 3 == 0 else (2 if n % 2 == 0 else n)
    nrows = int(np.ceil(n / ncols))

    # Compact panels with larger, more legible text (smaller figure, bigger
    # fonts after scaling into the column).
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows),
        squeeze=False, constrained_layout=True,
    )
    cmap = cm.get_cmap("viridis")  # perceptually uniform, colourblind-safe
    mesh = None

    for ax, p in zip(axes.ravel(), panels):
        mesh = ax.contourf(
            p.xx, p.yy, p.err_surface, levels=20, cmap=cmap, vmin=0.0, vmax=1.0,
        )
        # Induced (ground-truth) weakspot outline.
        ax.add_patch(plt.Circle(
            p.center, p.radius, fill=False, linewidth=1.8,
            edgecolor="white", linestyle="--",
        ))
        ax.plot(*p.center, marker="+", color="white", markersize=9, markeredgewidth=1.8)
        # Withheld points, lightly, to show the gap is genuinely empty.
        if len(p.X_excl):
            ax.scatter(p.X_excl[:, 0], p.X_excl[:, 1], s=4, c="white",
                       alpha=0.35, linewidths=0)
        ratio_txt = "n/a" if not np.isfinite(p.ratio) else f"{p.ratio:.1f}x"
        ax.text(
            0.04, 0.96, f"err in/out = {ratio_txt}",
            transform=ax.transAxes, va="top", ha="left", fontsize=12,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55),
        )
        ax.set_title(f"centre = ({p.center[0]:.2f}, {p.center[1]:.2f})", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=11)
        ax.set_aspect("equal")

    # Hide any unused axes.
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    if mesh is not None:
        # Colourbar on the right, with clean even-spaced ticks starting at zero.
        cbar = fig.colorbar(mesh, ax=axes, location="right", shrink=0.85,
                            pad=0.02, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_label("model absolute error (per-panel normalised)", fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    if title is None:
        title = (
            "Model error concentrates in the induced data gap "
            "across weakspot locations\n"
            f"({model_name}, Gaussian bumps: n_bumps={n_bumps}, "
            f"noise sigma={noise_std})"
        )
    fig.suptitle(title, fontsize=15)
    return fig


def render_alignment_delta_figure(
    panels: list[AlignmentPanel],
    baseline_raw: np.ndarray,
    *,
    model_name: str = "MLP Neural Network",
    n_bumps: int = 5,
    noise_std: float = 0.10,
    title: str | None = None,
):
    """Render, per location, the with-weakspot error surface minus the
    no-weakspot baseline (``run_baseline_error_surface``).

    Positive (warm) residual marks error the withheld data *caused*; the common
    background error from noise and landscape difficulty cancels, so the gap
    stands out far more cleanly than in the raw view. A figure is returned for
    the caller to ``savefig`` or ``st.pyplot``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deltas = [p.err_surface_raw - baseline_raw for p in panels]
    # Shared symmetric colour range, robust to a few extreme pixels. Round the
    # range up to a clean even-tenth top so the colourbar starts and ends on
    # even numbers (e.g. -0.8 ... +0.8).
    allabs = np.abs(np.concatenate([d.ravel() for d in deltas]))
    vmax = float(np.percentile(allabs, 99)) or 1.0
    top = float(np.ceil(vmax / 0.2) * 0.2)
    step = 0.2 if top <= 0.4 else 0.4 if top <= 1.0 else round(top / 4, 1)
    cticks = list(np.round(np.arange(-top, top + 1e-9, step), 2))
    levels = np.linspace(-top, top, 21)

    n = len(panels)
    ncols = 3 if n % 3 == 0 else (2 if n % 2 == 0 else n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows),
        squeeze=False, constrained_layout=True,
    )
    mesh = None
    for ax, p, d in zip(axes.ravel(), panels, deltas):
        # Same viridis colour scheme as the alignment figure.
        mesh = ax.contourf(p.xx, p.yy, d, levels=levels, cmap="viridis",
                           vmin=-top, vmax=top, extend="both")
        ax.add_patch(plt.Circle(
            p.center, p.radius, fill=False, linewidth=1.8,
            edgecolor="white", linestyle="--",
        ))
        ax.plot(*p.center, marker="+", color="white", markersize=9, markeredgewidth=1.6)
        ax.set_title(f"centre = ({p.center[0]:.2f}, {p.center[1]:.2f})", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=11)
        ax.set_aspect("equal")
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    if mesh is not None:
        # Colourbar on the right with even start/end ticks (see cticks above).
        cbar = fig.colorbar(mesh, ax=axes, location="right", shrink=0.85,
                            pad=0.02, ticks=cticks)
        cbar.set_label("error change vs. no-weakspot baseline", fontsize=13)
        cbar.ax.tick_params(labelsize=11)

    if title is None:
        title = (
            "Error caused by withholding the data "
            "(induced minus no-weakspot baseline)\n"
            f"({model_name}, Gaussian bumps: n_bumps={n_bumps}, "
            f"noise sigma={noise_std})"
        )
    fig.suptitle(title, fontsize=15)
    return fig
