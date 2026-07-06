"""
Core building blocks for the data-selective-training experiment.

Everything operates in the normalised unit square [0,1]^2 so the surfaces plug
straight into ``scripts.weakspot`` (detection, extraction, IoU). The synthetic
ground-truth landscape is the same Gaussian-bumps pool used by the Weakspot
Experiment, rescaled from [0,6]^2 to [0,1]^2, exposed here as an evaluable
function ``true_function`` so that freshly selected / evaluation points can be
labelled on demand.

Pipeline stages (see the page ``07_Data_Selective_Training.py``):

    Setup → Training → Evaluation → Weakspot ID → Data Selection →
    Retraining → Re-Evaluation → Overview

Design choices
--------------
* **Noise** is added to *training* and *selected* labels only. The evaluation
  set is labelled with the noiseless ground truth, so the reported RMSE/MAE/R²
  and the error surface measure genuine model error (and therefore genuine
  weakspots) rather than label noise.
* **Distribution shift** is a covariate shift: training inputs are drawn from a
  mixture of a uniform component and a Gaussian-concentrated component, so a
  fraction of the data piles up around ``shift_center`` and other regions become
  under-sampled. The evaluation set is always uniform (the deployment
  distribution).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scripts.weakspot.datasets import (
    _BUMP_POOL_CENTRES, _BUMP_POOL_HEIGHTS, _BUMP_POOL_WIDTHS, _DEFAULT_N_BUMPS,
)

# Landscape defined directly on the unit square [0,1]^2 (pool is on [0,6]^2).
_C01 = _BUMP_POOL_CENTRES / 6.0
_W01 = _BUMP_POOL_WIDTHS / 6.0
_H01 = _BUMP_POOL_HEIGHTS


# ─────────────────────────────────────────────────────────────
# Ground-truth landscape
# ─────────────────────────────────────────────────────────────
def true_function(Xn, n_bumps: int = _DEFAULT_N_BUMPS) -> np.ndarray:
    """Noiseless ground-truth target on [0,1]^2 for arbitrary points.

    ``Xn`` : (N, 2) inputs in [0,1]^2. Returns (N,) targets.
    Complexity is the first ``n_bumps`` of the fixed 10-bump pool.
    """
    Xn = np.atleast_2d(np.asarray(Xn, dtype=float))
    nb = max(1, min(int(n_bumps), len(_C01)))
    y = np.zeros(len(Xn))
    for (cx, cy), h, w in zip(_C01[:nb], _H01[:nb], _W01[:nb]):
        y += h * np.exp(-((Xn[:, 0] - cx) ** 2 + (Xn[:, 1] - cy) ** 2) / (2 * w ** 2))
    return y


def landscape_surface(grid_flat, n_bumps: int = _DEFAULT_N_BUMPS) -> np.ndarray:
    """True function evaluated on a detection grid (for plotting the landscape)."""
    return true_function(grid_flat, n_bumps=n_bumps)


# ─────────────────────────────────────────────────────────────
# Sampling (with optional covariate distribution shift)
# ─────────────────────────────────────────────────────────────
def sample_inputs(n, rng, shift_strength: float = 0.0,
                  shift_center=(0.5, 0.5), shift_spread: float = 0.15) -> np.ndarray:
    """Sample ``n`` inputs in [0,1]^2.

    ``shift_strength`` in [0,1] is the fraction of points drawn from a Gaussian
    concentrated at ``shift_center`` (spread ``shift_spread``); the remainder is
    uniform. 0 = fully uniform (no shift).
    """
    n = int(n)
    s = float(np.clip(shift_strength, 0.0, 1.0))
    n_shift = int(round(s * n))
    n_unif = n - n_shift
    parts = []
    if n_unif > 0:
        parts.append(rng.uniform(0.0, 1.0, size=(n_unif, 2)))
    if n_shift > 0:
        g = rng.normal(loc=np.asarray(shift_center, dtype=float),
                       scale=float(shift_spread), size=(n_shift, 2))
        parts.append(np.clip(g, 0.0, 1.0))
    X = np.vstack(parts) if parts else rng.uniform(0.0, 1.0, size=(n, 2))
    rng.shuffle(X)
    return X


def label(Xn, rng, n_bumps: int, noise_std: float) -> np.ndarray:
    """Label inputs with the ground truth plus optional Gaussian noise."""
    y = true_function(Xn, n_bumps=n_bumps)
    if noise_std and noise_std > 0:
        y = y + rng.normal(0.0, float(noise_std), size=len(y))
    return y


def induce_weakspot(Xn, y, center, radius: float):
    """Split into kept (outside the circle) and excluded (inside) — the induced gap.

    Returns: X_keep, y_keep, X_excl, y_excl.
    """
    d = np.linalg.norm(Xn - np.asarray(center, dtype=float), axis=1)
    inside = d <= float(radius)
    return Xn[~inside], y[~inside], Xn[inside], y[inside]


# ─────────────────────────────────────────────────────────────
# Weakspot-guided data selection
# ─────────────────────────────────────────────────────────────
# Available selection strategies (the first is the legacy default that every
# previously-swept CSV row used). Kept in one place so the pages and the sweep
# stay in sync.
SEL_METHODS = ("Weakpoint distance", "Weight by landscape", "Shape aware")
DEFAULT_SEL_METHOD = SEL_METHODS[0]


def gaussian_weights(X_pool, center, sigma: float) -> np.ndarray:
    """Isotropic Gaussian selection weight for each candidate around ``center``."""
    d2 = np.sum((X_pool - np.asarray(center, dtype=float)) ** 2, axis=1)
    return np.exp(-d2 / (2.0 * max(float(sigma), 1e-3) ** 2))


def surface_weights(X_pool, surf) -> np.ndarray:
    """Weight each candidate by the (normalised) detection surface at its location.

    ``surf`` is the flattened detection surface over a regular ``res × res`` grid
    on [0,1]^2 (as produced by ``create_grid`` → ``column_stack(xx.ravel, yy.ravel)``),
    so the nearest grid cell is found by rounding the candidate coordinates. Unlike
    the distance kernel this is **multi-modal** — every weak region gets sampled in
    proportion to how weak it is — and needs no single extracted centre.
    """
    surf = np.asarray(surf, dtype=float).ravel()
    res = int(round(np.sqrt(len(surf))))
    if res * res != len(surf):
        return np.ones(len(X_pool))            # not a square grid → degenerate
    g = surf.reshape(res, res)                 # g[i, j] at (x=lin[j], y=lin[i])
    j = np.clip(np.round(np.asarray(X_pool)[:, 0] * (res - 1)).astype(int), 0, res - 1)
    i = np.clip(np.round(np.asarray(X_pool)[:, 1] * (res - 1)).astype(int), 0, res - 1)
    return np.clip(np.nan_to_num(g[i, j], nan=0.0), 0.0, None)


def shape_weights(X_pool, center, sigma: float, ext) -> np.ndarray:
    """Anisotropic Gaussian shaped to the **detected** weakspot ellipse.

    Uses the extraction's per-axis spreads (``sigma_x1``/``sigma_x2``) for the
    kernel's aspect ratio while ``sigma`` sets the overall width (the ellipse is
    renormalised so its geometric-mean spread equals ``sigma``). Falls back to the
    isotropic kernel when no ellipse was extracted.
    """
    center = np.asarray(center, dtype=float)
    sx1 = float(ext["sigma_x1"]) if ext and ext.get("sigma_x1") else 0.0
    sx2 = float(ext["sigma_x2"]) if ext and ext.get("sigma_x2") else 0.0
    if sx1 <= 0 or sx2 <= 0:
        return gaussian_weights(X_pool, center, sigma)
    geo = np.sqrt(sx1 * sx2)
    ax = max(float(sigma) * sx1 / geo, 1e-3)
    ay = max(float(sigma) * sx2 / geo, 1e-3)
    dx = np.asarray(X_pool)[:, 0] - center[0]
    dy = np.asarray(X_pool)[:, 1] - center[1]
    return np.exp(-(dx * dx / (2.0 * ax * ax) + dy * dy / (2.0 * ay * ay)))


def selection_weights(X_pool, center, sigma: float, method: str = DEFAULT_SEL_METHOD,
                      ext=None, surf=None) -> np.ndarray:
    """Dispatch to the chosen selection strategy; returns raw (unnormalised) weights.

    ``method`` ∈ :data:`SEL_METHODS`. "Weight by landscape" needs ``surf``; "Shape
    aware" needs ``ext``; both fall back to the distance kernel if their input is
    missing, so callers can always pass a ``center``/``sigma`` safely.
    """
    if method == "Weight by landscape" and surf is not None:
        w = surface_weights(X_pool, surf)
        if np.any(w > 0):
            return w
        return gaussian_weights(X_pool, center, sigma)   # degenerate surface
    if method == "Shape aware":
        return shape_weights(X_pool, center, sigma, ext)
    return gaussian_weights(X_pool, center, sigma)        # "Weakpoint distance"


def select_by_weakspot(X_pool, center, sigma: float, n_select: int, rng,
                       mode: str = "Sample ∝ weight",
                       method: str = DEFAULT_SEL_METHOD, ext=None, surf=None):
    """Pick ``n_select`` candidate indices for retraining, per the chosen strategy.

    ``method`` selects the weighting strategy (see :func:`selection_weights`);
    ``mode`` = "Sample ∝ weight" (stochastic, weighted-without-replacement) or
    "Top-weighted" (deterministic, the highest-weight points). The default
    ``method`` reproduces the original distance-kernel behaviour exactly, so
    existing call sites and swept results are unchanged.
    Returns (indices, weights).
    """
    n_select = int(min(n_select, len(X_pool)))
    w = selection_weights(X_pool, center, sigma, method, ext=ext, surf=surf) + 1e-12
    if n_select <= 0:
        return np.array([], dtype=int), w
    if mode.startswith("Top"):
        idx = np.argsort(w)[-n_select:][::-1]
    else:
        p = w / w.sum()
        idx = rng.choice(len(X_pool), size=n_select, replace=False, p=p)
    return idx, w


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate(model, X_eval, y_eval):
    """Predict, compute per-point absolute error and summary metrics."""
    y_pred = model.predict(X_eval)
    err = np.abs(y_eval - y_pred)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
        "MAE": float(mean_absolute_error(y_eval, y_pred)),
        "R2": float(r2_score(y_eval, y_pred)),
    }
    return y_pred, err, metrics


def region_error(X_eval, err, center, radius: float):
    """Mean eval error inside vs outside the induced weakspot circle."""
    d = np.linalg.norm(X_eval - np.asarray(center, dtype=float), axis=1)
    inside = d <= float(radius)
    e_in = float(err[inside].mean()) if inside.any() else float("nan")
    e_out = float(err[~inside].mean()) if (~inside).any() else float("nan")
    return e_in, e_out


def normalize_surface(surf: np.ndarray) -> np.ndarray:
    """Min-max scale a detection surface to [0,1]."""
    surf = np.asarray(surf, dtype=float)
    lo, hi = float(surf.min()), float(surf.max())
    return (surf - lo) / (hi - lo) if hi > lo else np.zeros_like(surf)
