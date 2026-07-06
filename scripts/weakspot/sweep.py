"""
Parameter sweep — runs many experiments back-to-back and appends each run's
per-method comparison-table rows to a CSV, with resume support.

Resume contract: every row carries a ``param_key`` that fully identifies the
configuration (sweep dims + fixed sidebar dims + grid resolution). On restart
we read the CSV, collect all completed param_keys, and skip those.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .datasets import (
    load_dataset, normalize, apply_exclusion_zone, make_train_test_split,
)
from .models import AVAILABLE_MODELS, build_model, train_and_evaluate
from .detection import DETECTION_METHODS, create_grid
from .extraction import (
    extract_weakspot, induced_mask, iou_ellipse_at_thresholds,
)


SWEEP_GRID = {
    "center_x":   [0.25, 0.50, 0.75],
    "center_y":   [0.25, 0.50, 0.75],
    "radius":     [0.05, 0.10, 0.15],
    "iterations": [100],                          # fixed (trimmed scope)
    "n_samples":  [500, 750, 1000, 1250, 1500],
    "n_bumps":    [3, 5, 7],                      # function complexity
    "noise_std":  [0.05, 0.15, 0.30],             # underlying-function noise σ
}

IOU_QUANTILES = (0.80, 0.85, 0.90, 0.95)
EXTRACT_QUANTILE = 0.85  # threshold used for centroid/sigma/area columns
GRID_RES_DEFAULT = 35

# Identifies a unique (sweep × fixed-config) combination for resume detection.
KEY_FIELDS = (
    "center_x", "center_y", "radius", "iterations", "n_samples",
    "n_bumps", "noise_std",
    "dataset_key", "model_name", "complexity", "test_size", "grid_res",
)


def parameter_grid() -> list[dict]:
    keys = list(SWEEP_GRID.keys())
    combos = [dict(zip(keys, v))
              for v in itertools.product(*[SWEEP_GRID[k] for k in keys])]
    return combos


def make_param_key(params: dict) -> str:
    return "|".join(f"{k}={params[k]}" for k in KEY_FIELDS)


def load_completed_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["param_key"])
    except Exception:
        return set()
    return set(df["param_key"].dropna().unique())


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode="w", header=True, index=False)


def run_one(params: dict) -> list[dict]:
    """Run a single experiment configuration; return one row per detection method."""
    pkey = make_param_key(params)

    X, y, *_ = load_dataset(
        params["dataset_key"], n_samples=params["n_samples"],
        n_bumps=params.get("n_bumps"), noise_std=params.get("noise_std"),
    )
    X_norm, _ = normalize(X)
    center = np.array([params["center_x"], params["center_y"]])

    X_tr, y_tr, X_excl, y_excl, _ = apply_exclusion_zone(
        X_norm, y, center, params["radius"]
    )
    X_train, _, y_train, _ = make_train_test_split(
        X_tr, y_tr, excl_mask=None, test_size=params["test_size"]
    )

    model = build_model(
        AVAILABLE_MODELS[params["model_name"]],
        complexity=params["complexity"],
        iterations=params["iterations"],
    )
    y_pred, errors, metrics = train_and_evaluate(
        model, X_train, y_train, X_norm, y
    )

    xx, yy, grid_flat = create_grid(resolution=params["grid_res"])
    gt_mask = induced_mask(xx, yy, "Circle (radius)", center, params["radius"], None)

    rows = []
    for name, fn in DETECTION_METHODS.items():
        try:
            surf = fn(X_norm, errors, grid_flat)
        except Exception as e:
            rows.append({
                "param_key": pkey, **params, "method": name,
                "error": str(e),
                **{f"iou_q{int(q*100)}": None for q in IOU_QUANTILES},
            })
            continue

        s_min, s_max = float(surf.min()), float(surf.max())
        if s_max > s_min:
            surf = (surf - s_min) / (s_max - s_min)

        ext = extract_weakspot(surf, xx, yy, threshold_quantile=EXTRACT_QUANTILE)
        iou = iou_ellipse_at_thresholds(surf, xx, yy, gt_mask, IOU_QUANTILES)

        if ext is None:
            cx_d = cy_d = sx1 = sx2 = area = dist = None
        else:
            cx_d, cy_d = ext["center"]
            sx1, sx2 = ext["sigma_x1"], ext["sigma_x2"]
            area = ext["area_frac"]
            dist = float(np.hypot(cx_d - center[0], cy_d - center[1]))

        row = {
            "param_key": pkey,
            **params,
            "method": name,
            "centroid_x1": cx_d, "centroid_x2": cy_d,
            "sigma_x1": sx1, "sigma_x2": sx2,
            "area_frac": area, "distance": dist,
            "model_rmse": metrics.get("RMSE"),
            "model_mae": metrics.get("MAE"),
            "model_r2": metrics.get("R²"),
        }
        for q in IOU_QUANTILES:
            row[f"iou_q{int(q * 100)}"] = iou.get(q)
        rows.append(row)

    return rows


def remaining_combos(fixed: dict, csv_path: Path) -> list[dict]:
    """Return sweep combos whose full param_key is not yet in the CSV."""
    done = load_completed_keys(csv_path)
    out = []
    for combo in parameter_grid():
        full = {**combo, **fixed}
        if make_param_key(full) not in done:
            out.append(full)
    return out
