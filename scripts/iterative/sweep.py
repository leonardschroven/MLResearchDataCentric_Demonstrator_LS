"""Resumable parameter sweep for the iterative data-selective-training study.

Mirrors ``scripts.dataselect.sweep`` — JSON grids in ``sweep_configs/``, one
results CSV per config, a ``param_key`` on every row for resume — with one
difference forced by the loop: a run is a *trajectory*, not a point, so each
configuration writes **one row per detector per iteration** (iteration 0 being
the shared initial model). That long format is what lets the visualisation page
plot learning curves, per-iteration win rates and forgetting drift without
re-running anything.

Resume contract
---------------
``make_param_key`` keys a configuration by how it *differs from the engine's
defaults* (:data:`scripts.iterative.loop.DEFAULTS`, i.e. the companion paper's
operating point). Adding a new axis later therefore leaves every existing key
untouched as long as its default is the neutral setting, so grids can be
extended indefinitely without invalidating collected results. All rows for a
configuration are appended together, so a key appears only once the whole
trajectory is on disk.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.iterative.loop import DEFAULTS, TRACKS, default_config, run_iterative
from scripts.iterative import models as M
from scripts.weakspot.detection import DETECTION_METHODS

CONFIG_DIR = Path(__file__).parent / "sweep_configs"
RESULTS_DIR = Path("data/experiment_results/iterative_data_selective_training")
DEFAULT_CONFIG = "iter_baseline"

# Set from the launching page's sidebar rather than swept.
FIXED_FIELDS = (
    "model_name", "complexity", "center_x", "center_y",
    "n_eval", "grid_res", "extract_q", "sel_mode",
    "shift_center_x", "shift_center_y", "shift_spread",
    "early_stopping", "single_shot", "tie_model_seed",
)

# Every engine parameter, in a stable order — each row records all of them, so a
# results file is self-describing even when the grid that produced it is gone.
PARAM_FIELDS = tuple(sorted(DEFAULTS.keys()))

_TRACK_METRICS = ("mae", "rmse", "r2", "err_in", "err_out", "n_train")


def available_configs() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Sweep config '{name}' not found at {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("name", name)
    cfg.setdefault("secs_per_run", 12.0)
    if not cfg.get("grid"):
        raise ValueError(f"Sweep config '{name}' has no non-empty 'grid'.")
    unknown = [d for d in cfg.get("detectors", []) if d not in DETECTION_METHODS]
    if unknown:
        raise ValueError(f"Sweep config '{name}' lists unknown detectors: {unknown}")
    bad = [k for k in cfg["grid"] if k not in DEFAULTS]
    if bad:
        raise ValueError(f"Sweep config '{name}' grids unknown parameters: {bad}")
    return cfg


def sweep_csv_path(name: str) -> Path:
    return RESULTS_DIR / f"sweep__{name}.csv"


def _build_columns() -> list[str]:
    track_cols = [f"{t}_{m}" for t in TRACKS for m in _TRACK_METRICS]
    return (
        ["param_key", "sweep_config", "method", "iteration"]
        + list(PARAM_FIELDS)
        + ["n_model_params", "lr_used", "mix_used", "sigma_used", "severity",
           "det_distance", "det_iou", "det_drift",
           "init_mae", "init_err_in", "init_err_out"]
        + track_cols
        + ["gap_mae", "gap_err_in", "gap_err_out",
           "d_mae_guided", "d_mae_random", "forget_guided", "forget_random",
           "ss_guided_mae", "ss_random_mae", "error"]
    )


def _apply_config(cfg: dict) -> None:
    global ACTIVE_CONFIG, SWEEP_CONFIG_NAME, SWEEP_GRID, SWEEP_DETECTORS
    global SECS_PER_RUN, CSV_PATH, COLUMNS
    ACTIVE_CONFIG = cfg
    SWEEP_CONFIG_NAME = cfg["name"]
    SWEEP_GRID = cfg["grid"]
    SWEEP_DETECTORS = list(cfg.get("detectors") or [DEFAULTS["detector"]])
    SECS_PER_RUN = float(cfg["secs_per_run"])
    CSV_PATH = sweep_csv_path(cfg["name"])
    COLUMNS = _build_columns()


def set_active_config(name: str) -> None:
    """Switch the active config (env var too, so parallel workers agree)."""
    os.environ["ITER_SWEEP_CONFIG"] = name
    _apply_config(load_config(name))


_apply_config(load_config(os.environ.get("ITER_SWEEP_CONFIG", DEFAULT_CONFIG)))


# ─────────────────────────────────────────────────────────────
# Grid / CSV plumbing
# ─────────────────────────────────────────────────────────────
def parameter_grid() -> list[dict]:
    keys = list(SWEEP_GRID.keys())
    return [dict(zip(keys, v))
            for v in itertools.product(*[SWEEP_GRID[k] for k in keys])]


def _norm(v):
    """Stable scalar form for keying (lists → sorted comma string)."""
    if isinstance(v, (list, tuple)):
        return ",".join(sorted(str(x) for x in v))
    return v


def make_param_key(params: dict) -> str:
    """Key a configuration by its *difference from the engine defaults*.

    Fields left at their default value are omitted, so introducing a new axis
    (defaulting to the neutral setting) leaves every previously written key
    unchanged and resume keeps working across grid revisions.
    """
    full = default_config(**{k: v for k, v in params.items() if k in DEFAULTS})
    parts = []
    for k in PARAM_FIELDS:
        v, d = _norm(full.get(k)), _norm(DEFAULTS.get(k))
        if v != d:
            parts.append(f"{k}={v}")
    return "|".join(parts) or "defaults"


def load_completed_keys(csv_path: Path | None = None) -> set[str]:
    csv_path = csv_path or CSV_PATH
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["param_key"])
    except Exception:
        try:
            df = pd.read_csv(csv_path, usecols=["param_key"],
                             engine="python", on_bad_lines="skip")
        except Exception:
            return set()
    return set(df["param_key"].dropna().unique())


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows).reindex(columns=COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(csv_path, mode="a" if csv_path.exists() else "w",
                  header=not csv_path.exists(), index=False)


def remaining_combos(fixed: dict, csv_path: Path | None = None) -> list[dict]:
    done = load_completed_keys(csv_path or CSV_PATH)
    return [full for full in ({**c, **fixed} for c in parameter_grid())
            if make_param_key(full) not in done]


# ─────────────────────────────────────────────────────────────
# One configuration = one trajectory per detector
# ─────────────────────────────────────────────────────────────
def _nan(x):
    """CSV-safe float (NaN for missing / non-finite)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def run_one(params: dict) -> list[dict]:
    """Run every detector's trajectory for one configuration.

    Returns one row per (detector × iteration). Unlike the single-round sweep the
    detectors cannot share a run past iteration 0 — each one steers its own
    selection and the trajectories diverge immediately — so the loop is executed
    once per detector, from the same seed and therefore the same initial model,
    landscape and candidate pools.
    """
    pkey = make_param_key(params)
    rows: list[dict] = []
    base = {k: v for k, v in params.items() if k in DEFAULTS}

    for name in SWEEP_DETECTORS:
        cfg = default_config(**{**base, "detector": name})
        try:
            res = run_iterative(cfg, keep_rounds=False)
        except Exception as e:                    # keep the sweep going
            rows.append({"param_key": pkey, "sweep_config": SWEEP_CONFIG_NAME,
                         "method": name, "iteration": -1,
                         **{k: _norm(cfg.get(k)) for k in PARAM_FIELDS},
                         "error": str(e)})
            continue

        H, S, D = res["tracks"], res["sched"], res["det"]
        ss = res.get("single_shot") or {}
        init = {"init_mae": H["gacc"]["mae"][0],
                "init_err_in": H["gacc"]["err_in"][0],
                "init_err_out": H["gacc"]["err_out"][0]}
        param_cols = {k: _norm(cfg.get(k)) for k in PARAM_FIELDS}
        n_par = M.n_parameters(cfg["arch"], float(cfg["complexity"]))

        for i in res["iters"]:
            row = {
                "param_key": pkey, "sweep_config": SWEEP_CONFIG_NAME,
                "method": name, "iteration": i, **param_cols,
                "n_model_params": n_par, **init,
                "ss_guided_mae": _nan(ss.get("guided", {}).get("mae")),
                "ss_random_mae": _nan(ss.get("random", {}).get("mae")),
            }
            # Per-round quantities exist for iterations 1…K only; the schedules
            # are indexed by the round that *produced* iteration i.
            if i > 0:
                j = i - 1
                row.update(lr_used=_nan(S["lr"][j]), mix_used=_nan(S["mix"][j]),
                           sigma_used=_nan(S["sigma"][j]),
                           severity=_nan(S["severity"][j]),
                           det_distance=_nan(D["distance"][j]),
                           det_iou=_nan(D["iou"][j]), det_drift=_nan(D["drift"][j]))
            for t in TRACKS:
                if t not in H:
                    continue
                for m in _TRACK_METRICS:
                    row[f"{t}_{m}"] = _nan(H[t][m][i])
            row.update(
                gap_mae=_nan(H["gacc"]["mae"][i] - H["racc"]["mae"][i]),
                gap_err_in=_nan(H["gacc"]["err_in"][i] - H["racc"]["err_in"][i]),
                gap_err_out=_nan(H["gacc"]["err_out"][i] - H["racc"]["err_out"][i]),
                d_mae_guided=_nan(H["gacc"]["mae"][i] - init["init_mae"]),
                d_mae_random=_nan(H["racc"]["mae"][i] - init["init_mae"]),
                forget_guided=_nan(H["gacc"]["err_out"][i] - init["init_err_out"]),
                forget_random=_nan(H["racc"]["err_out"][i] - init["init_err_out"]),
            )
            rows.append(row)
    return rows


def safe_run_one(params: dict) -> list[dict]:
    """``run_one`` that never raises — a failed config is simply left un-done
    and retried on the next run rather than killing a parallel batch."""
    try:
        return run_one(params)
    except Exception:
        return []
