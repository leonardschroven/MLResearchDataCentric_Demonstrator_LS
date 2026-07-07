"""
Parameter sweep for the Data-Selective Training experiment.

Each configuration runs the whole pipeline and writes **one row per detector**
(the top-5 detectors share the config's initial training + evaluation, exactly like
the weakspot sweep writes one row per method):

    setup → initial train → evaluate → for each detector:
        detect weakspot → guided select → retrain on NEW points only → evaluate
    (+ a detector-independent random baseline computed once per config)

Resume contract
---------------
Every row carries a ``param_key`` built from the **full** parameter set (swept
*and* fixed). On restart we read the CSV, collect completed keys and skip them.
Because the key includes the fixed fields too, promoting a fixed axis to a swept
one later (e.g. starting to vary ``n_candidate``) still matches previously-run
rows at that value, so expanding the grid never restarts from scratch. Rows for
one config are appended together, so a key only appears once the config is fully
done.

Cost: ~2.1 s per configuration (top-5 detectors). Estimated total time is roughly
``(product of swept-axis lengths) x 2.1 s`` — see the page for a live estimate.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.weakspot.models import AVAILABLE_MODELS, build_model
from scripts.weakspot.detection import DETECTION_METHODS, create_grid
from scripts.weakspot.extraction import (
    extract_weakspot, induced_mask, iou_ellipse_at_thresholds,
)
from scripts.dataselect import pipeline as P


GRID_RES_DEFAULT = 35      # ideal grid resolution from the weakspot paper
EXTRACT_QUANTILE = 0.85    # stage-2 extraction threshold
IOU_HEADLINE_Q = 0.90      # IoU quantile recorded per row
_IOU_COL = f"iou_q{int(IOU_HEADLINE_Q * 100)}"

# The selection method that predates the ``sel_method`` axis; all historical rows
# used it, so it is treated as the key-neutral default for backward-compatible resume.
LEGACY_SEL_METHOD = P.DEFAULT_SEL_METHOD

# Held fixed at the page's sidebar values when the sweep is launched (these are NOT
# part of the config grid). ``early_stopping`` default True is key-neutral (see
# make_param_key) so it matches historical rows; False produces distinct keys.
FIXED_FIELDS = (
    "model_name", "complexity", "center_x", "center_y",
    "n_eval", "grid_res", "extract_q", "sel_mode",
    "shift_center_x", "shift_center_y", "shift_spread",
    "early_stopping",
)


# ─────────────────────────────────────────────────────────────
# Sweep configuration is loaded from a JSON file in ``sweep_configs/`` instead of
# being hardcoded, so multiple grids can coexist — a broad general sweep plus
# narrower configs that isolate a few dimensions. Each config writes to its own CSV,
# ``sweep__<config-name>.csv``, and every row carries a ``sweep_config`` column, so a
# results file is always traceable to the config that produced it. The active config
# defaults to ``DEFAULT_CONFIG`` and can be overridden with the ``SWEEP_CONFIG`` env
# var; parallel workers re-import this module and read the same var, so they agree.
# ─────────────────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent / "sweep_configs"
# Per-experiment subfolder so this page's results sit apart from the weakspot ones.
RESULTS_DIR = Path("data/experiment_results/data_selective_training")
DEFAULT_CONFIG = "alpha_boundary"


def available_configs() -> list[str]:
    """Names (filename stems) of every sweep config JSON, sorted."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def load_config(name: str) -> dict:
    """Load and validate a sweep config by name (filename without ``.json``)."""
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Sweep config '{name}' not found at {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("name", name)
    cfg.setdefault("secs_per_run", 2.1)
    if not cfg.get("grid"):
        raise ValueError(f"Sweep config '{name}' has no non-empty 'grid'.")
    unknown = [d for d in cfg.get("detectors", []) if d not in DETECTION_METHODS]
    if unknown:
        raise ValueError(f"Sweep config '{name}' lists unknown detectors: {unknown}")
    return cfg


def sweep_csv_path(name: str) -> Path:
    """Results CSV for a config — the config name is baked into the filename."""
    return RESULTS_DIR / f"sweep__{name}.csv"


def _build_columns() -> list[str]:
    # Canonical CSV column order. Every row is reindexed to this before writing, so
    # appended chunks always share identical columns (incl. the optional ``error``
    # column) — otherwise a batch with a detector failure gains a column and the CSV
    # becomes ragged/unparseable.
    return (
        ["param_key", "sweep_config"] + list(SWEEP_GRID.keys()) + list(FIXED_FIELDS)
        + ["method", "n_selected", "distance", _IOU_COL,
           "centroid_x1", "centroid_x2", "sigma_x1", "sigma_x2", "area_frac",
           "init_rmse", "init_mae", "init_r2", "init_err_in", "init_err_out",
           "guided_rmse", "guided_mae", "guided_r2", "guided_err_in", "guided_err_out",
           "base_rmse", "base_mae", "base_r2", "base_err_in", "base_err_out",
           "d_mae_guided", "d_mae_base", "gap_mae_guided_minus_base",
           "d_errin_guided", "d_errin_base", "error"]
    )


def _apply_config(cfg: dict) -> None:
    """Populate the module-level globals derived from a loaded config."""
    global ACTIVE_CONFIG, SWEEP_CONFIG_NAME, SWEEP_GRID, SWEEP_DETECTORS
    global SECS_PER_RUN, CSV_PATH, KEY_FIELDS, COLUMNS
    ACTIVE_CONFIG = cfg
    SWEEP_CONFIG_NAME = cfg["name"]
    SWEEP_GRID = cfg["grid"]
    SWEEP_DETECTORS = list(cfg["detectors"])
    SECS_PER_RUN = float(cfg["secs_per_run"])
    CSV_PATH = sweep_csv_path(cfg["name"])
    KEY_FIELDS = tuple(SWEEP_GRID.keys()) + FIXED_FIELDS   # resume key spans every field
    COLUMNS = _build_columns()


def set_active_config(name: str) -> None:
    """Switch the active sweep config (updates globals + env so workers agree)."""
    os.environ["SWEEP_CONFIG"] = name
    _apply_config(load_config(name))


# Initialise at import from the env override (default = the shipped config).
_apply_config(load_config(os.environ.get("SWEEP_CONFIG", DEFAULT_CONFIG)))


# ─────────────────────────────────────────────────────────────
# Grid / CSV plumbing (mirrors scripts.weakspot.sweep)
# ─────────────────────────────────────────────────────────────
def parameter_grid() -> list[dict]:
    keys = list(SWEEP_GRID.keys())
    return [dict(zip(keys, v))
            for v in itertools.product(*[SWEEP_GRID[k] for k in keys])]


def make_param_key(params: dict) -> str:
    # Sorted field order → the key string is stable even if an axis later moves
    # between SWEEP_GRID and FIXED_FIELDS, so resume survives grid restructuring.
    #
    # ``sel_method`` is omitted from the key when it equals the legacy method, so a
    # legacy-method config produces the *exact* key it had before the axis existed
    # and matches the historical CSV rows (which are reused). Non-legacy methods
    # keep the field, so they get distinct keys and are run as new work.
    parts = []
    for k in sorted(KEY_FIELDS):
        if k == "sel_method" and str(params.get(k, LEGACY_SEL_METHOD)) == LEGACY_SEL_METHOD:
            continue
        # early_stopping=True is the historical default → omit from the key so those
        # runs stay key-compatible; False keeps the field and gets distinct keys.
        if k == "early_stopping" and bool(params.get(k, True)):
            continue
        # mix_ratio=1.0 is pure guided (historical default) → key-neutral.
        if k == "mix_ratio" and float(params.get(k, 1.0)) == 1.0:
            continue
        parts.append(f"{k}={params[k]}")
    return "|".join(parts)


def load_completed_keys(csv_path: Path = CSV_PATH) -> set[str]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["param_key"])
    except Exception:
        # Never silently return an empty set on a readable-but-ragged file: that
        # would make the sweep re-run completed configs and append duplicates.
        # Fall back to a tolerant parse that skips only the malformed lines.
        try:
            df = pd.read_csv(csv_path, usecols=["param_key"],
                             engine="python", on_bad_lines="skip")
        except Exception:
            return set()
    return set(df["param_key"].dropna().unique())


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # Reindex to the fixed schema so every appended chunk has identical columns.
    df_new = pd.DataFrame(rows).reindex(columns=COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode="w", header=True, index=False)


def remaining_combos(fixed: dict, csv_path: Path = CSV_PATH) -> list[dict]:
    """Sweep combos whose full param_key is not yet in the CSV."""
    done = load_completed_keys(csv_path)
    out = []
    for combo in parameter_grid():
        full = {**combo, **fixed}
        if make_param_key(full) not in done:
            out.append(full)
    return out


# ─────────────────────────────────────────────────────────────
# One configuration
# ─────────────────────────────────────────────────────────────
def safe_run_one(params: dict) -> list[dict]:
    """``run_one`` that never raises — for parallel batches. On an unexpected
    failure it returns an empty list, so the config is simply left un-done and
    retried on the next run rather than crashing the whole batch. (Top-level so
    it is picklable by the loky backend.)"""
    try:
        return run_one(params)
    except Exception:
        return []


def run_one(params: dict) -> list[dict]:
    """Run one configuration; return one row per detector (top-5 → 5 rows)."""
    pkey = make_param_key(params)
    rng = np.random.RandomState(int(params["seed"]))
    center = np.array([params["center_x"], params["center_y"]])
    nb = int(params["n_bumps"])
    noise = float(params["noise_std"])
    grid_res = int(params["grid_res"])
    extract_q = float(params["extract_q"])
    n_select = int(params["n_select"])
    iters_retrain = int(params["iters_retrain"])
    es = bool(params.get("early_stopping", True))   # MLP early stopping (default on)

    # ---- setup ----
    X_all = P.sample_inputs(
        int(params["n_pool_total"]), rng,
        shift_strength=float(params["shift_strength"]),
        shift_center=(float(params["shift_center_x"]), float(params["shift_center_y"])),
        shift_spread=float(params["shift_spread"]),
    )
    y_all = P.label(X_all, rng, n_bumps=nb, noise_std=noise)
    r = float(params["radius"])
    if r > 0:
        X_keep, y_keep, _, _ = P.induce_weakspot(X_all, y_all, center, r)
    else:                       # radius 0 → no weakspot, train on the full dataset
        X_keep, y_keep = X_all, y_all
    k = min(int(params["n_train"]), len(X_keep))
    tr_idx = rng.choice(len(X_keep), size=k, replace=False)
    X_tr0, y_tr0 = X_keep[tr_idx], y_keep[tr_idx]

    X_eval = P.sample_inputs(int(params["n_eval"]), rng, shift_strength=0.0)
    y_eval = P.true_function(X_eval, n_bumps=nb)

    xx, yy, grid_flat = create_grid(resolution=grid_res)
    gt_mask = (induced_mask(xx, yy, "Circle (radius)", center, r, None)
               if r > 0 else np.zeros(xx.shape, dtype=bool))

    def _region(err):
        return P.region_error(X_eval, err, center, r) if r > 0 else (float("nan"), float("nan"))

    # ---- initial train + eval ----
    model0 = build_model(AVAILABLE_MODELS[params["model_name"]],
                         complexity=float(params["complexity"]),
                         iterations=int(params["iters_initial"]), early_stopping=es)
    model0.fit(X_tr0, y_tr0)
    _, err0, mi = P.evaluate(model0, X_eval, y_eval)
    ein0, eout0 = _region(err0)

    # ---- candidate pool (shared) ----
    X_cand = P.sample_inputs(int(params["n_candidate"]), rng, shift_strength=0.0)
    y_cand = P.label(X_cand, rng, n_bumps=nb, noise_std=noise)

    # ---- random baseline (detector-independent, computed once) ----
    n_sel = min(n_select, len(X_cand))
    rand_idx = rng.choice(len(X_cand), size=n_sel, replace=False)
    modelR = build_model(AVAILABLE_MODELS[params["model_name"]],
                         complexity=float(params["complexity"]),
                         iterations=iters_retrain, early_stopping=es)
    modelR.fit(X_cand[rand_idx], y_cand[rand_idx])
    _, errR, mb = P.evaluate(modelR, X_eval, y_eval)
    einR, eoutR = _region(errR)

    base_cols = {
        "base_rmse": mb["RMSE"], "base_mae": mb["MAE"], "base_r2": mb["R2"],
        "base_err_in": einR, "base_err_out": eoutR,
    }
    init_cols = {
        "init_rmse": mi["RMSE"], "init_mae": mi["MAE"], "init_r2": mi["R2"],
        "init_err_in": ein0, "init_err_out": eout0,
    }

    # ---- per-detector guided selection + retrain (top-5 detectors only) ----
    rows = []
    for name in SWEEP_DETECTORS:
        fn = DETECTION_METHODS[name]
        try:
            surf0 = P.normalize_surface(fn(X_eval, err0, grid_flat))
            ext = extract_weakspot(surf0, xx, yy, threshold_quantile=extract_q)
            iou = iou_ellipse_at_thresholds(surf0, xx, yy, gt_mask, (IOU_HEADLINE_Q,))
            if ext is None:
                c = grid_flat[int(np.argmax(surf0))]
                cx_d = cy_d = sx1 = sx2 = area = None
                dist = float("nan")
            else:
                cx_d, cy_d = ext["center"]
                sx1, sx2, area = ext["sigma_x1"], ext["sigma_x2"], ext["area_frac"]
                dist = (float(np.hypot(cx_d - center[0], cy_d - center[1]))
                        if r > 0 else float("nan"))
                c = (cx_d, cy_d)

            sel_idx, _ = P.select_by_weakspot(
                X_cand, c, float(params["sel_sigma"]), n_select, rng,
                mode=params["sel_mode"],
                method=params.get("sel_method", LEGACY_SEL_METHOD),
                ext=ext, surf=surf0, mix_ratio=float(params.get("mix_ratio", 1.0)))
            model1 = build_model(AVAILABLE_MODELS[params["model_name"]],
                                 complexity=float(params["complexity"]),
                                 iterations=iters_retrain, early_stopping=es)
            model1.fit(X_cand[sel_idx], y_cand[sel_idx])
            _, err1, mg = P.evaluate(model1, X_eval, y_eval)
            eing, eoutg = _region(err1)

            row = {
                "param_key": pkey, "sweep_config": SWEEP_CONFIG_NAME,
                **params, "method": name,
                "n_selected": int(len(sel_idx)),
                # weakspot-ID metrics (pre-retrain)
                "distance": dist, f"iou_q{int(IOU_HEADLINE_Q*100)}":
                    float(iou.get(IOU_HEADLINE_Q, float("nan"))),
                "centroid_x1": cx_d, "centroid_x2": cy_d,
                "sigma_x1": sx1, "sigma_x2": sx2, "area_frac": area,
                **init_cols,
                # guided outcome
                "guided_rmse": mg["RMSE"], "guided_mae": mg["MAE"], "guided_r2": mg["R2"],
                "guided_err_in": eing, "guided_err_out": eoutg,
                **base_cols,
                # head-to-head deltas
                "d_mae_guided": mg["MAE"] - mi["MAE"],
                "d_mae_base": mb["MAE"] - mi["MAE"],
                "gap_mae_guided_minus_base": mg["MAE"] - mb["MAE"],
                "d_errin_guided": eing - ein0,
                "d_errin_base": einR - ein0,
            }
        except Exception as e:  # keep the sweep going; record the failure
            row = {"param_key": pkey, "sweep_config": SWEEP_CONFIG_NAME,
                   **params, "method": name, "error": str(e)}
        rows.append(row)

    return rows
