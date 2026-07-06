"""
Data-Selective Training experiment.

Directs supervision to the region where a model is weakest. One press of
**Run Pipeline** executes the seven stages end to end:

    1. Setup        — landscape, noise, distribution shift, induced weakspot
    2. Training     — train a model (variable time / #points)
    3. Evaluation   — score it on a uniform landscape sample
    4. Weakspot ID  — locate where it fails (same detectors as the Weakspot page)
    5. Data Selection — pick new labelled points via a Gaussian kernel on the weakspot
    6. Retraining   — retrain from scratch on the new points only (variable time)
    7. Re-Evaluation — score again; the Overview tab tabulates before vs after

The parameter *sweep* over these settings is a separate, later step.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from scripts.weakspot.models import AVAILABLE_MODELS, build_model
from scripts.weakspot.detection import DETECTION_METHODS, create_grid
from scripts.weakspot.extraction import (
    extract_weakspot, induced_mask, iou_ellipse_at_thresholds,
)
from scripts.weakspot.plotting import (
    plot_data_overview, plot_ground_truth_error, plot_ground_truth_error_3d,
    plot_comparison,
)
from scripts.dataselect import pipeline as P
from scripts.dataselect.plots import (
    plot_landscape, plot_selection, plot_training_points,
)

st.set_page_config(page_title="Data-Selective Training", layout="wide")
st.title("🎯 Data-Selective Training via Weakspot Identification")
st.markdown(
    "Train a model, find where it is weakest, feed it new data **where it knows "
    "the least**, and retrain. Compare against the model before selection."
)

IOU_HEADLINE_Q = 0.90


# ─────────────────────────────────────────────────────────────
# SIDEBAR — all pipeline parameters
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Pipeline Setup")
seed = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)

st.sidebar.subheader("1. Dataset & Landscape")
n_bumps = st.sidebar.slider("Function complexity (n_bumps)", 1, 10, 5, 1,
                            help="Number of Gaussian bumps in the ground-truth landscape.")
noise_std = st.sidebar.slider("Label noise σ (training only)", 0.0, 0.5, 0.10, 0.01,
                              help="Gaussian noise added to TRAINING labels. The "
                                   "evaluation set uses the noiseless ground truth.")
n_pool_total = st.sidebar.slider("Available dataset size", 500, 4000, 1500, 100,
                                 help="Total labelled points available before the weakspot is induced.")

st.sidebar.subheader("2. Distribution Shift")
shift_strength = st.sidebar.slider("Shift strength", 0.0, 1.0, 0.0, 0.05,
                                   help="Fraction of training inputs drawn from a "
                                        "Gaussian-concentrated cluster (covariate shift). "
                                        "0 = uniform. Evaluation stays uniform.")
scx = st.sidebar.slider("Shift centre x₁", 0.0, 1.0, 0.30, 0.05,
                        disabled=(shift_strength == 0.0))
scy = st.sidebar.slider("Shift centre x₂", 0.0, 1.0, 0.30, 0.05,
                        disabled=(shift_strength == 0.0))
shift_spread = st.sidebar.slider("Shift spread σ", 0.05, 0.40, 0.15, 0.01,
                                 disabled=(shift_strength == 0.0))

st.sidebar.subheader("3. Induced Weakspot (leave out data)")
induce_ws = st.sidebar.checkbox(
    "Induce a weakspot (leave out a data gap)", value=True,
    help="Off = train the initial model on the FULL dataset with no gap. The "
         "model's weakness then comes only from noise, distribution shift, "
         "function complexity and finite data — not an artificial hole.")
center_x = st.sidebar.slider("Weakspot centre x₁", 0.1, 0.9, 0.5, 0.01,
                             disabled=not induce_ws)
center_y = st.sidebar.slider("Weakspot centre x₂", 0.1, 0.9, 0.5, 0.01,
                             disabled=not induce_ws)
radius = st.sidebar.slider("Exclusion radius", 0.02, 0.30, 0.12, 0.01,
                           disabled=not induce_ws)

st.sidebar.subheader("4. Model & Training")
_mk = list(AVAILABLE_MODELS.keys())
model_name = st.sidebar.selectbox("Algorithm", _mk,
                                  index=_mk.index("MLP Neural Network") if "MLP Neural Network" in _mk else 0)
complexity = st.sidebar.slider("Model complexity", 0.0, 1.0, 0.5, 0.05)
iters_initial = st.sidebar.slider("Initial training time (iterations)", 20, 500, 100, 10)
n_train = st.sidebar.slider("Training points to use", 100, 3000, 800, 50,
                            help="How many of the (kept) points to train on.")

st.sidebar.subheader("5. Evaluation")
n_eval = st.sidebar.slider("Evaluation points on landscape", 200, 3000, 1000, 100,
                           help="Uniform sample from the landscape (deployment distribution).")

st.sidebar.subheader("6. Weakspot Identification")
_dm = list(DETECTION_METHODS.keys())
detect_name = st.sidebar.selectbox(
    "Detection method", _dm,
    index=_dm.index("EVT × GPR (geometric)") if "EVT × GPR (geometric)" in _dm else 0,
    help="Same detectors as the Weakspot Experiment page.",
)
grid_res = st.sidebar.slider("Grid resolution", 20, 60, 35, 5)
extract_q = st.sidebar.slider("Extraction threshold quantile", 0.50, 0.99, 0.85, 0.01)

st.sidebar.subheader("7. Data Selection")
sel_method = st.sidebar.selectbox(
    "Selection method", list(P.SEL_METHODS), index=0,
    help="How candidate points are weighted around the model's weakness:\n\n"
         "• **Weakpoint distance** — isotropic Gaussian around the single detected "
         "centre (the original strategy).\n"
         "• **Weight by landscape** — sample ∝ the whole detected error/weakspot "
         "surface, so *every* weak region is covered (multi-modal, needs no single "
         "centre and is robust to a mis-located one).\n"
         "• **Shape aware** — Gaussian shaped to the detected weakspot *ellipse* "
         "(anisotropic; σ below sets the overall width).")
sel_mode = st.sidebar.radio("Selection rule", ["Sample ∝ weight", "Top-weighted"])
sel_sigma = st.sidebar.slider("Selection Gaussian σ (variance)", 0.02, 0.80, 0.35, 0.01,
                              help="Width of the Gaussian selection kernel around the "
                                   "detected weakspot. Small = tightly focused; above "
                                   "~0.5 the kernel is nearly flat over [0,1]², so "
                                   "selection approaches the uniform random baseline.")
n_select = st.sidebar.slider("Points to add", 20, 1000, 500, 20)
n_candidate = st.sidebar.slider("Candidate pool size", 200, 4000, 1500, 100,
                                help="Fresh labelled points available to select from.")

st.sidebar.subheader("8. Retraining")
iters_retrain = st.sidebar.slider("Retraining time (iterations)", 20, 800, 150, 10)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Pipeline", type="primary", width='stretch')


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _detect_surface(X_eval, err, grid_flat):
    """Run the chosen detector and min-max normalise its surface."""
    fn = DETECTION_METHODS[detect_name]
    surf = fn(X_eval, err, grid_flat)
    return P.normalize_surface(surf)


def _weakspot_metrics(surface, xx, yy, gt_mask, induced_center, has_ws=True):
    """Extract a parametric weakspot and (if a gap was induced) score it against it.

    The detected centre is always returned so data selection can proceed; the
    ground-truth distance / IoU are NaN when no weakspot was induced.
    """
    ext = extract_weakspot(surface, xx, yy, threshold_quantile=extract_q)
    if ext is None:
        # fall back to the surface argmax so selection still has a centre
        grid_flat = np.column_stack([xx.ravel(), yy.ravel()])
        c = grid_flat[int(np.argmax(surface))]
        return None, (float(c[0]), float(c[1])), float("nan"), float("nan")
    cx, cy = ext["center"]
    if not has_ws:
        return ext, (cx, cy), float("nan"), float("nan")
    dist = float(np.hypot(cx - induced_center[0], cy - induced_center[1]))
    iou = iou_ellipse_at_thresholds(surface, xx, yy, gt_mask, (IOU_HEADLINE_Q,))
    return ext, (cx, cy), dist, float(iou.get(IOU_HEADLINE_Q, float("nan")))


# ─────────────────────────────────────────────────────────────
# RUN THE PIPELINE
# ─────────────────────────────────────────────────────────────
if run_btn or st.session_state.pop("_trigger_run", False):
    rng = np.random.RandomState(int(seed))
    center = np.array([center_x, center_y])

    # ---- 1. SETUP -------------------------------------------------------
    with st.spinner("Setup — sampling landscape & inducing the weakspot…"):
        X_all = P.sample_inputs(n_pool_total, rng, shift_strength=shift_strength,
                                shift_center=(scx, scy), shift_spread=shift_spread)
        y_all = P.label(X_all, rng, n_bumps=n_bumps, noise_std=noise_std)
        if induce_ws:
            X_keep, y_keep, X_excl, y_excl = P.induce_weakspot(X_all, y_all, center, radius)
            r_eff = float(radius)
        else:                                   # no gap — use the full dataset
            X_keep, y_keep = X_all, y_all
            X_excl, y_excl = X_all[:0], y_all[:0]
            r_eff = 0.0

        # training subsample (respect the requested #points)
        k = min(int(n_train), len(X_keep))
        tr_idx = rng.choice(len(X_keep), size=k, replace=False)
        X_tr0, y_tr0 = X_keep[tr_idx], y_keep[tr_idx]

        # evaluation set — uniform (deployment), noiseless labels
        X_eval = P.sample_inputs(n_eval, rng, shift_strength=0.0)
        y_eval = P.true_function(X_eval, n_bumps=n_bumps)

        xx, yy, grid_flat = create_grid(resolution=int(grid_res))
        gt_mask = (induced_mask(xx, yy, "Circle (radius)", center, r_eff, None)
                   if r_eff > 0 else np.zeros(xx.shape, dtype=bool))
        true_grid = P.landscape_surface(grid_flat, n_bumps=n_bumps)

    # ---- 2. TRAINING (initial) -----------------------------------------
    with st.spinner(f"Training {model_name} ({iters_initial} iters)…"):
        model0 = build_model(AVAILABLE_MODELS[model_name],
                             complexity=complexity, iterations=int(iters_initial))
        model0.fit(X_tr0, y_tr0)

    # ---- 3. EVALUATION (initial) ---------------------------------------
    _, err0, metrics0 = P.evaluate(model0, X_eval, y_eval)
    ein0, eout0 = (P.region_error(X_eval, err0, center, r_eff)
                   if r_eff > 0 else (float("nan"), float("nan")))

    # ---- 4. WEAKSPOT ID (initial) --------------------------------------
    with st.spinner(f"Identifying weakspot ({detect_name})…"):
        surf0 = _detect_surface(X_eval, err0, grid_flat)
        ext0, sel_center, dist0, iou0 = _weakspot_metrics(
            surf0, xx, yy, gt_mask, center, has_ws=induce_ws)

    # ---- 5. DATA SELECTION ---------------------------------------------
    with st.spinner("Selecting new data around the weakspot…"):
        X_cand = P.sample_inputs(n_candidate, rng, shift_strength=0.0)
        y_cand = P.label(X_cand, rng, n_bumps=n_bumps, noise_std=noise_std)
        sel_idx, sel_w = P.select_by_weakspot(
            X_cand, sel_center, sel_sigma, int(n_select), rng, mode=sel_mode,
            method=sel_method, ext=ext0, surf=surf0)
        X_sel, y_sel = X_cand[sel_idx], y_cand[sel_idx]

    # ---- 6. RETRAINING (on the newly selected points ONLY) --------------
    with st.spinner(f"Retraining ({iters_retrain} iters) on new points only…"):
        if len(X_sel):
            X_tr1, y_tr1 = X_sel, y_sel
        else:                       # no points selected → fall back to original
            X_tr1, y_tr1 = X_tr0, y_tr0
        model1 = build_model(AVAILABLE_MODELS[model_name],
                             complexity=complexity, iterations=int(iters_retrain))
        model1.fit(X_tr1, y_tr1)

    # ---- 7. RE-EVALUATION ----------------------------------------------
    _, err1, metrics1 = P.evaluate(model1, X_eval, y_eval)
    ein1, eout1 = (P.region_error(X_eval, err1, center, r_eff)
                   if r_eff > 0 else (float("nan"), float("nan")))
    surf1 = _detect_surface(X_eval, err1, grid_flat)
    ext1, sel_center1, dist1, iou1 = _weakspot_metrics(
        surf1, xx, yy, gt_mask, center, has_ws=induce_ws)

    # ---- BASELINE: random selection (same #points, no weakspot focus) ---
    with st.spinner("Baseline — random selection + retrain (no weakspot focus)…"):
        n_sel_actual = len(X_sel)
        rand_idx = (rng.choice(len(X_cand), size=n_sel_actual, replace=False)
                    if n_sel_actual > 0 else np.array([], dtype=int))
        X_rand, y_rand = X_cand[rand_idx], y_cand[rand_idx]
        if len(X_rand):             # train on the new random points only
            X_trR, y_trR = X_rand, y_rand
        else:
            X_trR, y_trR = X_tr0, y_tr0
        modelR = build_model(AVAILABLE_MODELS[model_name],
                             complexity=complexity, iterations=int(iters_retrain))
        modelR.fit(X_trR, y_trR)
    _, errR, metricsR = P.evaluate(modelR, X_eval, y_eval)
    einR, eoutR = (P.region_error(X_eval, errR, center, r_eff)
                   if r_eff > 0 else (float("nan"), float("nan")))
    surfR = _detect_surface(X_eval, errR, grid_flat)
    extR, _cR, distR, iouR = _weakspot_metrics(
        surfR, xx, yy, gt_mask, center, has_ws=induce_ws)

    st.session_state["dst"] = dict(
        xx=xx, yy=yy, grid_flat=grid_flat, gt_mask=gt_mask, true_grid=true_grid,
        center=center, radius=r_eff, has_ws=induce_ws,
        X_all=X_all, y_all=y_all, X_keep=X_keep, X_excl=X_excl,
        X_tr0=X_tr0, y_tr0=y_tr0, X_eval=X_eval, y_eval=y_eval,
        err0=err0, metrics0=metrics0, ein0=ein0, eout0=eout0,
        surf0=surf0, ext0=ext0, sel_center=sel_center, dist0=dist0, iou0=iou0,
        X_cand=X_cand, sel_w=sel_w, X_sel=X_sel, y_sel=y_sel, sel_idx=sel_idx,
        X_tr1=X_tr1, err1=err1, metrics1=metrics1, ein1=ein1, eout1=eout1,
        surf1=surf1, ext1=ext1, dist1=dist1, iou1=iou1,
        model_name=model_name, detect_name=detect_name,
        iters_initial=iters_initial, iters_retrain=iters_retrain,
        n_select=len(X_sel), sel_sigma=sel_sigma, sel_mode=sel_mode,
        sel_method=sel_method,
        # random baseline
        X_rand=X_rand, y_rand=y_rand, X_trR=X_trR, errR=errR, metricsR=metricsR,
        einR=einR, eoutR=eoutR, surfR=surfR, extR=extR, distR=distR, iouR=iouR,
    )


# ─────────────────────────────────────────────────────────────
# STEP FLOW STRIP
# ─────────────────────────────────────────────────────────────
st.markdown(
    "**Pipeline:** `Setup` → `Training` → `Evaluation` → `Weakspot ID` → "
    "`Data Selection` → `Retraining` → `Re-Evaluation` → `Overview`"
)

# ─────────────────────────────────────────────────────────────
# RUN OPTIONS — single experiment vs full parameter sweep
# ─────────────────────────────────────────────────────────────
import time
from scripts.dataselect import sweep as SW

st.header("🧪 Run Options")
fixed_params = dict(
    model_name=model_name, complexity=float(complexity),
    center_x=float(center_x), center_y=float(center_y),
    n_eval=int(n_eval), grid_res=int(grid_res), extract_q=float(extract_q),
    sel_mode=sel_mode, shift_center_x=float(scx), shift_center_y=float(scy),
    shift_spread=float(shift_spread), seed=int(seed),
)  # radius is swept (incl. 0.0 = no weakspot), so it is NOT fixed here
all_combos = SW.parameter_grid()
total_runs = len(all_combos)
remaining = SW.remaining_combos(fixed_params, SW.CSV_PATH)
n_done = total_runs - len(remaining)
done_total = len(SW.load_completed_keys(SW.CSV_PATH))
est_total_min = total_runs * SW.SECS_PER_RUN / 60
est_remain_min = len(remaining) * SW.SECS_PER_RUN / 60

col_l, col_r = st.columns(2, gap="large")
with col_l:
    st.subheader("Single Run")
    st.caption("Runs one experiment with the current sidebar settings and "
               "updates the tabs below (same as the sidebar button).")
    if st.button("▶ Run single experiment", type="primary",
                 width='stretch', key="run_single_bottom"):
        st.session_state["_trigger_run"] = True
        st.rerun()

with col_r:
    st.subheader("Parameter Sweep")
    st.caption(
        f"Runs the full grid ({total_runs} configs × 17 detectors), appending each "
        f"config's rows to CSV. **Resumable** — completed configs are skipped."
    )
    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("Total configs", total_runs)
    ci2.metric("Done (this config)", n_done)
    ci3.metric("Remaining", len(remaining))
    import os
    _max_workers = os.cpu_count() or 4
    run_parallel = st.checkbox(
        "⚡ Run in parallel (use multiple CPU cores)", value=False,
        help="Runs configurations across CPU cores with joblib. Much faster; uses "
             "more CPU and RAM. The main process still does all CSV writes, so the "
             "resume/CSV integrity is unaffected.",
    )
    n_workers = st.slider(
        "Worker processes", 1, _max_workers, max(1, _max_workers // 2),
        disabled=not run_parallel,
        help=f"This machine has {_max_workers} logical cores. Each worker holds a "
             "full copy of the libraries plus one config's data, so memory scales "
             "with worker count — if workers get killed (out-of-memory), lower this. "
             "The run auto-reduces workers on a crash rather than failing.",
    ) if run_parallel else 1
    run_sweep_btn = st.button(
        "▶ Run Sweep" if remaining else "✓ Sweep complete for these fixed values",
        type="primary", disabled=not remaining, width='stretch',
        key="run_sweep_dst",
    )

st.markdown(
    f"""
**Swept axes:** {', '.join(f'`{k}`×{len(v)}' for k, v in SW.SWEEP_GRID.items())}.
`radius`=0.0 configs induce **no weakspot** (full-dataset baseline).
**Detectors:** all 17 (one CSV row each).
**Held fixed at sidebar values:** model=`{model_name}`, complexity=`{complexity}`,
weakspot centre=({center_x}, {center_y}), n_eval=`{n_eval}`,
grid_res=`{grid_res}`, extract_q=`{extract_q}`, selection rule=`{sel_mode}`, seed=`{seed}`.

**Estimated time** (~{SW.SECS_PER_RUN:.0f} s/config): full sweep ≈ **{est_total_min:.0f} min**
· remaining ≈ **{est_remain_min:.0f} min**.  **CSV:** `{SW.CSV_PATH}` ·
rows completed across all fixed-value sets: **{done_total}**.
"""
)
if run_parallel and n_workers > 1 and remaining:
    st.caption(
        f"⚡ With **{n_workers} workers**, the remaining ≈ {est_remain_min:.0f} min "
        f"should drop to roughly **{est_remain_min / n_workers:.0f} min** "
        f"(near-linear speed-up, minus pool overhead)."
    )

if run_sweep_btn and remaining:
    n_total = len(remaining)
    progress = st.progress(0.0, text=f"0 / {n_total} — starting…")
    t0 = time.time()

    def _tick(done, elapsed):
        eta_min = (elapsed / max(done, 1)) * (n_total - done) / 60
        progress.progress(done / n_total,
                          text=f"{done}/{n_total} · "
                               f"{'⚡ ' + str(n_workers) + ' workers' if run_parallel else 'sequential'} · "
                               f"elapsed {elapsed:.0f}s · ETA {eta_min:.1f} min")

    if run_parallel and n_workers > 1:
        # Workers only COMPUTE (run_one is pure); the main process does every CSV
        # write, so there is no concurrent-write risk. If the OS kills a worker
        # (usually out-of-memory) the loky pool dies, so we catch that, halve the
        # worker count, and continue from where we were — no config is skipped
        # (the failed batch was not appended, so it is simply retried) and all
        # completed configs are already on disk.
        from joblib import Parallel, delayed, parallel_backend
        i, w, done = 0, n_workers, 0
        while i < n_total:
            batch = max(w * 4, 1)
            try:
                with parallel_backend("loky", inner_max_num_threads=1):
                    with Parallel(n_jobs=w) as parallel:
                        while i < n_total:
                            chunk = remaining[i:i + batch]
                            results = parallel(delayed(SW.safe_run_one)(p) for p in chunk)
                            for rows in results:
                                if rows:
                                    SW.append_rows(SW.CSV_PATH, rows)
                            i += len(chunk); done += len(chunk)
                            _tick(done, time.time() - t0)
                break  # finished all configs
            except Exception as e:
                new_w = max(1, w // 2)
                st.warning(
                    f"Parallel workers were terminated ({type(e).__name__}, usually "
                    f"out-of-memory). Reducing workers **{w} → {new_w}** and "
                    f"continuing — completed configs are saved and resumable."
                )
                w = new_w
                if w == 1:
                    for params in remaining[i:]:
                        rows = SW.safe_run_one(params)
                        if rows:
                            SW.append_rows(SW.CSV_PATH, rows)
                        i += 1; done += 1
                        _tick(done, time.time() - t0)
                    break
    else:
        for i, params in enumerate(remaining):
            try:
                SW.append_rows(SW.CSV_PATH, SW.run_one(params))
            except Exception as e:
                st.warning(f"Config {i+1} failed: {e}")
            _tick(i + 1, time.time() - t0)

    progress.empty()
    st.success(
        f"Sweep complete in {(time.time()-t0)/60:.1f} min "
        f"({'parallel, ' + str(n_workers) + ' workers' if run_parallel else 'sequential'}). "
        f"Rows appended to `{SW.CSV_PATH}`. Open "
        f"**Data-Selective — Visualise Results** to explore."
    )

if SW.CSV_PATH.exists():
    with st.expander("Preview last 15 CSV rows"):
        try:
            st.dataframe(pd.read_csv(SW.CSV_PATH, low_memory=False).tail(15),
                         width='stretch', hide_index=True)
        except Exception as e:
            st.warning(f"Could not read CSV: {e}")

st.markdown("---")
if "dst" not in st.session_state:
    st.info("Press **▶ Run single experiment** above (or the sidebar button) to "
            "see the step-by-step results.")
    st.stop()

R = st.session_state["dst"]
xx, yy = R["xx"], R["yy"]
center, radius = R["center"], R["radius"]

tabs = st.tabs([
    "① Setup", "② Training", "③ Evaluation", "④ Weakspot ID",
    "⑤ Data Selection", "⑥ Retraining", "⑦ Re-Evaluation", "🧾 Overview",
])

# ── ① SETUP ──────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Step 1 — Setup")
    st.caption(
        "The ground-truth landscape is a sum of Gaussian bumps. A weakspot is "
        "induced by removing every training point inside the red circle. Optional "
        "distribution shift concentrates training inputs elsewhere."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_landscape(xx, yy, R["true_grid"], center, radius),
            width='stretch')
    with c2:
        st.plotly_chart(
            plot_data_overview(R["X_keep"], P.true_function(R["X_keep"]),
                               R["X_excl"], center, radius,
                               "x₁", "x₂", "f(x₁,x₂)", excl_mode="Circle (radius)"),
            width='stretch')
    if R.get("has_ws", True):
        st.info(
            f"Available: **{len(R['X_all'])}** · kept (outside gap): **{len(R['X_keep'])}** · "
            f"excluded (the gap): **{len(R['X_excl'])}** · evaluation: **{len(R['X_eval'])}** (uniform)."
        )
    else:
        st.info(
            f"**No weakspot induced** — the initial model is trained on the full "
            f"dataset. Kept/available: **{len(R['X_keep'])}** · evaluation: "
            f"**{len(R['X_eval'])}** (uniform). Model weakness comes from noise, "
            f"distribution shift, complexity and finite data."
        )

# ── ② TRAINING ───────────────────────────────────────────────
with tabs[1]:
    st.subheader("Step 2 — Training (initial model)")
    st.caption(
        f"Trained **{R['model_name']}** for **{R['iters_initial']}** iterations on "
        f"**{len(R['X_tr0'])}** points, none of them inside the induced weakspot."
    )
    st.plotly_chart(
        plot_training_points(R["X_tr0"], R["y_tr0"], None, center, radius,
                             title="Initial training set (gap left empty)"),
        width='stretch')

# ── ③ EVALUATION ─────────────────────────────────────────────
with tabs[2]:
    st.subheader("Step 3 — Evaluation (before selection)")
    st.caption(
        "Absolute error on the uniform evaluation set, measured against the "
        "noiseless ground truth. Error should pile up inside the induced gap."
    )
    m = R["metrics0"]
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{m['RMSE']:.4f}")
    c2.metric("MAE", f"{m['MAE']:.4f}")
    c3.metric("R²", f"{m['R2']:.4f}")
    st.plotly_chart(plot_ground_truth_error(R["X_eval"], R["err0"], center, radius),
                    width='stretch')
    st.plotly_chart(plot_ground_truth_error_3d(R["X_eval"], R["err0"], center, radius),
                    width='stretch', key="eval0_3d")
    if R.get("has_ws", True) and R["eout0"] == R["eout0"] and R["eout0"] > 0:
        st.success(
            f"Mean error inside weakspot: **{R['ein0']:.4f}** vs outside "
            f"**{R['eout0']:.4f}**  →  ratio **{R['ein0']/R['eout0']:.2f}×**.")
    else:
        st.info("No weakspot induced — the initial model was trained on the full "
                "dataset (in/out-of-weakspot error not applicable).")

# ── ④ WEAKSPOT ID ────────────────────────────────────────────
with tabs[3]:
    st.subheader("Step 4 — Weakspot Identification")
    st.caption(
        f"Detector: **{R['detect_name']}**. Left = model error surface; right = "
        f"detected surface with the extracted 2σ ellipse (q={extract_q:.2f})."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Centroid distance ↓",
              f"{R['dist0']:.4f}" if R["dist0"] == R["dist0"] else "n/a")
    c2.metric(f"IoU @ q={IOU_HEADLINE_Q:.2f} ↑",
              f"{R['iou0']:.4f}" if R["iou0"] == R["iou0"] else "n/a")
    c3.metric("Detected centre", f"({R['sel_center'][0]:.2f}, {R['sel_center'][1]:.2f})")
    if not R.get("has_ws", True):
        st.caption("No weakspot induced → no ground-truth region, so distance / IoU "
                   "are not applicable. The detector still locates the model's "
                   "weakest region, which drives selection.")
    st.plotly_chart(
        plot_comparison(xx, yy, R["surf0"], R["surf0"], R["detect_name"],
                        center, radius, R["X_eval"], R["err0"],
                        extraction=R["ext0"]),
        width='stretch', key="wsid0")

# ── ⑤ DATA SELECTION ─────────────────────────────────────────
with tabs[4]:
    st.subheader("Step 5 — Weakspot-Guided Data Selection")
    st.caption(
        f"Both strategies pick **{R['n_select']} brand-new points** from the **same "
        f"candidate pool** ({len(R['X_cand'])} fresh labelled points, none in the "
        f"original training set); these become the **new training set** for retraining "
        f"(the original data is not reused). Left: weakspot-guided — strategy "
        f"**{R.get('sel_method', 'Weakpoint distance')}** "
        f"(σ = **{R['sel_sigma']:.2f}**, rule = *{R['sel_mode']}*) focused on the "
        f"detected weakspot. Right: random baseline — uniform draws, ignoring the weakspot."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weakspot-guided selection**")
        st.plotly_chart(
            plot_selection(R["X_cand"], R["sel_w"], R["X_sel"], R["sel_center"],
                           R["sel_sigma"], center, radius,
                           title="Weakspot-guided: new points near the weakspot"),
            width='stretch')
    with c2:
        st.markdown("**Random baseline selection**")
        st.plotly_chart(
            plot_selection(R["X_cand"], None, R["X_rand"], None, None,
                           center, radius, show_kernel=False,
                           title="Random baseline: new points, uniform"),
            width='stretch', key="rand_sel")

# ── ⑥ RETRAINING ─────────────────────────────────────────────
with tabs[5]:
    st.subheader("Step 6 — Retraining (new points only)")
    st.caption(
        f"Each model is retrained **from scratch on only its {R['n_select']} newly "
        f"selected points** — the original training set is **not** reused. "
        f"(**{R['model_name']}**, {R['iters_retrain']} iterations.) The dashed circle "
        f"marks the induced weakspot; note how tightly the weakspot-guided training "
        f"set clusters there compared with the spread-out random baseline."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weakspot-guided** — trained on new points only")
        st.plotly_chart(
            plot_training_points(R["X_sel"], R["y_sel"], None, center, radius,
                                 title="Weakspot-guided new points (training set)"),
            width='stretch', key="retrain_ws")
    with c2:
        st.markdown("**Random baseline** — trained on new points only")
        st.plotly_chart(
            plot_training_points(R["X_rand"], R["y_rand"], None, center, radius,
                                 title="Random new points (training set)"),
            width='stretch', key="retrain_rand")

# ── ⑦ RE-EVALUATION ──────────────────────────────────────────
with tabs[6]:
    st.subheader("Step 7 — Re-Evaluation (after selection)")
    m0, m1, mR = R["metrics0"], R["metrics1"], R["metricsR"]
    st.markdown("**Weakspot-guided** (Δ vs initial):")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{m1['RMSE']:.4f}", f"{m1['RMSE']-m0['RMSE']:+.4f}",
              delta_color="inverse")
    c2.metric("MAE", f"{m1['MAE']:.4f}", f"{m1['MAE']-m0['MAE']:+.4f}",
              delta_color="inverse")
    c3.metric("R²", f"{m1['R2']:.4f}", f"{m1['R2']-m0['R2']:+.4f}")
    st.markdown("**Random baseline** (Δ vs initial):")
    b1, b2, b3 = st.columns(3)
    b1.metric("RMSE", f"{mR['RMSE']:.4f}", f"{mR['RMSE']-m0['RMSE']:+.4f}",
              delta_color="inverse")
    b2.metric("MAE", f"{mR['MAE']:.4f}", f"{mR['MAE']-m0['MAE']:+.4f}",
              delta_color="inverse")
    b3.metric("R²", f"{mR['R2']:.4f}", f"{mR['R2']-m0['R2']:+.4f}")
    st.plotly_chart(plot_ground_truth_error(R["X_eval"], R["err1"], center, radius),
                    width='stretch')
    st.plotly_chart(plot_ground_truth_error_3d(R["X_eval"], R["err1"], center, radius),
                    width='stretch', key="eval1_3d")
    if R.get("has_ws", True) and R["ein1"] == R["ein1"]:
        st.success(
            f"Mean error inside weakspot: **{R['ein1']:.4f}** "
            f"(was {R['ein0']:.4f}, Δ {R['ein1']-R['ein0']:+.4f})."
        )
    else:
        st.info("No weakspot induced — in-weakspot error not applicable.")

# ── 🧾 OVERVIEW ──────────────────────────────────────────────
with tabs[7]:
    st.subheader("Overview — Before vs After Weakspot-Guided Selection")
    st.caption(
        "Model metrics on the uniform evaluation set and weakspot-identification "
        "metrics (same as the Weakspot Experiment page): centroid distance to the "
        "induced centre and ellipse-IoU at q=0.90. Lower error / distance is "
        "better; higher R² / IoU is better."
    )

    def _row(stage, mtr, ein, eout, dist, iou, ntr):
        return {
            "Stage": stage,
            "Train pts": ntr,
            "RMSE": round(mtr["RMSE"], 4),
            "MAE": round(mtr["MAE"], 4),
            "R²": round(mtr["R2"], 4),
            "Err in weakspot": round(ein, 4) if ein == ein else None,
            "Err outside": round(eout, 4) if eout == eout else None,
            "WS distance ↓": round(dist, 4) if dist == dist else None,
            f"WS IoU@{IOU_HEADLINE_Q:.2f} ↑": round(iou, 4) if iou == iou else None,
        }

    df = pd.DataFrame([
        _row("Before (initial)", R["metrics0"], R["ein0"], R["eout0"],
             R["dist0"], R["iou0"], len(R["X_tr0"])),
        _row("Weakspot-guided", R["metrics1"], R["ein1"], R["eout1"],
             R["dist1"], R["iou1"], len(R["X_tr1"])),
        _row("Random baseline", R["metricsR"], R["einR"], R["eoutR"],
             R["distR"], R["iouR"], len(R["X_trR"])),
    ])
    st.dataframe(df, width='stretch', hide_index=True)

    # Weakspot-guided vs random baseline — the head-to-head the baseline enables
    dmae_ws = R["metrics1"]["MAE"] - R["metrics0"]["MAE"]
    dmae_rand = R["metricsR"]["MAE"] - R["metrics0"]["MAE"]
    gap = R["metrics1"]["MAE"] - R["metricsR"]["MAE"]      # <0 → guided beats random
    dein_ws = R["ein1"] - R["ein0"]
    dein_rand = R["einR"] - R["ein0"]
    verdict = ("**beats** the random baseline ✅" if gap < 0
               else "does **not** beat the random baseline ⚠️")
    _fmt = lambda v: f"{v:+.4f}" if v == v else "n/a"
    ws_note = ("" if R.get("has_ws", True)
               else " _(no weakspot induced — in-weakspot figures are n/a)_")
    st.markdown(
        f"**Weakspot-guided vs random baseline** (each retrained from scratch on its "
        f"own {R['n_select']} new points, {R['iters_retrain']} iters, detector "
        f"*{R['detect_name']}*, selection σ {R['sel_sigma']:.2f}):{ws_note}\n\n"
        f"- Overall MAE Δ vs initial — guided **{dmae_ws:+.4f}**, random **{dmae_rand:+.4f}** "
        f"(guided − random = **{gap:+.4f}**)\n"
        f"- In-weakspot error Δ vs initial — guided **{_fmt(dein_ws)}**, random **{_fmt(dein_rand)}**\n\n"
        f"Weakspot-guided selection {verdict} for this configuration."
    )
    st.caption(
        "The full parameter sweep over model state, selection σ, detector and "
        "training time is the next step — this page runs a single configuration."
    )
