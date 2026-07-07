import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import numpy as np
import pandas as pd

from scripts.weakspot.datasets import (
    AVAILABLE_DATASETS, load_dataset, normalize,
    apply_exclusion_zone, apply_exclusion_knn, make_train_test_split
)
from scripts.weakspot.models import AVAILABLE_MODELS, build_model, train_and_evaluate
from scripts.weakspot.detection import DETECTION_METHODS, create_grid
from scripts.weakspot.extraction import extract_weakspot, induced_mask, iou_ellipse_at_thresholds
from scripts.weakspot.plotting import (
    plot_data_overview, plot_ground_truth_error, plot_ground_truth_error_3d,
    plot_all_methods, plot_comparison, plot_metrics_bar, plot_error_surface
)

st.set_page_config(page_title="Weakspot Experiment", layout="wide")
st.title("🔍 Weakspot Identification Experiment")
st.markdown(
    "Induce a weakspot by excluding a circular region from training, "
    "then compare methods that detect where the model performs poorly."
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Experimental Parameters
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Experimental Setup")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Dataset")
dataset_name = st.sidebar.selectbox(
    "Dataset",
    list(AVAILABLE_DATASETS.keys()),
    index=list(AVAILABLE_DATASETS.keys()).index("Synthetic Gaussian Bumps"),
)
n_samples = st.sidebar.slider("Sample size", 500, 2000, 1000, step=100)
n_bumps = st.sidebar.slider(
    "Function complexity (n_bumps)", 1, 10, 5, 1,
    help="Number of Gaussian bumps in the synthetic ground-truth function. "
         "Higher = more complex landscape. Ignored for non-bumps datasets."
)
noise_std = st.sidebar.slider(
    "Noise σ on the underlying function", 0.0, 0.50, 0.10, 0.01,
    help="Standard deviation of additive Gaussian noise on y. "
         "Ignored for California Housing (real data)."
)

st.sidebar.subheader("2. Induced Weakspot")
center_x     = st.sidebar.slider("Weakspot center  x₁ (normalized)", 0.1, 0.9, 0.5, 0.01)
center_y     = st.sidebar.slider("Weakspot center  x₂ (normalized)", 0.1, 0.9, 0.5, 0.01)
excl_mode    = st.sidebar.radio("Exclusion method", ["Circle (radius)", "k-Nearest Points"])
radius       = st.sidebar.slider("Exclusion radius", 0.02, 0.30, 0.12, 0.01,
                                  disabled=(excl_mode != "Circle (radius)"))
n_exclude    = st.sidebar.slider("Points to exclude (k)", 10, 300, 60, 10,
                                  disabled=(excl_mode != "k-Nearest Points"))

st.sidebar.subheader("3. Model")
_model_keys = list(AVAILABLE_MODELS.keys())
model_name = st.sidebar.selectbox(
    "Algorithm", _model_keys,
    index=_model_keys.index("MLP Neural Network") if "MLP Neural Network" in _model_keys else 0
)
complexity  = st.sidebar.slider("Model complexity", 0.0, 1.0, 0.5, 0.05,
                                 help="Controls capacity: depth, regularization, hidden size")
iterations  = st.sidebar.slider("Training iterations", 20, 500, 100, 10,
                                 help="n_estimators for tree models / max_iter for MLP")
test_size   = st.sidebar.slider("Test / train split ratio", 0.10, 0.40, 0.20, 0.05)

st.sidebar.subheader("4. Detection Methods")
selected_methods = st.sidebar.multiselect(
    "Methods to run",
    list(DETECTION_METHODS.keys()),
    default=list(DETECTION_METHODS.keys())
)

grid_res = st.sidebar.slider("Grid resolution", 20, 60, 35, 5,
                              help="Higher = finer heatmaps but slower")

st.sidebar.subheader("5. Weakspot Extraction (Stage 2)")
extract_quantile = st.sidebar.slider(
    "Threshold quantile", 0.50, 0.99, 0.85, 0.01,
    help="Pixels with surface ≥ this quantile are classified as weakspot. "
         "Higher = stricter (smaller, tighter blob). Lower = looser (larger, fuzzier)."
)

st.sidebar.subheader("6. Visualisation")
smooth_sigma = st.sidebar.slider(
    "Surface smoothness (σ)", 0.00, 0.20, 0.00, 0.01,
    help="Gaussian kernel width for smoothing the polygon error surface. "
         "0 = raw data, higher = smoother."
)

st.sidebar.markdown("---")
run_btn_sidebar = st.sidebar.button(
    "🚀 Run Experiment", type="primary", width='stretch', key="run_sidebar"
)
# A bottom-of-page button (defined later) sets this flag via session_state.
run_btn = run_btn_sidebar or st.session_state.pop("_trigger_single_run", False)

# ─────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────

show_welcome = not run_btn and "exp_results" not in st.session_state
if show_welcome:
    st.info("Configure the parameters in the sidebar and click **Run Experiment** "
            "— or scroll to the bottom for the parameter-sweep option.")
    st.markdown("""
### Recommended open-license 2D regression datasets

| Dataset | Input features | Target | Source | License |
|---|---|---|---|---|
| California Housing | Longitude, Latitude | Median House Value | sklearn / StatLib | CC0 |
| Synthetic Peaks | x₁, x₂ | f(x₁,x₂) | Generated | — |
| Fish Market | Length, Height | Weight | [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/fish-market) | CC0 |
| Forest Fires | Temperature, RH | Burned Area | [UCI](https://archive.ics.uci.edu/dataset/162/forest+fires) | CC BY 4.0 |
""")

if run_btn:
    center = np.array([center_x, center_y])
    dataset_key = AVAILABLE_DATASETS[dataset_name]

    with st.spinner("Loading data…"):
        X, y, x1_name, x2_name, y_name = load_dataset(
            dataset_key, n_samples=n_samples,
            n_bumps=int(n_bumps), noise_std=float(noise_std),
        )
        X_norm, scaler = normalize(X)

    with st.spinner("Applying exclusion zone…"):
        if excl_mode == "Circle (radius)":
            X_tr, y_tr, X_excl, y_excl, dists = apply_exclusion_zone(
                X_norm, y, center, radius
            )
            excl_desc = f"Circle  r={radius}  around ({center_x:.2f}, {center_y:.2f})"
        else:
            X_tr, y_tr, X_excl, y_excl, dists = apply_exclusion_knn(
                X_norm, y, center, n_exclude
            )
            excl_desc = f"{n_exclude} nearest points to ({center_x:.2f}, {center_y:.2f})"

        # Train on non-excluded points only (respecting test_size split)
        X_train, _, y_train, _ = make_train_test_split(
            X_tr, y_tr, excl_mask=None, test_size=test_size
        )
        # Evaluate on ALL points — including the excluded (weakspot) region
        X_test, y_test = X_norm, y

    with st.spinner(f"Training {model_name}…"):
        model = build_model(
            AVAILABLE_MODELS[model_name], complexity=complexity,
            iterations=iterations
        )
        y_pred, errors, metrics = train_and_evaluate(
            model, X_train, y_train, X_test, y_test
        )

    xx, yy, grid_flat = create_grid(resolution=grid_res)

    surfaces = {}
    n_methods = len(selected_methods)
    prog = st.progress(0, text="Running detection methods…")
    for i, name in enumerate(selected_methods):
        prog.progress((i + 1) / n_methods, text=f"Running: {name}")
        fn = DETECTION_METHODS[name]
        try:
            surf = fn(X_test, errors, grid_flat)
            # Normalize surface to [0,1]
            s_min, s_max = surf.min(), surf.max()
            if s_max > s_min:
                surf = (surf - s_min) / (s_max - s_min)
            surfaces[name] = surf
        except Exception as ex:
            st.warning(f"{name} failed: {ex}")
    prog.empty()

    # Ground truth error surface — linear interpolation on actual data, no model
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    interp_lin = LinearNDInterpolator(X_test, errors)
    gt_surface_raw = interp_lin(grid_flat)
    nan_mask = np.isnan(gt_surface_raw)
    if nan_mask.any():
        interp_near = NearestNDInterpolator(X_test, errors)
        gt_surface_raw[nan_mask] = interp_near(grid_flat[nan_mask])
    gt_min, gt_max = gt_surface_raw.min(), gt_surface_raw.max()
    gt_surface = (gt_surface_raw - gt_min) / (gt_max - gt_min + 1e-10)

    st.session_state["exp_results"] = {
        "X_norm": X_norm, "y": y, "X_excl": X_excl,
        "X_test": X_test, "y_test": y_test, "errors": errors,
        "metrics": metrics, "surfaces": surfaces, "gt_surface": gt_surface,
        "xx": xx, "yy": yy, "center": center, "radius": radius,
        "x1_name": x1_name, "x2_name": x2_name, "y_name": y_name,
        "model_name": model_name, "n_excl": len(X_excl),
        "n_train": len(X_train), "n_test": len(X_test),  # n_test = all points
        "n_total": len(X_norm), "excl_desc": excl_desc,
        "excl_mode": excl_mode,
    }

# ─────────────────────────────────────────────────────────────
# RESULTS — only rendered if an experiment has been run
# ─────────────────────────────────────────────────────────────
if "exp_results" in st.session_state:
    R = st.session_state["exp_results"]
    xx, yy = R["xx"], R["yy"]
    center, radius = R["center"], R["radius"]
    surfaces, gt_surface = R["surfaces"], R["gt_surface"]
    X_test, errors = R["X_test"], R["errors"]

    # Stage 2 — extract a parametric weakspot per surface
    extractions = {
        name: extract_weakspot(surf, xx, yy, threshold_quantile=extract_quantile)
        for name, surf in surfaces.items()
    }

    # Ellipse-vs-induced IoU at fixed thresholds (independent of sidebar quantile)
    IOU_QUANTILES = (0.80, 0.85, 0.90, 0.95)
    gt_mask = induced_mask(xx, yy, R["excl_mode"], center, radius, R.get("X_excl"))
    ious = {
        name: iou_ellipse_at_thresholds(surf, xx, yy, gt_mask, IOU_QUANTILES)
        for name, surf in surfaces.items()
    }

    # Summary strip
    st.info(
        f"**Exclusion:** {R['excl_desc']}  |  "
        f"**Total samples:** {R['n_total']}  →  "
        f"**Excluded:** {R['n_excl']}  |  "
        f"**Train:** {R['n_train']}  |  "
        f"**Test:** {R['n_test']}"
    )
    c1, c2, c3 = st.columns(3)
    for col, (k, v) in zip([c1, c2, c3], R["metrics"].items()):
        col.metric(k, v)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data & Exclusion Zone", "📉 Model Errors", "🔍 Detection & Comparison",
         "🧭 Ground-Truth Validation"]
    )

    with tab1:
        fig_data = plot_data_overview(
            R["X_norm"], R["y"], R["X_excl"],
            center, radius, R["x1_name"], R["x2_name"], R["y_name"],
            excl_mode=R["excl_mode"]
        )
        st.plotly_chart(fig_data, width='stretch')
        st.caption(
            f"Red ✕ = {R['n_excl']} excluded points. Dashed circle = induced weakspot zone."
        )

    with tab2:
        st.plotly_chart(plot_metrics_bar(R["metrics"]), width='stretch')
        fig_err = plot_ground_truth_error(X_test, errors, center, radius)
        st.plotly_chart(fig_err, width='stretch')
        fig_err_3d = plot_ground_truth_error_3d(X_test, errors, center, radius, sigma=smooth_sigma)
        st.plotly_chart(fig_err_3d, width='stretch', key="tab2_gt_3d")
        st.caption(
            "Errors shown directly at test point locations — no interpolation. "
            "3D surface built via Delaunay triangulation of actual test points."
        )

    with tab3:
        if not surfaces:
            st.warning("No methods produced results.")
        else:
            # ── All methods grid ───────────────────────────────────────────
            st.subheader("All Detection Methods")
            fig_all = plot_all_methods(
                xx, yy, surfaces, center, radius, X_test, errors,
                extractions=extractions,
            )
            st.plotly_chart(fig_all, width='stretch')
            st.caption(
                f"Surfaces normalized to [0,1]. Red dashed = induced zone. "
                f"Cyan = extracted 2σ ellipse at threshold quantile **{extract_quantile:.2f}**. "
                f"Cyan ✕ = weighted centroid of the largest above-threshold component."
            )

            st.markdown("---")

            # ── Side-by-side comparison ────────────────────────────────────
            st.subheader("Side-by-Side: Ground Truth vs Detected")
            best_name = st.selectbox("Compare method", list(surfaces.keys()), index=0)
            fig_cmp = plot_comparison(
                xx, yy, gt_surface, surfaces[best_name], best_name,
                center, radius, X_test, errors, sigma=smooth_sigma,
                extraction=extractions.get(best_name),
            )
            st.plotly_chart(fig_cmp, width='stretch')

            st.subheader("Ground Truth — 3D Polygon Surface")
            fig_gt_3d = plot_ground_truth_error_3d(
                X_test, errors, center, radius, sigma=smooth_sigma
            )
            st.plotly_chart(fig_gt_3d, width='stretch', key="tab3_gt_3d")
            st.caption(
                "Surface built by Delaunay triangulation of actual test points — no algorithm involved. "
                "Adjust **Surface smoothness (σ)** in the sidebar to smooth the mesh."
            )

            # ── Position, size, and ellipse-IoU table ──────────────────────
            st.subheader("Weakspot Position, Size & Ellipse-IoU")
            st.caption(
                f"**Centroid / σ x₁ / σ x₂ / Area / Distance** computed at the "
                f"sidebar threshold (**q={extract_quantile:.2f}**). "
                "**IoU @ q=…** = `|ellipse ∩ induced| / |ellipse ∪ induced|` — IoU "
                "between the **2σ ellipse** the algorithm extracts at that threshold "
                "and the **induced weakspot** (the circle / convex hull of the "
                "originally-excluded points). Range [0, 1]; 1 = perfect region match. "
                "The ellipse is rebuilt for each quantile, so different q values "
                "produce different ellipses and different IoUs."
            )

            pos_rows = []
            for name, surf in surfaces.items():
                ext = extractions.get(name)
                iou = ious.get(name, {})
                row = {"Method": name}

                if ext is None:
                    row.update({
                        "Centroid x₁": None, "Centroid x₂": None,
                        "σ x₁": None, "σ x₂": None, "Area": None, "Distance": None,
                    })
                else:
                    cx_d, cy_d = ext["center"]
                    dist = float(np.hypot(cx_d - center[0], cy_d - center[1]))
                    row.update({
                        "Centroid x₁": round(cx_d, 3),
                        "Centroid x₂": round(cy_d, 3),
                        "σ x₁": round(ext["sigma_x1"], 3),
                        "σ x₂": round(ext["sigma_x2"], 3),
                        "Area": round(ext["area_frac"], 3),
                        "Distance": round(dist, 4),
                    })

                for q in IOU_QUANTILES:
                    v = iou.get(q, float("nan"))
                    row[f"IoU q={q:.2f}"] = round(v, 4) if v == v else None
                pos_rows.append(row)

            df_pos = (
                pd.DataFrame(pos_rows)
                .sort_values("IoU q=0.90", ascending=False, na_position="last")
                .reset_index(drop=True)
            )
            st.dataframe(df_pos, width='stretch', hide_index=True)

            # Highlight best by ellipse-IoU at q=0.90
            valid = df_pos.dropna(subset=["IoU q=0.90"])
            if not valid.empty:
                best_row = valid.iloc[0]
                st.success(
                    f"**Best ellipse-IoU @ q=0.90:** {best_row['Method']} = "
                    f"**{best_row['IoU q=0.90']}** — "
                    f"centroid ({best_row['Centroid x₁']}, {best_row['Centroid x₂']}), "
                    f"size σ=({best_row['σ x₁']}, {best_row['σ x₂']}), "
                    f"distance {best_row['Distance']} from induced center "
                    f"({float(center[0]):.3f}, {float(center[1]):.3f})."
                )

    with tab4:
        st.subheader("Does the model actually fail where data was withheld?")
        st.markdown(
            "The experiment treats the **induced data gap** as the ground-truth "
            "weakspot. This check re-runs the **current sidebar configuration** at "
            "the nine sweep weakspot locations (3×3 centre grid, circular "
            "exclusion) and overlays each induced gap on the model's error "
            "surface. If the gap is a faithful ground truth, the error should be "
            "elevated inside the dashed circle at every location."
        )
        if st.button("🧭 Run ground-truth validation (trains 9 models)",
                     key="run_validation"):
            from scripts.weakspot.validation import (
                run_alignment_grid, render_alignment_figure,
                run_baseline_error_surface, render_alignment_delta_figure,
            )
            _common = dict(
                n_bumps=int(n_bumps), noise_std=float(noise_std),
                model_name=model_name, complexity=float(complexity),
                iterations=int(iterations), n_samples=int(n_samples),
                grid_res=int(grid_res),
            )
            with st.spinner("Training the model at 9 induced-weakspot locations…"):
                panels = run_alignment_grid(
                    dataset_key=AVAILABLE_DATASETS[dataset_name],
                    radius=float(radius), test_size=float(test_size), **_common,
                )
                fig = render_alignment_figure(panels, model_name=model_name,
                                              n_bumps=int(n_bumps), noise_std=float(noise_std))
            st.pyplot(fig, width='stretch')
            ratios = [p.ratio for p in panels if p.ratio == p.ratio]
            if ratios:
                st.success(
                    f"Inside/outside error ratio across locations: "
                    f"min {min(ratios):.1f}× · mean {sum(ratios)/len(ratios):.1f}× · "
                    f"max {max(ratios):.1f}×. Values above 1 mean the model performs "
                    f"worse inside the induced gap, supporting its use as ground truth."
                )

            st.markdown("**Baseline-subtracted view** — error increase caused by the gap")
            with st.spinner("Training a baseline model on the complete data (no weakspot)…"):
                _xx, _yy, baseline_raw = run_baseline_error_surface(
                    test_size=float(test_size), **_common
                )
                fig_delta = render_alignment_delta_figure(
                    panels, baseline_raw, model_name=model_name,
                    n_bumps=int(n_bumps), noise_std=float(noise_std),
                )
            st.pyplot(fig_delta, width='stretch')
            st.caption(
                "Top: model error with the induced gap overlaid. Bottom: the same "
                "surfaces minus a model trained on the complete data (no weakspot), "
                "so the common background error cancels and the gap's effect stands "
                "out. Uses the same dataset, model, exclusion, and training pipeline "
                "as the single-run experiment — only the weakspot location varies. "
                "Same routine that produces the paper figures "
                "(`scripts/generate_alignment_figure.py`)."
            )

# ─────────────────────────────────────────────────────────────
# RUN OPTIONS — two buttons at the bottom of the page
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🧪 Run Options")

from scripts.weakspot.sweep import (
    SWEEP_GRID, parameter_grid, remaining_combos, run_one, append_rows,
    load_completed_keys, EXTRACT_QUANTILE as SWEEP_EXTRACT_Q,
    IOU_QUANTILES as SWEEP_IOUS, GRID_RES_DEFAULT,
)
import time

CSV_PATH = Path("data/experiment_results/weakspot_experiment/sweep_results.csv")

fixed_params = {
    "dataset_key": AVAILABLE_DATASETS[dataset_name],
    "model_name": model_name,
    "complexity": float(complexity),
    "test_size": float(test_size),
    "grid_res": int(grid_res),
}

all_combos = parameter_grid()
total_runs = len(all_combos)
remaining = remaining_combos(fixed_params, CSV_PATH)
n_done = total_runs - len(remaining)
done_keys_total = len(load_completed_keys(CSV_PATH))

# Benchmarked at ~4.7 s/config for 17 methods (n=1000, MLP, 100 iters, n_bumps=5).
# Larger n_samples and higher complexity push it higher — use 5 s as the middle.
SECS_PER_RUN = 5.0
est_total_min = total_runs * SECS_PER_RUN / 60
est_remain_min = len(remaining) * SECS_PER_RUN / 60

col_left, col_right = st.columns(2, gap="large")

# ── LEFT: single-run button (replicates sidebar Run Experiment) ──────
with col_left:
    st.subheader("Single Run")
    st.caption(
        "Same as the sidebar **Run Experiment** button — runs one experiment "
        "with the current sidebar settings and updates the tabs above."
    )
    run_btn_bottom = st.button(
        "▶ Run Experiment",
        type="primary", width='stretch', key="run_bottom",
    )
    if run_btn_bottom:
        st.session_state["_trigger_single_run"] = True
        st.rerun()

# ── RIGHT: parameter sweep button ────────────────────────────────────
with col_right:
    st.subheader("Parameter Sweep")
    st.caption(
        "Batch runs the full grid (540 configs × 8 methods) and appends each "
        "per-method row to a CSV. **Resumable** — already-done param keys are "
        "skipped, so you can stop and restart."
    )
    cinfo1, cinfo2, cinfo3 = st.columns(3)
    cinfo1.metric("Total", total_runs)
    cinfo2.metric("Done (this config)", n_done)
    cinfo3.metric("Remaining", len(remaining))
    run_sweep_btn = st.button(
        "▶ Run Sweep" if remaining else "✓ Sweep complete for this config",
        type="primary", disabled=not remaining, width='stretch',
        key="run_sweep",
    )

st.markdown(
    f"""
**Sweep grid (varied):**
- center_x ∈ {SWEEP_GRID['center_x']} × center_y ∈ {SWEEP_GRID['center_y']}  → 9 locations
- radius ∈ {SWEEP_GRID['radius']}
- iterations ∈ {SWEEP_GRID['iterations']}
- n_samples ∈ {SWEEP_GRID['n_samples']}
- **n_bumps ∈ {SWEEP_GRID['n_bumps']}** (function complexity)
- **noise_std ∈ {SWEEP_GRID['noise_std']}** (underlying-function noise σ)

**Held fixed at current sidebar values:** dataset = `{dataset_name}`,
model = `{model_name}`, complexity = `{complexity}`, test split = `{test_size}`,
grid resolution = `{grid_res}`. Each run records all 17 detection methods.

**Per-run extraction threshold:** q = {SWEEP_EXTRACT_Q} (centroid/σ/area columns).
**IoU thresholds reported per row:** q ∈ {list(SWEEP_IOUS)}.

**Estimated time** (~{SECS_PER_RUN:.1f} s/run): full sweep ≈ **{est_total_min:.0f} min** ·
remaining ≈ **{est_remain_min:.0f} min**.

**CSV path:** `{CSV_PATH}`  ·  total rows currently completed across all configs:
**{done_keys_total}**.
"""
)

if run_sweep_btn and remaining:
    progress = st.progress(0.0, text=f"0 / {len(remaining)} — starting…")
    status = st.empty()
    t0 = time.time()
    n_total = len(remaining)
    for i, params in enumerate(remaining):
        try:
            rows = run_one(params)
            append_rows(CSV_PATH, rows)
        except Exception as e:
            st.warning(
                f"Run {i+1} failed for "
                f"(c=({params['center_x']},{params['center_y']}), r={params['radius']}, "
                f"it={params['iterations']}, n={params['n_samples']}): {e}"
            )
        elapsed = time.time() - t0
        per_run = elapsed / (i + 1)
        eta_min = per_run * (n_total - i - 1) / 60
        progress.progress(
            (i + 1) / n_total,
            text=(f"{i+1}/{n_total} — "
                  f"c=({params['center_x']:.2f},{params['center_y']:.2f}) "
                  f"r={params['radius']} it={params['iterations']} "
                  f"n={params['n_samples']}  ·  "
                  f"elapsed {elapsed:.0f}s  ·  ETA {eta_min:.1f} min"),
        )
    progress.empty()
    st.success(
        f"Sweep complete in {(time.time()-t0)/60:.1f} min. "
        f"Results appended to `{CSV_PATH}`. Rerun this page to see the updated counters."
    )

# Show the most recent rows so user gets immediate feedback
if CSV_PATH.exists():
    with st.expander("Preview last 20 CSV rows"):
        try:
            df_preview = pd.read_csv(CSV_PATH).tail(20)
            st.dataframe(df_preview, width='stretch', hide_index=True)
        except Exception as e:
            st.warning(f"Could not read CSV: {e}")
