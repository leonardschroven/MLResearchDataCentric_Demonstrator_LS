import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st
import numpy as np

from scripts.weakspot.datasets import (
    AVAILABLE_DATASETS, load_dataset, normalize,
    apply_exclusion_zone, apply_exclusion_knn, make_train_test_split
)
from scripts.weakspot.models import AVAILABLE_MODELS, build_model, train_and_evaluate
from scripts.weakspot.detection import DETECTION_METHODS, create_grid
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
dataset_name = st.sidebar.selectbox("Dataset", list(AVAILABLE_DATASETS.keys()))
n_samples = st.sidebar.slider("Sample size", 500, 2000, 1000, step=100)

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

st.sidebar.subheader("5. Visualisation")
smooth_sigma = st.sidebar.slider(
    "Surface smoothness (σ)", 0.00, 0.20, 0.00, 0.01,
    help="Gaussian kernel width for smoothing the polygon error surface. "
         "0 = raw data, higher = smoother."
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Experiment", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────

if not run_btn and "exp_results" not in st.session_state:
    st.info("Configure the parameters in the sidebar and click **Run Experiment**.")
    st.markdown("""
### Recommended open-license 2D regression datasets

| Dataset | Input features | Target | Source | License |
|---|---|---|---|---|
| California Housing | Longitude, Latitude | Median House Value | sklearn / StatLib | CC0 |
| Synthetic Peaks | x₁, x₂ | f(x₁,x₂) | Generated | — |
| Fish Market | Length, Height | Weight | [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/fish-market) | CC0 |
| Forest Fires | Temperature, RH | Burned Area | [UCI](https://archive.ics.uci.edu/dataset/162/forest+fires) | CC BY 4.0 |
""")
    st.stop()

if run_btn:
    center = np.array([center_x, center_y])
    dataset_key = AVAILABLE_DATASETS[dataset_name]

    with st.spinner("Loading data…"):
        X, y, x1_name, x2_name, y_name = load_dataset(dataset_key, n_samples=n_samples)
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
# RESULTS
# ─────────────────────────────────────────────────────────────
if "exp_results" not in st.session_state:
    st.stop()

R = st.session_state["exp_results"]
xx, yy = R["xx"], R["yy"]
center, radius = R["center"], R["radius"]
surfaces, gt_surface = R["surfaces"], R["gt_surface"]
X_test, errors = R["X_test"], R["errors"]

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
    ["📊 Data & Exclusion Zone", "📉 Model Errors", "🔍 Detection Methods", "⚖️ Comparison"]
)

with tab1:
    fig_data = plot_data_overview(
        R["X_norm"], R["y"], R["X_excl"],
        center, radius, R["x1_name"], R["x2_name"], R["y_name"],
        excl_mode=R["excl_mode"]
    )
    st.plotly_chart(fig_data, use_container_width=True)
    st.caption(
        f"Red ✕ = {R['n_excl']} excluded points. Dashed circle = induced weakspot zone."
    )

with tab2:
    st.plotly_chart(plot_metrics_bar(R["metrics"]), use_container_width=True)
    fig_err = plot_ground_truth_error(X_test, errors, center, radius)
    st.plotly_chart(fig_err, use_container_width=True)
    fig_err_3d = plot_ground_truth_error_3d(X_test, errors, center, radius, sigma=smooth_sigma)
    st.plotly_chart(fig_err_3d, use_container_width=True, key="tab2_gt_3d")
    st.caption(
        "Errors shown directly at test point locations — no interpolation. "
        "3D surface built via Delaunay triangulation of actual test points."
    )

with tab3:
    if surfaces:
        fig_all = plot_all_methods(xx, yy, surfaces, center, radius, X_test, errors)
        st.plotly_chart(fig_all, use_container_width=True)
        st.caption(
            "All surfaces normalized to [0,1]. Red dashed circle = induced exclusion zone."
        )
    else:
        st.warning("No methods produced results.")

with tab4:
    if surfaces:
        import pandas as pd

        best_name = st.selectbox("Compare method", list(surfaces.keys()), index=0)
        fig_cmp = plot_comparison(
            xx, yy, gt_surface, surfaces[best_name], best_name,
            center, radius, X_test, errors, sigma=smooth_sigma
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.subheader("Ground Truth — 3D Polygon Surface")
        fig_gt_3d = plot_ground_truth_error_3d(
            X_test, errors, center, radius, sigma=smooth_sigma
        )
        st.plotly_chart(fig_gt_3d, use_container_width=True, key="tab4_gt_3d")
        st.caption(
            "Surface built by Delaunay triangulation of actual test points — no algorithm involved. "
            "Adjust **Surface smoothness (σ)** in the sidebar to smooth the mesh."
        )

        # ── Position & distance table ──────────────────────────────────
        st.subheader("Weakspot Position Analysis")
        st.caption(
            "Found position = grid point with the highest predicted error for each method. "
            "Distance is Euclidean in normalized [0,1]² space."
        )

        _, _, grid_flat_cmp = create_grid(resolution=len(xx[0]))

        pos_rows = []
        for name, surf in surfaces.items():
            max_idx = int(np.argmax(surf))
            found_x, found_y = grid_flat_cmp[max_idx]
            dist = float(np.sqrt((found_x - center[0])**2 + (found_y - center[1])**2))

            # Overlap score
            denom = np.linalg.norm(gt_surface) * np.linalg.norm(surf)
            overlap = float(np.dot(gt_surface, surf) / denom) if denom > 0 else 0.0

            pos_rows.append({
                "Method": name,
                "Induced x₁": round(float(center[0]), 3),
                "Induced x₂": round(float(center[1]), 3),
                "Found x₁": round(found_x, 3),
                "Found x₂": round(found_y, 3),
                "Distance": round(dist, 4),
                "Overlap Score": round(overlap, 4),
            })

        df_pos = (
            pd.DataFrame(pos_rows)
            .sort_values("Distance")
            .reset_index(drop=True)
        )
        st.dataframe(df_pos, use_container_width=True, hide_index=True)

        # Highlight best method
        best_row = df_pos.iloc[0]
        st.success(
            f"**Closest detection:** {best_row['Method']} "
            f"— found at ({best_row['Found x₁']}, {best_row['Found x₂']}), "
            f"distance {best_row['Distance']} from induced center "
            f"({best_row['Induced x₁']}, {best_row['Induced x₂']})."
        )
    else:
        st.warning("Run the experiment first.")
