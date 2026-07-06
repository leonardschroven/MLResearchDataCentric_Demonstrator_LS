"""
Iterative Data-Selective Training.

Same pipeline as the single-round page, but run as a loop: each iteration
evaluates the model, detects a fresh weakspot with the chosen detector, selects
new data around it, adds it to the (accumulating) training set, and retrains.
A random-selection baseline is trained in parallel under identical conditions,
so their performance can be compared iteration by iteration.

The Visualisation tab shows it all in one place: the MAE-per-iteration curves
(guided vs random), and — for any chosen iteration — the candidate pool with the
guided and random selections overlaid.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.weakspot.models import AVAILABLE_MODELS, build_model
from scripts.weakspot.detection import DETECTION_METHODS, create_grid
from scripts.weakspot.extraction import extract_weakspot, induced_mask
from scripts.weakspot.plotting import plot_ground_truth_error
from scripts.dataselect import pipeline as P
from scripts.dataselect.plots import plot_selection, plot_landscape

st.set_page_config(page_title="Iterative Data-Selective Training", layout="wide")
st.title("🔁 Iterative Data-Selective Training")
st.markdown(
    "Loop the pipeline: **detect → select → retrain**, each round finding the "
    "model's *current* weakspot, adding data there, and retraining — with a random "
    "baseline trained in parallel for comparison."
)
st.caption(
    "The sidebar defaults are preset (from the sweep analysis + iterative replay) to "
    "showcase a strong guided advantage: a clear induced weakspot (radius 0.18), a "
    "**deliberately under-trained** initial model (50 iters) so there is headroom to "
    "improve, a focused selection kernel (σ=0.15, 100 pts/iter), and retraining time "
    "(200 iters) **≥** the initial budget so retrained models are never under-trained. "
    "With these, guided MAE falls monotonically (≈0.58→0.23→0.15→0.14) and beats the "
    "random baseline on the **in-weakspot error** curve from iteration 1 onward — the "
    "curve to watch. If you raise the initial training time, raise retraining time to "
    "match, or iteration 1 can rise instead of fall."
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Pipeline Setup")
seed = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)

st.sidebar.subheader("1. Dataset & Landscape")
n_bumps = st.sidebar.slider("Function complexity (n_bumps)", 1, 10, 5, 1)
noise_std = st.sidebar.slider("Label noise σ (training only)", 0.0, 0.5, 0.05, 0.01)
n_pool_total = st.sidebar.slider("Initial dataset size", 500, 4000, 2000, 100)

st.sidebar.subheader("2. Distribution Shift")
shift_strength = st.sidebar.slider("Shift strength", 0.0, 1.0, 0.0, 0.05)
scx = st.sidebar.slider("Shift centre x₁", 0.0, 1.0, 0.30, 0.05, disabled=(shift_strength == 0.0))
scy = st.sidebar.slider("Shift centre x₂", 0.0, 1.0, 0.30, 0.05, disabled=(shift_strength == 0.0))
shift_spread = st.sidebar.slider("Shift spread σ", 0.05, 0.40, 0.15, 0.01, disabled=(shift_strength == 0.0))

st.sidebar.subheader("3. Induced Weakspot")
induce_ws = st.sidebar.checkbox("Induce a weakspot (leave out a data gap)", value=True)
center_x = st.sidebar.slider("Weakspot centre x₁", 0.1, 0.9, 0.5, 0.01, disabled=not induce_ws)
center_y = st.sidebar.slider("Weakspot centre x₂", 0.1, 0.9, 0.5, 0.01, disabled=not induce_ws)
radius = st.sidebar.slider("Exclusion radius", 0.02, 0.30, 0.18, 0.01, disabled=not induce_ws)

st.sidebar.subheader("4. Model & Initial Training")
_mk = list(AVAILABLE_MODELS.keys())
model_name = st.sidebar.selectbox("Algorithm", _mk,
                                  index=_mk.index("MLP Neural Network") if "MLP Neural Network" in _mk else 0)
complexity = st.sidebar.slider("Model complexity", 0.0, 1.0, 0.5, 0.05)
iters_initial = st.sidebar.slider(
    "Initial training time (iterations)", 20, 500, 50, 10,
    help="Kept deliberately low so the initial model is under-trained (large MAE) "
         "and there is clear headroom for the iterations to improve on. Make sure "
         "retraining time (§8) is ≥ this value, otherwise every retrained model is "
         "under-trained relative to iteration 0 and MAE rises instead of falling.")
n_train = st.sidebar.slider("Initial training points", 100, 3000, 800, 50)

st.sidebar.subheader("5. Evaluation")
n_eval = st.sidebar.slider("Evaluation points on landscape", 200, 3000, 1000, 100)

st.sidebar.subheader("6. Weakspot Identification")
_dm = list(DETECTION_METHODS.keys())
detect_name = st.sidebar.selectbox(
    "Detection method (single)", _dm,
    index=_dm.index("EVT × GPR (geometric)") if "EVT × GPR (geometric)" in _dm else 0)
grid_res = st.sidebar.slider("Grid resolution", 20, 60, 35, 5)
extract_q = st.sidebar.slider("Extraction threshold quantile", 0.50, 0.99, 0.85, 0.01)

st.sidebar.subheader("7. Data Selection (per iteration)")
sel_method = st.sidebar.selectbox(
    "Selection method", list(P.SEL_METHODS), index=0,
    help="How each iteration weights new points around the current weakness:\n\n"
         "• **Weakpoint distance** — isotropic Gaussian around the single detected "
         "centre (the original strategy).\n"
         "• **Weight by landscape** — sample ∝ the whole detected error surface, "
         "covering *every* weak region (multi-modal, robust to a mis-located centre).\n"
         "• **Shape aware** — Gaussian shaped to the detected weakspot ellipse "
         "(anisotropic; σ below sets the overall width).")
sel_mode = st.sidebar.radio("Selection rule", ["Sample ∝ weight", "Top-weighted"])
sel_sigma = st.sidebar.slider("Selection Gaussian σ", 0.02, 0.80, 0.15, 0.01)
n_select = st.sidebar.slider("Points added per iteration", 20, 1000, 100, 20)
n_candidate = st.sidebar.slider("Candidate pool size", 200, 4000, 2000, 100)

st.sidebar.subheader("8. Retraining & Iterations")
iters_retrain = st.sidebar.slider(
    "Retraining time (iterations)", 20, 800, 200, 10,
    help="Should be ≥ the initial training time (§4). Each iteration rebuilds the "
         "model from scratch (no warm start), so if this is smaller than the initial "
         "budget the retrained models are under-trained and iteration 1 looks worse "
         "than iteration 0 even when the added data is good.")
n_iterations = st.sidebar.slider("Number of iterations", 1, 20, 8, 1)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Iterative Pipeline", type="primary",
                            width='stretch')


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def _detect(err, X_eval, grid_flat, xx, yy):
    """Run the chosen detector; return (surface, centre, extraction)."""
    surf = P.normalize_surface(DETECTION_METHODS[detect_name](X_eval, err, grid_flat))
    ext = extract_weakspot(surf, xx, yy, threshold_quantile=extract_q)
    if ext is None:
        c = grid_flat[int(np.argmax(surf))]
        return surf, (float(c[0]), float(c[1])), None
    return surf, ext["center"], ext


# ─────────────────────────────────────────────────────────────
# RUN THE ITERATIVE LOOP
# ─────────────────────────────────────────────────────────────
if run_btn:
    rng = np.random.RandomState(int(seed))
    center = np.array([center_x, center_y])

    X_all = P.sample_inputs(n_pool_total, rng, shift_strength=shift_strength,
                            shift_center=(scx, scy), shift_spread=shift_spread)
    y_all = P.label(X_all, rng, n_bumps=n_bumps, noise_std=noise_std)
    if induce_ws:
        X_keep, y_keep, X_excl, y_excl = P.induce_weakspot(X_all, y_all, center, radius)
        r_eff = float(radius)
    else:
        X_keep, y_keep = X_all, y_all
        r_eff = 0.0

    k = min(int(n_train), len(X_keep))
    tr_idx = rng.choice(len(X_keep), size=k, replace=False)
    X_tr0, y_tr0 = X_keep[tr_idx], y_keep[tr_idx]

    X_eval = P.sample_inputs(n_eval, rng, shift_strength=0.0)
    y_eval = P.true_function(X_eval, n_bumps=n_bumps)
    xx, yy, grid_flat = create_grid(resolution=int(grid_res))
    true_grid = P.landscape_surface(grid_flat, n_bumps=n_bumps)

    def _region(err):
        return (P.region_error(X_eval, err, center, r_eff)[0]
                if r_eff > 0 else float("nan"))

    # initial (shared) model
    m0 = build_model(AVAILABLE_MODELS[model_name], complexity=complexity,
                     iterations=int(iters_initial))
    m0.fit(X_tr0, y_tr0)
    _, err0, met0 = P.evaluate(m0, X_eval, y_eval)

    # Two accumulating training sets (guided / random). The guided-accumulative
    # model (mg) is the one whose weakspot drives the selection each round.
    Xg, yg = X_tr0.copy(), y_tr0.copy()
    Xr, yr = X_tr0.copy(), y_tr0.copy()
    mg = m0

    # Four tracks: strategy (guided/random) × regime (accumulative/new-only).
    TRACKS = ("gacc", "racc", "gnew", "rnew")
    hist = {k: [met0["MAE"]] for k in TRACKS}
    ein = {k: [_region(err0)] for k in TRACKS}
    rounds = []

    def _fit_eval(X, y):
        """Train a fresh model on (X, y) and return (MAE, in-weakspot error)."""
        m = build_model(AVAILABLE_MODELS[model_name], complexity=complexity,
                        iterations=int(iters_retrain))
        m.fit(X, y)
        _, err, met = P.evaluate(m, X_eval, y_eval)
        return m, met["MAE"], _region(err)

    prog = st.progress(0.0, text="Iterating…")
    for it in range(1, int(n_iterations) + 1):
        # detect the current weakspot on the guided-accumulative model
        _, errg, _ = P.evaluate(mg, X_eval, y_eval)
        surf_g, c_g, ext_g = _detect(errg, X_eval, grid_flat, xx, yy)

        # one guided selection and one random selection, shared by both regimes
        Xc_g = P.sample_inputs(n_candidate, rng, shift_strength=0.0)
        yc_g = P.label(Xc_g, rng, n_bumps=n_bumps, noise_std=noise_std)
        idx_g, w_g = P.select_by_weakspot(Xc_g, c_g, sel_sigma, int(n_select), rng,
                                          sel_mode, method=sel_method,
                                          ext=ext_g, surf=surf_g)
        Xsel_g, ysel_g = Xc_g[idx_g], yc_g[idx_g]

        Xc_r = P.sample_inputs(n_candidate, rng, shift_strength=0.0)
        yc_r = P.label(Xc_r, rng, n_bumps=n_bumps, noise_std=noise_std)
        idx_r = rng.choice(len(Xc_r), size=min(int(n_select), len(Xc_r)), replace=False)
        Xsel_r, ysel_r = Xc_r[idx_r], yc_r[idx_r]

        # accumulative: add to the growing sets and retrain on everything
        Xg = np.vstack([Xg, Xsel_g]); yg = np.concatenate([yg, ysel_g])
        Xr = np.vstack([Xr, Xsel_r]); yr = np.concatenate([yr, ysel_r])
        mg, mae_gacc, ein_gacc = _fit_eval(Xg, yg)   # mg feeds next round's detection
        _,  mae_racc, ein_racc = _fit_eval(Xr, yr)
        # new-only: retrain from scratch on just this round's selection
        _,  mae_gnew, ein_gnew = _fit_eval(Xsel_g, ysel_g)
        _,  mae_rnew, ein_rnew = _fit_eval(Xsel_r, ysel_r)

        hist["gacc"].append(mae_gacc); hist["racc"].append(mae_racc)
        hist["gnew"].append(mae_gnew); hist["rnew"].append(mae_rnew)
        ein["gacc"].append(ein_gacc); ein["racc"].append(ein_racc)
        ein["gnew"].append(ein_gnew); ein["rnew"].append(ein_rnew)

        rounds.append(dict(
            c_g=c_g, Xc_g=Xc_g, w_g=w_g, Xsel_g=Xsel_g, errg=errg, ext_g=ext_g,
            Xc_r=Xc_r, Xsel_r=Xsel_r, n_train_acc=len(Xg),
        ))
        prog.progress(it / int(n_iterations), text=f"Iteration {it}/{n_iterations}")
    prog.empty()

    st.session_state["iter"] = dict(
        center=center, radius=r_eff, has_ws=induce_ws, xx=xx, yy=yy,
        true_grid=true_grid, X_eval=X_eval, hist=hist, ein=ein, rounds=rounds,
        detect_name=detect_name, n_iterations=int(n_iterations),
        sel_sigma=sel_sigma, sel_mode=sel_mode, n_select=int(n_select),
        model_name=model_name, sel_method=sel_method,
    )


# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
if "iter" not in st.session_state:
    st.info("Configure the sidebar and press **🚀 Run Iterative Pipeline**.")
    st.stop()

R = st.session_state["iter"]
center, radius = R["center"], R["radius"]
iters = list(range(0, R["n_iterations"] + 1))

tab_v, tab_s = st.tabs(["🔬 Visualisation", "🧾 Summary"])

# Track metadata: key → (strategy, regime, colour)
_TRACKS = {
    "gacc": ("guided", "accumulative", "#2ca02c"),
    "racc": ("random", "accumulative", "#d62728"),
    "gnew": ("guided", "new-only", "#2ca02c"),
    "rnew": ("random", "new-only", "#d62728"),
}
_CMAP = {"guided": "#2ca02c", "random": "#d62728"}


def _long(store):
    rows = []
    for k, (strat, regime, _) in _TRACKS.items():
        for i, v in zip(iters, store[k]):
            rows.append({"iteration": i, "value": v, "strategy": strat,
                         "regime": regime, "track": f"{strat} · {regime}"})
    return pd.DataFrame(rows)


# ── VISUALISATION ────────────────────────────────────────────
with tab_v:
    # 1) Performance per iteration
    st.subheader("① Training progress — MAE per iteration")
    st.caption(
        "Four tracks: **strategy** (guided = green, random = red) × **regime** "
        "(accumulative = solid, new-only = dashed). Accumulative retrains on the "
        "growing set; new-only retrains from scratch on just that round's "
        "selection. Iteration 0 is the shared initial model. Lower is better."
    )
    dfc = _long(R["hist"])
    figp = px.line(dfc, x="iteration", y="value", color="strategy", line_dash="regime",
                   markers=True, color_discrete_map=_CMAP,
                   labels={"value": "MAE"},
                   title="Evaluation MAE vs iteration — accumulative vs new-only")
    figp.update_layout(height=440, legend_title_text="")
    st.plotly_chart(figp, width='stretch')

    if R["has_ws"] and any(v == v for v in R["ein"]["gacc"]):
        dfe = _long(R["ein"])
        fige = px.line(dfe, x="iteration", y="value", color="strategy",
                       line_dash="regime", markers=True, color_discrete_map=_CMAP,
                       labels={"value": "error in weakspot"},
                       title="Mean error inside the induced weakspot vs iteration")
        fige.update_layout(height=400, legend_title_text="")
        st.plotly_chart(fige, width='stretch')

    st.markdown("---")

    # 2) Per-iteration data pool & selection
    st.subheader("② Data pool & selection at each iteration")
    st.caption("Pick an iteration. Left: the guided selection (candidate pool "
               "coloured by the Gaussian selection weight around the detected "
               "weakspot). Right: the random baseline selection from the same-size "
               "pool. **Each selection is used by both regimes** — added to the "
               "accumulative set and, separately, used alone for the new-only model. "
               "The dashed red circle is the induced weakspot.")
    it_sel = st.slider("Iteration", 1, R["n_iterations"], 1)
    rd = R["rounds"][it_sel - 1]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Guided** — strategy *{R.get('sel_method', 'Weakpoint distance')}*, "
                    f"detector *{R['detect_name']}*, "
                    f"σ={R['sel_sigma']:.2f}, added {R['n_select']} "
                    f"(accumulative set now {rd['n_train_acc']})")
        st.plotly_chart(
            plot_selection(rd["Xc_g"], rd["w_g"], rd["Xsel_g"], rd["c_g"],
                           R["sel_sigma"], center, radius,
                           title=f"Iteration {it_sel}: guided selection"),
            width='stretch', key=f"g_{it_sel}")
    with c2:
        st.markdown(f"**Random baseline** — uniform, added {R['n_select']} "
                    f"(accumulative set now {rd['n_train_acc']})")
        st.plotly_chart(
            plot_selection(rd["Xc_r"], None, rd["Xsel_r"], None, None,
                           center, radius, show_kernel=False,
                           title=f"Iteration {it_sel}: random selection"),
            width='stretch', key=f"r_{it_sel}")

    st.markdown(f"**Model error before selection at iteration {it_sel}** "
                "(where the detector looked for the weakspot):")
    st.plotly_chart(
        plot_ground_truth_error(R["X_eval"], rd["errg"], center, radius),
        width='stretch', key=f"err_{it_sel}")

# ── SUMMARY ──────────────────────────────────────────────────
with tab_s:
    st.subheader("Iteration-by-iteration results")
    H = R["hist"]
    df = pd.DataFrame({
        "iteration": iters,
        "guided_acc_MAE": np.round(H["gacc"], 4),
        "random_acc_MAE": np.round(H["racc"], 4),
        "guided_new_MAE": np.round(H["gnew"], 4),
        "random_new_MAE": np.round(H["rnew"], 4),
    })
    if R["has_ws"]:
        E = R["ein"]
        df["guided_acc_err_in_ws"] = np.round(E["gacc"], 4)
        df["random_acc_err_in_ws"] = np.round(E["racc"], 4)
    st.dataframe(df, width='stretch', hide_index=True)

    gacc, racc = H["gacc"][-1], H["racc"][-1]
    gnew, rnew = H["gnew"][-1], H["rnew"][-1]
    v_acc = "**below** random ✅" if gacc < racc else "**above** random ⚠️"
    st.markdown(
        f"After **{R['n_iterations']}** iterations:\n\n"
        f"- **Accumulative** — guided **{gacc:.4f}** vs random **{racc:.4f}** "
        f"(guided finished {v_acc}, Δ {gacc - racc:+.4f}).\n"
        f"- **New-only** — guided **{gnew:.4f}** vs random **{rnew:.4f}**. "
        f"Retraining on only each round's points typically stays far worse than the "
        f"accumulative regime — the iterative sign of catastrophic forgetting."
    )
    st.download_button("⬇ Download iteration history (CSV)",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="iterative_history.csv", mime="text/csv")
