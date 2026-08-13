"""Iterative Data-Selective Training.

The follow-up experiment to the companion paper *Data-Selective Machine Learning
Training Based on Statistical Weakspot Identification*, which analysed a single
round of

    evaluate → identify weakspot → select data → retrain

deliberately, "in order to isolate the effect of one targeted selection from the
confounding effects of iteration", and closed by naming the loop as its own
future work. This page runs that loop: each round re-evaluates the model,
detects its *current* weakspot, selects new data around it and continues
training, with a matched random-selection baseline trained in parallel under
identical conditions and from the same candidate pool.

Everything the loop makes newly meaningful is exposed as an axis:

* **Architecture and optimiser** — the single-round study held one MLP shape
  fixed. Under repetition, capacity and learning rate govern how much of the old
  input space survives each round of continued training.
* **Per-iteration schedules** for the learning rate and for the paper's two
  dominant coverage controls (rehearsal mix α, kernel width σ), including an
  adaptive variant that relaxes the focus as the detected weakspot heals — the
  extension the paper's Discussion names explicitly.
* **Regimes** — accumulative (the industrial protocol the paper listed as
  untested), new-only (the paper's own conservative protocol, repeated), and a
  size-matched scaled-accum control.
* **Staging** — a budget- and compute-matched single-shot reference answering
  whether K rounds of n points beat one round of K·n.

The heavy lifting lives in ``scripts.iterative`` so this page, the parameter
sweep and its parallel workers all run the same code. The single-round pages
(07/08) and the weakspot pages (05/06) are untouched.
"""
from __future__ import annotations

import os
import sys
import time
from math import prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from scripts.dataselect import pipeline as P
from scripts.dataselect.plots import plot_landscape, plot_selection, plot_training_points
from scripts.iterative import loop as L
from scripts.iterative import models as M
from scripts.iterative import plots as IP
from scripts.iterative import sweep as SW
from scripts.weakspot.detection import DETECTION_METHODS
from scripts.weakspot.models import AVAILABLE_MODELS
from scripts.weakspot.plotting import plot_data_overview, plot_ground_truth_error

st.set_page_config(page_title="Iterative Data-Selective Training", layout="wide")
st.title("🔁 Iterative Data-Selective Training")
st.markdown(
    "Loop the pipeline — **detect → select → continue training** — each round "
    "finding the model's *current* weakspot, adding data there and training on, "
    "with a random baseline advancing in parallel from the same candidate pool."
)
st.caption(
    "Defaults are the companion paper's operating point (5 bumps, label noise "
    "0.05, gap radius 0.25, an undertrained initial model of 12 iterations on 100 "
    "points, kernel σ=0.5, rehearsal mix α=0.5, 100 points added per round, 400 "
    "retraining iterations, warm-start), now run **eight times over** instead of "
    "once. Every axis this study adds starts at its neutral setting, so an "
    "out-of-the-box run is exactly the paper's configuration looped — switch one "
    "schedule, architecture or regime on at a time and the change is attributable."
)

# ─────────────────────────────────────────────────────────────
# PRESETS — starting points, each a complete config
# ─────────────────────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    "Paper operating point (looped)": {},
    "Paper broad-sweep best (small gap 0.12)": dict(radius=0.12),
    "Forgetting stress test (narrow σ, pure guided)": dict(
        sel_sigma=0.1, mix_ratio=1.0, regimes=list(L.REGIMES)),
    "Relax-the-focus schedule (adaptive α)": dict(
        mix_schedule="Adaptive (weakspot severity)", mix_ratio=1.0, sel_sigma=0.2),
    "Consolidating learning rate (cosine decay)": dict(
        lr_schedule="Cosine anneal", iters_retrain=200),
    "Fixed dataset (pool consumed)": dict(
        pool_mode=L.POOL_MODES[1], n_candidate=1200, n_iterations=8),
    "Legacy page-09 showcase": dict(
        n_bumps=5, noise_std=0.05, radius=0.22, iters_initial=50, n_train=800,
        sel_sigma=0.15, mix_ratio=1.0, n_select=100, iters_retrain=200,
        n_iterations=8, detector="EVT × GPR (geometric)", n_pool_total=2000),
}

# Every sidebar widget is keyed and its value seeded here, so a preset can rewrite
# the whole sidebar in one go. Because the keys already exist in session state the
# widgets below deliberately take **no** ``value``/``index``/``default`` argument —
# passing one alongside a pre-seeded key is what Streamlit warns about.
S = st.session_state
for _k, _v in L.DEFAULTS.items():
    S.setdefault(f"it_{_k}", _v)
S.setdefault("it_induce", L.DEFAULTS["radius"] > 0)
S.setdefault("it_training_mode", "Warm-start (transfer)")

st.sidebar.header("⚙️ Pipeline Setup")
_preset = st.sidebar.selectbox(
    "Preset", list(PRESETS), key="it_preset",
    help="A complete starting configuration. **Paper operating point** reproduces "
         "the companion study's best single-round setting, looped. The others each "
         "isolate one question this study adds. Pressing *Apply* overwrites every "
         "sidebar value, so a preset is always a clean starting point.")
if st.sidebar.button("↺ Apply preset", width='stretch'):
    _full = L.default_config(**PRESETS[_preset])
    for _k, _v in _full.items():
        S[f"it_{_k}"] = _v
    # Two widgets present a config field in a different form; keep them in step.
    S["it_induce"] = _full["radius"] > 0
    S["it_training_mode"] = ("Warm-start (transfer)" if _full["warm_start"]
                             else "From scratch (retrain)")
    st.rerun()

sb = st.sidebar
seed = sb.number_input("Random seed", 0, 9999, step=1, key="it_seed")

sb.subheader("1. Dataset & Landscape")
n_bumps = sb.slider("Function complexity (n_bumps)", 1, 10, step=1, key="it_n_bumps")
noise_std = sb.slider("Label noise σ (training only)", 0.0, 0.5, step=0.01,
                      key="it_noise_std")
n_pool_total = sb.slider("Initial dataset size", 500, 4000, step=100,
                         key="it_n_pool_total")

sb.subheader("2. Distribution Shift")
shift_strength = sb.slider("Shift strength", 0.0, 1.0, step=0.05,
                           key="it_shift_strength")
_no_shift = shift_strength == 0.0
scx = sb.slider("Shift centre x₁", 0.0, 1.0, step=0.05,
                disabled=_no_shift, key="it_shift_center_x")
scy = sb.slider("Shift centre x₂", 0.0, 1.0, step=0.05,
                disabled=_no_shift, key="it_shift_center_y")
shift_spread = sb.slider("Shift spread σ", 0.05, 0.40, step=0.01,
                         disabled=_no_shift, key="it_shift_spread")

sb.subheader("3. Induced Weakspot")
induce_ws = sb.checkbox("Induce a weakspot (leave out a data gap)", key="it_induce")
center_x = sb.slider("Weakspot centre x₁", 0.1, 0.9, step=0.01,
                     disabled=not induce_ws, key="it_center_x")
center_y = sb.slider("Weakspot centre x₂", 0.1, 0.9, step=0.01,
                     disabled=not induce_ws, key="it_center_y")
radius = sb.slider(
    "Exclusion radius", 0.0, 0.40, step=0.01, disabled=not induce_ws,
    key="it_radius",
    help="0.25 is the enlarged gap the paper's isolation experiments used, so a "
         "local repair is big enough to move the whole-area metric; 0.12 was its "
         "broad-sweep best. With a small gap the random baseline hits it by chance "
         "and the guided advantage collapses.")

sb.subheader("4. Model & Architecture")
model_name = sb.selectbox("Algorithm", list(AVAILABLE_MODELS), key="it_model_name")
_is_mlp = AVAILABLE_MODELS[model_name] == "mlp"
arch = sb.selectbox(
    "MLP architecture", list(M.ARCHITECTURES), key="it_arch",
    disabled=not _is_mlp,
    help="Hidden-layer shape. **Complexity-scaled (2 × h)** reproduces the "
         "companion paper's model exactly (h = complexity × 128). The loop makes "
         "this a real axis: capacity decides how much of the already-learnt input "
         "space a round of continued training on the weak region can preserve.")
complexity = sb.slider("Model complexity", 0.0, 1.0, step=0.05,
                       key="it_complexity",
                       help="Capacity for the non-MLP algorithms, and the width h "
                            "of the complexity-scaled MLP architecture.")
activation = sb.selectbox("Activation", list(M.ACTIVATIONS),
                          key="it_activation", disabled=not _is_mlp)
solver = sb.selectbox("Solver", list(M.SOLVERS), key="it_solver",
                      disabled=not _is_mlp,
                      help="`adam` adapts its own step sizes; `sgd` follows the "
                           "learning rate literally, so the schedule below bites "
                           "harder.")
alpha_l2 = sb.select_slider(
    "L2 penalty α", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    key="it_alpha", disabled=not _is_mlp,
    help="Weight decay. A stronger penalty resists the per-round drift towards the "
         "weak region — the optimiser-side counterpart of uniform rehearsal.")
batch_size = sb.number_input(
    "Batch size (0 = auto)", 0, 2048, step=16, key="it_batch_size",
    disabled=not _is_mlp,
    help="0 uses sklearn's default (min(200, n)). Small batches take more, noisier "
         "steps per round on the same data.")
lr_init = sb.select_slider(
    "Base learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    key="it_lr_init", disabled=not _is_mlp,
    help="Starting rate, and the value the schedule in §8 decays from. 1e-3 is "
         "sklearn's default and what the companion paper used implicitly.")

sb.subheader("5. Initial Training")
iters_initial = sb.slider(
    "Initial training time (iterations)", 3, 2000, step=1,
    key="it_iters_initial",
    help="Kept deliberately low (12 at the paper's operating point) so the initial "
         "model is undertrained and there is headroom for the loop. The paper found "
         "headroom mattered more than the reliability of the diagnosis — whether "
         "that survives repetition is one of this study's questions.")
n_train = sb.slider("Initial training points", 50, 3000, step=10, key="it_n_train")

sb.subheader("6. Evaluation")
n_eval = sb.slider("Evaluation points on landscape", 200, 3000, step=100,
                   key="it_n_eval")

sb.subheader("7. Weakspot Identification")
detector = sb.selectbox(
    "Detection method", list(DETECTION_METHODS), key="it_detector",
    help="Quantile Regression was the strongest of all seventeen detectors at the "
         "paper's operating point; every one of them beat the random baseline there, "
         "so this axis is second-order — settle coverage first.")
detect_mode = sb.radio(
    "Target", list(L.DETECT_MODES), key="it_detect_mode",
    help="**Re-detect each iteration** tracks the weakness as it migrates — the "
         "point of looping. **Detect once** pins the first detection as a fixed "
         "target, isolating how much of the benefit comes from *tracking* rather "
         "than from aiming once.")
grid_res = sb.slider("Grid resolution", 20, 60, step=5, key="it_grid_res")
extract_q = sb.slider("Extraction threshold quantile", 0.50, 0.99, step=0.01,
                      key="it_extract_q")

sb.subheader("8. Data Selection")
sel_method = sb.selectbox(
    "Selection strategy", list(P.SEL_METHODS), key="it_sel_method",
    help="• **Weakpoint distance** — isotropic Gaussian on the detected centre.\n"
         "• **Weight by landscape** — sample ∝ the whole detected error surface "
         "(multi-modal, robust to a mis-located centre).\n"
         "• **Shape aware** — Gaussian shaped to the detected ellipse.")
sel_mode = sb.radio("Selection rule", ["Sample ∝ weight", "Top-weighted"],
                    key="it_sel_mode")
sel_sigma = sb.slider(
    "Kernel width σ (start)", 0.02, 1.50, step=0.01, key="it_sel_sigma",
    help="One of the paper's two dominant controls. Narrow = pours data into the "
         "gap and risks forgetting; above ~0.5 the kernel is nearly flat on the unit "
         "square and guided selection approaches the random baseline.")
sigma_schedule = sb.selectbox(
    "σ schedule", list(M.SIGMA_SCHEDULES), key="it_sigma_schedule",
    help="How σ changes per round. **Adaptive** widens the kernel as the detected "
         "weakspot heals (severity measured from the model's own error surface, no "
         "ground truth) — the kernel-side form of the paper's proposed 'relax the "
         "focus as the weak region shrinks'.")
sigma_rate = sb.slider("σ schedule rate", 0.0, 2.0, step=0.05,
                       key="it_sigma_rate",
                       disabled=(sigma_schedule == "Constant"),
                       help="Total relative widening/narrowing reached at the final "
                            "iteration (linear), or the per-round growth factor − 1 "
                            "(exponential).")
mix_ratio = sb.slider(
    "Rehearsal mix α (start)", 0.0, 1.0, step=0.05, key="it_mix_ratio",
    help="Fraction of each round's points drawn by the weakspot kernel; the rest are "
         "uniform rehearsal. **The paper's master switch** — α=0 is the random "
         "baseline by construction, and past a crossover near α≈0.4 under a narrow "
         "kernel its single-round model forgot the rest of the space catastrophically.")
mix_schedule = sb.selectbox("α schedule", list(M.MIX_SCHEDULES), key="it_mix_schedule")
mix_rate = sb.slider("α schedule rate", 0.0, 1.0, step=0.05,
                     key="it_mix_rate", disabled=(mix_schedule == "Constant"),
                     help="Fraction of α given up by the final iteration (linear "
                          "schedules) or the per-round multiplier (exponential).")
n_select = sb.slider("Points added per iteration", 10, 1000, step=10, key="it_n_select")
n_candidate = sb.slider("Candidate pool size", 200, 4000, step=100,
                        key="it_n_candidate")
pool_mode = sb.radio(
    "Candidate pool", list(L.POOL_MODES), key="it_pool_mode",
    help="**Fresh pool each iteration** is the paper's protocol. **Fixed pool** draws "
         "one pool up front and consumes it without replacement — the fixed-dataset "
         "industrial setting the study is motivated by, where guidance eventually "
         "exhausts the candidates near the weakspot.")

sb.subheader("9. Retraining & Loop")
training_mode = sb.radio(
    "Training mode", ["Warm-start (transfer)", "From scratch (retrain)"],
    key="it_training_mode",
    help="**Warm-start** continues each track's model from the previous round's "
         "weights, which is what makes this a *continued* training study. MLP only; "
         "other algorithms fall back to from-scratch.")
early_stop = sb.checkbox("Early stopping (regularise MLP)", key="it_early_stopping")
iters_retrain = sb.slider("Retraining time per iteration", 20, 2000, step=10,
                          key="it_iters_retrain")
n_iterations = sb.slider("Number of iterations", 1, 30, step=1, key="it_n_iterations")
lr_schedule = sb.selectbox(
    "Learning-rate schedule", list(M.LR_SCHEDULES),
    key="it_lr_schedule", disabled=not _is_mlp,
    help="One step per **loop iteration**, not per epoch: how far each round may "
         "move the weights. Decay lets early rounds absorb the weak region and late "
         "rounds consolidate — the optimiser-side counter to forgetting.")
_lr_flat = lr_schedule == "Constant"
lr_gamma = sb.slider("LR decay γ", 0.1, 1.0, step=0.05, key="it_lr_gamma",
                     disabled=_lr_flat or not _is_mlp)
lr_min = sb.select_slider("LR floor (cosine)", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
                          key="it_lr_min", disabled=_lr_flat or not _is_mlp)
lr_step = sb.slider("LR step / restart period", 1, 10, step=1,
                    key="it_lr_step", disabled=_lr_flat or not _is_mlp)
regimes = sb.multiselect(
    "Regimes to run", list(L.REGIMES), key="it_regimes",
    help="**Accumulative** retrains on the whole growing set (always run — the "
         "industrial protocol the paper left untested). **New-only** uses just that "
         "round's points, repeating the paper's conservative protocol. "
         "**Scaled-accum** subsamples the accumulated pool to the new-only point "
         "count: if it tracks accumulative at equal size, the win is the training "
         "*distribution*, not the amount of data.")
single_shot = sb.checkbox(
    "Budget-matched single-shot control", key="it_single_shot",
    help="Also trains one model on K × n_select points added in a single round, with "
         "K × the retraining budget — so data and compute match and only the staging "
         "differs. Answers whether iterating is worth anything at all.")
tie_model_seed = sb.checkbox("Tie weight init to the seed", key="it_tie_model_seed",
                             help="Off (default, and what the paper did) fixes the "
                                  "MLP initialisation at 42 so only the data varies "
                                  "across seeds. On also varies the initialisation.")

sb.markdown("---")
run_btn = sb.button("🚀 Run Iterative Pipeline", type="primary", width='stretch')


def _current_config() -> dict:
    """The sidebar as an engine config."""
    return L.default_config(
        seed=int(seed), n_bumps=int(n_bumps), noise_std=float(noise_std),
        n_pool_total=int(n_pool_total), shift_strength=float(shift_strength),
        shift_center_x=float(scx), shift_center_y=float(scy),
        shift_spread=float(shift_spread),
        radius=float(radius) if induce_ws else 0.0,
        center_x=float(center_x), center_y=float(center_y),
        model_name=model_name, complexity=float(complexity), arch=arch,
        activation=activation, solver=solver, alpha=float(alpha_l2),
        batch_size=int(batch_size), lr_init=float(lr_init),
        iters_initial=int(iters_initial), n_train=int(n_train), n_eval=int(n_eval),
        grid_res=int(grid_res), extract_q=float(extract_q),
        detector=detector, detect_mode=detect_mode,
        sel_method=sel_method, sel_mode=sel_mode,
        sel_sigma=float(sel_sigma), sigma_schedule=sigma_schedule,
        sigma_rate=float(sigma_rate), mix_ratio=float(mix_ratio),
        mix_schedule=mix_schedule, mix_rate=float(mix_rate),
        n_select=int(n_select), n_candidate=int(n_candidate), pool_mode=pool_mode,
        warm_start=training_mode.startswith("Warm"), early_stopping=bool(early_stop),
        iters_retrain=int(iters_retrain), n_iterations=int(n_iterations),
        lr_schedule=lr_schedule, lr_gamma=float(lr_gamma), lr_min=float(lr_min),
        lr_step=int(lr_step), regimes=list(regimes) or ["accumulative"],
        single_shot=bool(single_shot), tie_model_seed=bool(tie_model_seed),
    )


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if run_btn or S.pop("_it_trigger", False):
    cfg = _current_config()
    if cfg["warm_start"] and AVAILABLE_MODELS[cfg["model_name"]] != "mlp":
        st.info(f"Warm-start is implemented for the MLP; **{cfg['model_name']}** "
                f"is rebuilt from scratch each iteration instead.")
    bar = st.progress(0.0, text="Iterating…")
    res = L.run_iterative(cfg, keep_rounds=True,
                          progress=lambda d, t, txt: bar.progress(d / max(t, 1), text=txt))
    bar.empty()
    S["iter2"] = res

if "iter2" not in S:
    st.info("Configure the sidebar and press **🚀 Run Iterative Pipeline** — or open "
            "the **🧪 Parameter Sweep** tab to run a whole grid.")

R = S.get("iter2")

tabs = st.tabs(["① Setup", "📉 Trajectory", "🔬 Round Explorer",
                "🎛️ Schedules & Detection", "🧾 Summary", "🧪 Parameter Sweep"])

# ─────────────────────────────────────────────────────────────
# ① SETUP
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    if R is None:
        st.info("Run the pipeline to populate this tab.")
    else:
        su, cfg = R["setup"], R["cfg"]
        st.subheader("Setup — landscape, induced gap and initial training set")
        st.caption(
            "The target is a sum of Gaussian bumps on the unit square. A weakspot is "
            "induced by withholding every training point inside the red circle; noise "
            "is added to training and selected labels only, so the evaluation error "
            "measures genuine model failure rather than label noise.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_landscape(su["xx"], su["yy"], su["true_grid"],
                                       su["center"], su["radius"]), width='stretch')
        c2.plotly_chart(plot_data_overview(
            su["X_keep"], P.true_function(su["X_keep"], n_bumps=cfg["n_bumps"]),
            su["X_excl"], su["center"], su["radius"], "x₁", "x₂", "f(x₁,x₂)",
            excl_mode="Circle (radius)"), width='stretch')
        st.plotly_chart(plot_training_points(
            su["X_tr0"], su["y_tr0"], None, su["center"], su["radius"],
            title="Initial training set (the gap is left empty)"), width='stretch')
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Initial training points", len(su["X_tr0"]))
        m2.metric("Withheld by the gap", len(su["X_excl"]))
        m3.metric("Model parameters", f"{su['n_params']:,}")
        m4.metric("Training mode",
                  "warm-start" if su["warm"] else "from scratch")

# ─────────────────────────────────────────────────────────────
# 📉 TRAJECTORY
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    if R is None:
        st.info("Run the pipeline to populate this tab.")
    else:
        H, cfg = R["tracks"], R["cfg"]
        iters = R["iters"]
        st.subheader("Training progress across the loop")
        st.caption(
            "**Colour = strategy** (green guided, red random); **line style = regime** "
            "(solid accumulative, dotted new-only, dashed scaled-accum). Iteration 0 is "
            "the shared initial model. Lower is better. Guided and random draw from the "
            "*same* candidate pool each round, so the only difference between a green "
            "and a red curve is the selection rule.")
        s1, s2 = st.columns([1.4, 1])
        sm = s1.selectbox("Trend smoothing", list(IP.SMOOTHERS), index=0,
                          help="Cosmetic only — raw values stay visible as faint dots, "
                               "and the Summary table and CSV always use raw numbers.")
        win = s2.slider("Smoothing window / span", 2, max(3, len(iters) - 1), 3,
                        disabled=(sm == "None"))

        refs = None
        if R.get("single_shot"):
            ss = R["single_shot"]
            refs = {"single-shot guided": (ss["guided"]["mae"], "#2ca02c"),
                    "single-shot random": (ss["random"]["mae"], "#d62728")}
        st.plotly_chart(IP.trend_figure(
            iters, H, "mae", "MAE",
            "Whole-area evaluation MAE per iteration", sm, win, refs), width='stretch')
        if refs:
            st.caption(
                "The dash-dot lines are the **budget- and compute-matched single-shot** "
                "controls: the same K × n_select points added in one round with K × the "
                "retraining budget. A trajectory that ends below its own control is "
                "evidence that *staging* the budget — not merely spending it — is what "
                "helps.")

        if cfg["radius"] > 0:
            st.markdown("---")
            st.subheader("Local repair vs collateral damage")
            st.caption(
                "The two halves of the paper's central tension, measured directly. "
                "**Left:** error inside the induced gap — what guidance is *for*. "
                "**Right:** error everywhere else — what over-concentration costs. "
                "The single-round study could only infer this trade-off from the "
                "whole-area metric; here it is separated and tracked round by round.")
            g1, g2 = st.columns(2)
            g1.plotly_chart(IP.trend_figure(iters, H, "err_in", "error in weakspot",
                                            "Inside the induced gap", sm, win),
                            width='stretch')
            g2.plotly_chart(IP.trend_figure(iters, H, "err_out", "error outside weakspot",
                                            "Outside the gap (forgetting)", sm, win),
                            width='stretch')

        st.markdown("---")
        st.subheader("Head-to-head: guided − random, per iteration")
        st.caption(
            "The quantity the companion paper reported as a single number for one "
            "round, now as a trajectory. Below zero = guided ahead. Whether an early "
            "advantage survives to the end of the loop, or is competed away as the "
            "random baseline accumulates coverage, is the question this chart answers.")
        st.plotly_chart(IP.gap_figure(iters, H, "mae",
                                      "Whole-area MAE advantage", sm, win),
                        width='stretch')
        if cfg["radius"] > 0:
            st.plotly_chart(IP.gap_figure(iters, H, "err_in",
                                          "In-weakspot advantage", sm, win),
                            width='stretch')

# ─────────────────────────────────────────────────────────────
# 🔬 ROUND EXPLORER
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    if R is None or not R["rounds"]:
        st.info("Run the pipeline to populate this tab.")
    else:
        su, cfg = R["setup"], R["cfg"]
        st.subheader("What each round actually selected")
        st.caption(
            "Pick a round. **Left:** the guided selection, with the candidate pool "
            "coloured by the selection weight and the σ rings drawn on the *detected* "
            "centre. **Right:** the random baseline drawn from the same pool. Both "
            "selections feed every regime — appended to the accumulative set and, "
            "separately, used alone by the new-only model. The dashed red circle is "
            "the induced gap.")
        it_sel = st.slider("Iteration", 1, len(R["rounds"]), 1, key="it_round_pick")
        rd = R["rounds"][it_sel - 1]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("α this round", f"{rd['mix']:.3f}")
        k2.metric("σ this round", f"{rd['sigma']:.3f}")
        k3.metric("learning rate", f"{rd['lr']:.2e}")
        k4.metric("weakspot severity", f"{rd['severity']:.2f}",
                  help="Mean error inside the detected region ÷ mean error outside, "
                       "computed from the model's own error distribution. 1.0 means "
                       "nothing stands out any more.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Guided** — *{cfg['sel_method']}* via *{cfg['detector']}*, "
                        f"{len(rd['Xsel_g'])} points added "
                        f"(accumulative set now {rd['n_train_acc']})")
            st.plotly_chart(plot_selection(
                rd["Xc_g"], rd["w_g"], rd["Xsel_g"], rd["c_g"], rd["sigma"],
                su["center"], su["radius"],
                title=f"Iteration {it_sel}: guided selection"),
                width='stretch', key=f"itg_{it_sel}")
        with c2:
            st.markdown(f"**Random baseline** — uniform from the same pool, "
                        f"{len(rd['Xsel_r'])} points added")
            st.plotly_chart(plot_selection(
                rd["Xc_r"], None, rd["Xsel_r"], None, None, su["center"], su["radius"],
                show_kernel=False, title=f"Iteration {it_sel}: random selection"),
                width='stretch', key=f"itr_{it_sel}")
        st.markdown(f"**Model error entering round {it_sel}** — the field the detector "
                    f"searched for the weakspot:")
        st.plotly_chart(plot_ground_truth_error(su["X_eval"], rd["err"],
                                                su["center"], su["radius"]),
                        width='stretch', key=f"iterr_{it_sel}")

# ─────────────────────────────────────────────────────────────
# 🎛️ SCHEDULES & DETECTION
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    if R is None:
        st.info("Run the pipeline to populate this tab.")
    else:
        su = R["setup"]
        st.subheader("The controls, as actually applied")
        st.caption(
            "The paper identified the rehearsal mix α and the kernel width σ as the "
            "two dominant factors, ahead of the model state and far ahead of anything "
            "describing the task. In a loop they are no longer constants. This chart "
            "shows what each round really used, with the learning rate on its own log "
            "axis and the detected weakspot severity overlaid — the signal an adaptive "
            "schedule reacts to.")
        st.plotly_chart(IP.schedule_figure(R["sched"]), width='stretch')

        st.markdown("---")
        st.subheader("Does the loop keep finding the weakspot?")
        st.caption(
            "Detection quality per round, and how far the detected centre moved from "
            "the previous one. As the gap heals the detector should legitimately lose "
            "it — distance and drift rise, IoU falls — because the weakest region is "
            "genuinely somewhere else by then. Distinguishing that healthy migration "
            "from a detector simply failing is what the right-hand trajectory is for.")
        d1, d2 = st.columns([1.2, 1])
        d1.plotly_chart(IP.detection_figure(R["det"]), width='stretch')
        d2.plotly_chart(IP.trajectory_figure(R["det"], su["center"], su["radius"]),
                        width='stretch')

# ─────────────────────────────────────────────────────────────
# 🧾 SUMMARY
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    if R is None:
        st.info("Run the pipeline to populate this tab.")
    else:
        H, cfg, iters = R["tracks"], R["cfg"], R["iters"]
        summ = L.summarise(R)
        st.subheader("Iteration-by-iteration results")
        cols = {"iteration": iters}
        for t in R["active"]:
            strat, regime = L.TRACKS[t]
            cols[f"{strat}_{regime}_MAE"] = np.round(H[t]["mae"], 4)
        if cfg["radius"] > 0:
            cols["guided_err_in"] = np.round(H["gacc"]["err_in"], 4)
            cols["random_err_in"] = np.round(H["racc"]["err_in"], 4)
            cols["guided_err_out"] = np.round(H["gacc"]["err_out"], 4)
            cols["random_err_out"] = np.round(H["racc"]["err_out"], 4)
        cols["guided_train_pts"] = H["gacc"]["n_train"]
        df = pd.DataFrame(cols)
        st.dataframe(df, width='stretch', hide_index=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final gap (guided − random)", f"{summ['final_gap_mae']:+.4f}",
                  help="Whole-area MAE at the last iteration. Negative = guided ahead.")
        m2.metric("Across-loop mean gap", f"{summ['auc_gap_mae']:+.4f}",
                  help="Mean of guided − random over iterations 1…K. More honest than "
                       "the final value when the curves cross.")
        m3.metric("Iterations guided led", f"{summ['win_rate_iters']*100:.0f}%")
        m4.metric("Best iteration (guided)", summ["best_iter_guided"],
                  help="Where guided accumulative reached its lowest MAE. Earlier than "
                       "K means the loop overshot and should have stopped.")

        verdict = ("**ahead of** the random baseline ✅" if summ["auc_gap_mae"] < 0
                   else "**behind** the random baseline ⚠️")
        lines = [
            f"Across **{cfg['n_iterations']}** iterations, guided selection finished "
            f"{verdict} on the across-loop average (Δ {summ['auc_gap_mae']:+.4f}), "
            f"leading in {summ['win_rate_iters']*100:.0f}% of rounds."
        ]
        if cfg["radius"] > 0:
            lines.append(
                f"- **In-weakspot** — final guided − random = "
                f"{summ['final_gap_err_in']:+.4f}. This is where guidance is supposed "
                f"to pay; a local win with a whole-area tie is the honest, defensible "
                f"claim, and matches what the single-round study reported."
            )
            lines.append(
                f"- **Forgetting** (error *outside* the gap, iteration K − iteration 0) "
                f"— guided **{summ['forgetting_guided']:+.4f}**, random "
                f"**{summ['forgetting_random']:+.4f}**. Positive means the model got "
                f"worse where it was already competent: the collapse the companion "
                f"paper attributed to over-concentration, now measured directly rather "
                f"than inferred."
            )
        if "gnew" in H and "gsca" in H:
            gsca, gnew = H["gsca"]["mae"][-1], H["gnew"]["mae"][-1]
            fair = ("**confirms the distribution matters** ✅" if gsca < gnew
                    else "does not separate from new-only here ⚠️")
            lines.append(
                f"- **Size-matched control** — guided scaled-accum **{gsca:.4f}** vs "
                f"guided new-only **{gnew:.4f}** ({gsca - gnew:+.4f}) → this {fair}. "
                f"Both train on the same number of points; only the distribution "
                f"differs, so a gap here is not explained by the amount of data."
            )
        if R.get("single_shot"):
            ss = R["single_shot"]
            stg = summ.get("staging_gain_guided", float("nan"))
            better = "better than" if stg > 0 else "no better than"
            lines.append(
                f"- **Staging** — the budget- and compute-matched single-shot guided "
                f"model reached **{ss['guided']['mae']:.4f}** against the loop's "
                f"**{H['gacc']['mae'][-1]:.4f}**, so iterating was {better} spending "
                f"the same budget at once (Δ {stg:+.4f}). The random single-shot "
                f"reference is **{ss['random']['mae']:.4f}**."
            )
        st.markdown("\n".join(lines))

        st.download_button("⬇ Download iteration history (CSV)",
                           data=df.to_csv(index=False).encode("utf-8"),
                           file_name="iterative_history.csv", mime="text/csv")
        with st.expander("Full configuration for this run"):
            st.json({k: (list(v) if isinstance(v, list) else v)
                     for k, v in cfg.items()})

# ─────────────────────────────────────────────────────────────
# 🧪 PARAMETER SWEEP
# ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Parameter sweep over the loop")
    st.caption(
        "Runs a whole grid of trajectories and appends **one row per (configuration × "
        "detector × iteration)** to this config's results CSV, which the "
        "**Iterative — Visualise Results** page reads. A configuration's rows are "
        "written together and keyed by how the configuration differs from the paper's "
        "operating point, so a run resumes exactly where it stopped and adding a new "
        "axis later never invalidates what is already collected.")

    cfg_names = SW.available_configs()
    cfg_choice = st.selectbox(
        "Sweep configuration", cfg_names,
        index=cfg_names.index(SW.SWEEP_CONFIG_NAME) if SW.SWEEP_CONFIG_NAME in cfg_names else 0,
        help="Grids live in `scripts/iterative/sweep_configs/<name>.json`. Each writes "
             "to its own `sweep__<name>.csv`. Add a JSON there to define a new study.")
    if cfg_choice != SW.SWEEP_CONFIG_NAME:
        SW.set_active_config(cfg_choice)
    st.info(f"**{SW.SWEEP_CONFIG_NAME}** → `{SW.CSV_PATH}`\n\n"
            f"{SW.ACTIVE_CONFIG.get('description', '')}")

    total_runs = prod(len(v) for v in SW.SWEEP_GRID.values())
    est_min = total_runs * SW.SECS_PER_RUN / 60
    swept = ", ".join(f"`{k}`×{len(v)}" for k, v in SW.SWEEP_GRID.items() if len(v) > 1)
    st.markdown(
        f"**Swept axes:** {swept or '— (single configuration)'}  \n"
        f"**Detectors:** {len(SW.SWEEP_DETECTORS)} "
        f"({', '.join(SW.SWEEP_DETECTORS)}) — each runs its own trajectory, since "
        f"detectors steer different selections and diverge after iteration 0.  \n"
        f"**Held at the sidebar values:** model=`{model_name}`, complexity="
        f"`{complexity}`, weakspot centre=({center_x}, {center_y}), n_eval=`{n_eval}`, "
        f"grid_res=`{grid_res}`, extract_q=`{extract_q}`, selection rule=`{sel_mode}`, "
        f"early stopping=`{early_stop}`, single-shot control=`{single_shot}`.")

    q1, q2, q3 = st.columns(3)
    q1.metric("Configurations", f"{total_runs:,}")
    q2.metric("Est. full sweep", f"{est_min:,.0f} min")
    q3.metric("Rows per config",
              f"~{len(SW.SWEEP_DETECTORS) * (int(n_iterations) + 1):,}")

    resume = st.checkbox(
        "Skip already-completed configurations (resume)", value=True,
        help="Scans the results CSV **only when the sweep is launched**, never on page "
             "render, so opening this tab stays fast even for large result files.")
    max_workers = os.cpu_count() or 4
    run_parallel = st.checkbox(
        "⚡ Run in parallel (multiple CPU cores)", value=False,
        help="Distributes configurations across cores with joblib. Workers only "
             "compute; the main process does every CSV write, so resume integrity is "
             "unaffected.")
    n_workers = st.slider("Worker processes", 1, max_workers, max(1, max_workers // 2),
                          disabled=not run_parallel) if run_parallel else 1
    if run_parallel and n_workers > 1:
        st.caption(f"With **{n_workers} workers** the ≈{est_min:,.0f} min estimate "
                   f"should fall to roughly **{est_min / n_workers:,.0f} min**.")
    go_sweep = st.button("▶ Run Sweep", type="primary", width='stretch',
                         disabled=total_runs == 0)

    fixed_params = dict(
        model_name=model_name, complexity=float(complexity),
        center_x=float(center_x), center_y=float(center_y), n_eval=int(n_eval),
        grid_res=int(grid_res), extract_q=float(extract_q), sel_mode=sel_mode,
        shift_center_x=float(scx), shift_center_y=float(scy),
        shift_spread=float(shift_spread), early_stopping=bool(early_stop),
        single_shot=bool(single_shot), tie_model_seed=bool(tie_model_seed),
    )

    if go_sweep:
        with st.spinner("Checking which configurations are already done…"):
            if resume:
                remaining = SW.remaining_combos(fixed_params)
            else:
                remaining = [{**c, **fixed_params} for c in SW.parameter_grid()]
        if not remaining:
            st.success(f"✓ All {total_runs:,} configurations are already in "
                       f"`{SW.CSV_PATH.name}` — nothing to run. Uncheck **resume** to "
                       f"recompute them.")
        else:
            n_total = len(remaining)
            bar = st.progress(0.0, text=f"0 / {n_total} — starting…")
            t0 = time.time()

            def _tick(done, elapsed):
                eta = (elapsed / max(done, 1)) * (n_total - done) / 60
                bar.progress(done / n_total,
                             text=f"{done}/{n_total} · "
                                  f"{'⚡ ' + str(n_workers) + ' workers' if run_parallel else 'sequential'}"
                                  f" · elapsed {elapsed:.0f}s · ETA {eta:.1f} min")

            if run_parallel and n_workers > 1:
                from joblib import Parallel, delayed, parallel_backend
                i, w, done = 0, n_workers, 0
                while i < n_total:
                    batch = max(w * 2, 1)
                    try:
                        with parallel_backend("loky", inner_max_num_threads=1):
                            with Parallel(n_jobs=w) as par:
                                while i < n_total:
                                    chunk = remaining[i:i + batch]
                                    for rows in par(delayed(SW.safe_run_one)(p)
                                                    for p in chunk):
                                        SW.append_rows(SW.CSV_PATH, rows)
                                    i += len(chunk); done += len(chunk)
                                    _tick(done, time.time() - t0)
                        break
                    except Exception as e:
                        new_w = max(1, w // 2)
                        st.warning(
                            f"Parallel workers were terminated ({type(e).__name__}, "
                            f"usually out-of-memory). Reducing workers **{w} → "
                            f"{new_w}** and continuing — completed configurations are "
                            f"saved and resumable.")
                        w = new_w
                        if w == 1:
                            for params in remaining[i:]:
                                SW.append_rows(SW.CSV_PATH, SW.safe_run_one(params))
                                i += 1; done += 1
                                _tick(done, time.time() - t0)
                            break
            else:
                for i, params in enumerate(remaining):
                    try:
                        SW.append_rows(SW.CSV_PATH, SW.run_one(params))
                    except Exception as e:
                        st.warning(f"Configuration {i + 1} failed: {e}")
                    _tick(i + 1, time.time() - t0)

            bar.empty()
            st.success(
                f"Sweep complete in {(time.time() - t0) / 60:.1f} min. Rows appended to "
                f"`{SW.CSV_PATH}`. Open **Iterative — Visualise Results** to explore.")
