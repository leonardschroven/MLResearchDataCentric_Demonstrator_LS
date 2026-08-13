"""Model architectures, optimiser settings and per-iteration schedules.

The single-round study held the regression model fixed: one MLP of one shape,
sklearn's default optimiser settings, one retraining budget. In a *repeated*
loop those choices stop being incidental. Each round continues training the same
weights on a training distribution that has just been shifted towards the weak
region, so the architecture (how much capacity is available to absorb the new
region without overwriting the old one) and the learning rate (how far the
weights are allowed to move per round) become first-class controls of the
forgetting/plasticity balance the companion paper identified as the binding
constraint.

This module therefore exposes three things:

``ARCHITECTURES``    named hidden-layer shapes, from a tiny single layer to deep
                     and bottleneck variants, plus the legacy complexity-scaled
                     ``(h, h)`` shape so previous results stay reproducible.
``lr_at``            the per-iteration learning rate under a chosen schedule
                     (constant, decays, cosine, warm restarts) — the analogue of
                     an epoch-wise LR schedule, one step per *loop iteration*.
``mix_at``/``sigma_at``  schedules for the two coverage controls the paper found
                     dominant (rehearsal mix α and kernel width σ), including an
                     *adaptive* variant that relaxes the focus as the detected
                     weakspot heals — the extension named in its future work.

Everything here is plain sklearn and numpy so the sweep can pickle it into
worker processes.
"""
from __future__ import annotations

import math

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.weakspot.models import AVAILABLE_MODELS, build_model  # noqa: F401

# ─────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────
# ``None`` = derive the shape from the ``complexity`` slider exactly as
# ``scripts.weakspot.models.build_model`` does, so the legacy default is
# bit-for-bit reproducible and single-round results remain comparable.
LEGACY_ARCH = "Complexity-scaled (2 × h)"

ARCHITECTURES: dict[str, tuple[int, ...] | None] = {
    LEGACY_ARCH:              None,
    "Tiny (1 × 16)":          (16,),
    "Small (1 × 64)":         (64,),
    "Standard (2 × 64)":      (64, 64),
    "Wide (2 × 256)":         (256, 256),
    "Deep (4 × 64)":          (64, 64, 64, 64),
    "Deep narrow (6 × 32)":   (32, 32, 32, 32, 32, 32),
    "Pyramid (128-64-32)":    (128, 64, 32),
    "Bottleneck (128-16-128)": (128, 16, 128),
}

ACTIVATIONS = ("relu", "tanh", "logistic")
SOLVERS = ("adam", "sgd")


def hidden_sizes(arch: str, complexity: float) -> tuple[int, ...]:
    """Hidden-layer shape for a named architecture.

    The legacy entry reproduces ``build_model``'s ``(h, h)`` with
    ``h = max(8, int(complexity * 128))``.
    """
    shape = ARCHITECTURES.get(arch, None)
    if shape is not None:
        return tuple(shape)
    h = max(8, int(float(complexity) * 128))
    return (h, h)


def n_parameters(arch: str, complexity: float, d_in: int = 2) -> int:
    """Weight+bias count for a shape — a cheap capacity axis for the analysis."""
    sizes = (d_in,) + hidden_sizes(arch, complexity) + (1,)
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def build_iter_model(model_key: str, *, arch: str = LEGACY_ARCH,
                     complexity: float = 0.5, iterations: int = 100,
                     warm_start: bool = False, early_stopping: bool = True,
                     lr_init: float = 1e-3, solver: str = "adam",
                     activation: str = "relu", alpha: float = 1e-4,
                     batch_size: int | str = "auto",
                     random_state: int = 42):
    """Build the regression model for one iterative run.

    For every non-MLP algorithm this delegates unchanged to
    ``scripts.weakspot.models.build_model`` (those have no warm-start path, so
    each iteration rebuilds them from scratch). For the MLP it builds the same
    ``StandardScaler → MLPRegressor`` pipeline but exposes the architecture and
    optimiser settings that the iterative study varies.

    With ``arch=LEGACY_ARCH``, ``lr_init=1e-3``, ``solver="adam"``,
    ``activation="relu"``, ``alpha=1e-4`` and ``batch_size="auto"`` the result is
    identical to ``build_model``'s MLP, so the paper's operating point is
    reproduced exactly.
    """
    if model_key != "mlp":
        return build_model(model_key, complexity=complexity, iterations=iterations,
                           random_state=random_state, warm_start=warm_start,
                           early_stopping=early_stopping)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=hidden_sizes(arch, complexity),
            activation=activation, solver=solver,
            alpha=float(alpha), batch_size=batch_size,
            learning_rate_init=float(lr_init),
            max_iter=int(iterations), random_state=int(random_state),
            early_stopping=bool(early_stopping), warm_start=bool(warm_start),
        )),
    ])


def apply_runtime(model, *, lr: float | None = None, epochs: int | None = None) -> None:
    """Set the learning rate / epoch budget on an already-built model.

    Used between loop iterations: sklearn rebuilds its optimiser on every
    ``fit`` call (``fit`` is not incremental even under ``warm_start``), so
    writing ``learning_rate_init`` here is what makes a per-iteration learning-
    rate schedule take effect on the *continued* weights. Silently does nothing
    for algorithms that have no such knob.
    """
    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if epochs is not None and hasattr(est, "max_iter"):
        est.max_iter = int(epochs)
    if lr is not None and hasattr(est, "learning_rate_init"):
        est.learning_rate_init = float(lr)


# ─────────────────────────────────────────────────────────────
# Per-iteration learning-rate schedule
# ─────────────────────────────────────────────────────────────
# One step per *loop iteration*, not per epoch: the question is how far the
# weights may travel in each round of continued training. A decaying rate lets
# the early rounds absorb the weak region and the late rounds consolidate, which
# is the natural counter to the drift the single-round study penalised.
LR_SCHEDULES = (
    "Constant",
    "Exponential decay",
    "Step decay",
    "Cosine anneal",
    "Warm restarts (SGDR)",
    "Inverse-time decay",
)


def lr_at(schedule: str, lr0: float, k: int, n_iterations: int,
          gamma: float = 0.7, lr_min: float = 1e-5, step: int = 2) -> float:
    """Learning rate for loop iteration ``k`` (0-based).

    ``gamma``  per-step multiplier (exponential/step) or decay rate (inverse).
    ``lr_min`` floor for the cosine schedules.
    ``step``   iterations per step (step decay) or per cycle (warm restarts).
    """
    lr0, k = float(lr0), max(int(k), 0)
    K = max(int(n_iterations) - 1, 1)
    if schedule == "Exponential decay":
        return max(lr0 * float(gamma) ** k, 0.0)
    if schedule == "Step decay":
        return max(lr0 * float(gamma) ** (k // max(int(step), 1)), 0.0)
    if schedule == "Cosine anneal":
        return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * k / K))
    if schedule == "Warm restarts (SGDR)":
        t = k % max(int(step), 1)
        T = max(int(step) - 1, 1)
        return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * t / T))
    if schedule == "Inverse-time decay":
        return lr0 / (1.0 + float(gamma) * k)
    return lr0                                              # "Constant"


# ─────────────────────────────────────────────────────────────
# Per-iteration coverage schedules (the paper's two dominant controls)
# ─────────────────────────────────────────────────────────────
MIX_SCHEDULES = (
    "Constant",
    "Linear decay",
    "Exponential decay",
    "Linear increase",
    "Adaptive (weakspot severity)",
)
SIGMA_SCHEDULES = (
    "Constant",
    "Linear widen",
    "Exponential widen",
    "Linear narrow",
    "Adaptive (weakspot severity)",
)

# Severity below this multiple of the ambient error counts as "healed": the
# detected region no longer stands out from the rest of the error surface, so an
# adaptive schedule hands the budget back to uniform rehearsal.
_HEALED = 1.0


def _relative_excess(severity: float, severity0: float) -> float:
    """How much of the *initial* weakspot severity is left, in [0, 1].

    ``severity`` is the model's mean error inside the detected region divided by
    its mean error outside — computed from the error distribution alone, with no
    access to the ground-truth gap, so an adaptive schedule stays implementable
    in the fixed-dataset setting the study targets.
    """
    e0 = max(float(severity0) - _HEALED, 1e-9)
    e = max(float(severity) - _HEALED, 0.0)
    return float(min(max(e / e0, 0.0), 1.0))


def mix_at(schedule: str, a0: float, k: int, n_iterations: int, rate: float = 0.5,
           severity: float = 1.0, severity0: float = 1.0) -> float:
    """Rehearsal mix α for loop iteration ``k`` (0-based), clipped to [0, 1].

    ``rate`` is the total fraction of α given up by the final iteration for the
    linear schedules and the per-iteration multiplier for the exponential one.
    The adaptive schedule scales α by how much of the initial weakspot severity
    survives, so focus relaxes towards pure rehearsal as the region heals.
    """
    a0, k = float(a0), max(int(k), 0)
    K = max(int(n_iterations) - 1, 1)
    if schedule == "Linear decay":
        v = a0 * (1.0 - float(rate) * k / K)
    elif schedule == "Exponential decay":
        v = a0 * float(rate) ** k
    elif schedule == "Linear increase":
        v = a0 + (1.0 - a0) * float(rate) * k / K
    elif schedule == "Adaptive (weakspot severity)":
        v = a0 * _relative_excess(severity, severity0)
    else:
        v = a0                                              # "Constant"
    return float(min(max(v, 0.0), 1.0))


def sigma_at(schedule: str, s0: float, k: int, n_iterations: int, rate: float = 0.5,
             severity: float = 1.0, severity0: float = 1.0) -> float:
    """Selection-kernel width σ for loop iteration ``k`` (0-based).

    ``rate`` is the total relative widening/narrowing reached at the final
    iteration (linear) or the per-iteration growth factor − 1 (exponential).
    The adaptive schedule widens the kernel as the weakspot heals, which is the
    kernel-side expression of the same "relax the focus" idea as ``mix_at``.
    """
    s0, k = float(s0), max(int(k), 0)
    K = max(int(n_iterations) - 1, 1)
    if schedule == "Linear widen":
        v = s0 * (1.0 + float(rate) * k / K)
    elif schedule == "Exponential widen":
        v = s0 * (1.0 + float(rate)) ** k
    elif schedule == "Linear narrow":
        v = s0 * (1.0 - float(rate) * k / K)
    elif schedule == "Adaptive (weakspot severity)":
        v = s0 * (1.0 + float(rate) * (1.0 - _relative_excess(severity, severity0)))
    else:
        v = s0                                              # "Constant"
    return float(max(v, 0.01))
