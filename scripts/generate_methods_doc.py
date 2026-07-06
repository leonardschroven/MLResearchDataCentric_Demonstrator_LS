"""
Generate a Word document describing every detection method in
``scripts/weakspot/detection.py`` — conceptual overview, math, algorithm,
and how it approaches weakspot identification.

Output: ``data/Weakspot_Detection_Methods.docx``
"""
from __future__ import annotations

from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


# ─────────────────────────────────────────────────────────────
# Content — structured as a list of method specs
# ─────────────────────────────────────────────────────────────

INTRO = """
This document describes every weakspot-detection method implemented in the
MS_Research_Demonstrator project. Each method receives the same three inputs
— test-point coordinates X ∈ [0,1]^N×2, absolute prediction errors e ∈ R^N,
and an evaluation grid G ∈ [0,1]^M×2 — and returns a continuous error-intensity
surface S : G → R≥0. A higher value of S(g) means the method considers the
location g more likely to be inside a weakspot. A separate stage-2 extractor
(see scripts/weakspot/extraction.py) thresholds the normalised surface,
isolates the largest connected component, and fits a 2σ ellipse to its
weighted spatial covariance — that ellipse is what gets compared against
the induced ground-truth weakspot via IoU.

The methods are organised in three tiers:

  A. Base methods (1–8) — each is a single statistical or learning model
     applied directly to (X, e).

  B. Simple ensembles (9–10) — arithmetic means of two unit-normalised
     base surfaces. Quick baselines.

  C. Advanced ensembles (11–17) — structurally-motivated combinations that
     exploit the complementary strengths of localiser methods (QR, EVT) and
     landscape-estimator methods (GPR, kNN). All assume the true weakspot
     centre is unknown at inference time.

Throughout, we use the following notation:

  N            number of test points
  M            number of grid pixels (R² for an R×R grid)
  x_i ∈ R²     i-th test-point location, normalised to [0,1]²
  e_i ∈ R≥0   absolute prediction error at x_i
  g ∈ R²       a grid pixel
  d(a,b)       Euclidean distance between two points
  S̃(g)        unit-normalised surface, i.e. (S(g) − min S) / (max S − min S)
  q_p(v)       p-th quantile of the vector v
""".strip()


def _math_block(*lines: str) -> tuple[str, str]:
    """Return a marker for a centred math block, joined by newlines."""
    return ("math", "\n".join(lines))


# Each section is a list of (tag, content). Tags are interpreted by `build_doc`.
#   ("h1"|"h2"|"h3", text)             — heading
#   ("p", text)                         — body paragraph
#   ("math", multi-line string)         — display equation block (monospace, centred)
#   ("code", multi-line string)         — algorithm pseudocode block
#   ("bullets", [str, ...])             — bullet list
#   ("strong", "key: value")            — bold-prefixed paragraph
#   ("hr",)                             — section separator

METHODS: list[dict] = [

    # ───────────────────────────── BASE METHODS ─────────────────────────────

    {
        "name": "1. Gaussian Process Regression (GPR)",
        "tier": "Base",
        "overview": (
            "Treats the error landscape as a sample from a Gaussian process "
            "prior. After conditioning on the observed (x_i, e_i) pairs, the "
            "posterior mean is read out as the predicted error surface."
        ),
        "math": [
            "Prior:    e(x) ~ GP( 0, k(x, x′) )",
            "",
            "Kernel:   k(x, x′) = Matérn_{ν=3/2}( ||x − x′|| ; ℓ ) + σ_n² · δ(x, x′)",
            "          = (1 + √3·d/ℓ) · exp(−√3·d/ℓ)   with   d = ||x − x′||",
            "",
            "Posterior mean at a grid pixel g:",
            "    μ(g) = k(g, X) · [K(X, X) + σ_n²·I]⁻¹ · e",
            "",
            "Surface:  S(g) = max( 0, μ(g) )",
        ],
        "algorithm": [
            "1. Sub-sample X, e to ≤ 300 points (computational cap).",
            "2. Fit kernel hyperparameters (ℓ, σ_n) by maximum log-marginal-likelihood.",
            "3. Compute Cholesky factor of K + σ_n²·I.",
            "4. Predict posterior mean μ(g) for every grid pixel.",
            "5. Clip negatives — error magnitudes are non-negative.",
        ],
        "approach": (
            "GPR's key property for weakspot detection is that it extrapolates "
            "smoothly into data-poor regions while quantifying its uncertainty "
            "there. Where the exclusion zone has removed training points, the "
            "posterior in that area is dominated by the surrounding (often "
            "high-error) neighbours plus a wider posterior σ — both of which "
            "are diagnostic of a weakspot."
        ),
        "strengths": [
            "Principled Bayesian uncertainty (the posterior σ is used by methods 8, 15).",
            "Smooth — no spurious local peaks.",
            "Extrapolates sensibly inside data holes.",
        ],
        "weaknesses": [
            "Over-smooths genuinely sharp weakspots.",
            "Kernel length-scale fitting can collapse with few or clustered points.",
            "O(N³) cost — hence the sub-sample.",
        ],
    },

    {
        "name": "2. Polynomial Response Surface",
        "tier": "Base",
        "overview": (
            "Fits a global polynomial of degree d (default d = 3) to the "
            "error values by ordinary least squares."
        ),
        "math": [
            "Feature map (degree d, 2D input):",
            "    Φ(x) = [ 1, x₁, x₂, x₁², x₁x₂, x₂², …, x₂^d ]   ∈ R^{(d+1)(d+2)/2}",
            "",
            "OLS fit:",
            "    β̂ = argmin_β   Σᵢ ( eᵢ − βᵀ Φ(xᵢ) )²",
            "       = ( Φ(X)ᵀ Φ(X) )⁻¹ Φ(X)ᵀ e",
            "",
            "Surface:  S(g) = max( 0, β̂ᵀ Φ(g) )",
        ],
        "algorithm": [
            "1. Build Vandermonde-style polynomial feature matrix Φ(X).",
            "2. Solve linear system for β̂.",
            "3. Evaluate β̂ᵀ Φ(g) on every grid pixel.",
            "4. Clip negatives.",
        ],
        "approach": (
            "Models the global error trend. If the underlying error landscape "
            "is well approximated by a low-order polynomial — for example a "
            "single broad weakspot near the centre of the domain — this gives "
            "a smooth, parameter-light estimate. For sharper or more localised "
            "weakspots the polynomial cannot bend tightly enough."
        ),
        "strengths": [
            "Fast and noise-robust on smooth fields.",
            "Provides a stable global background.",
        ],
        "weaknesses": [
            "Cannot represent sharp localised peaks.",
            "Biased outside the convex hull of the data (poles at corners).",
            "Degree must be picked a priori.",
        ],
    },

    {
        "name": "3. RBF Interpolation",
        "tier": "Base",
        "overview": (
            "Smooth interpolation: every training point contributes a radial "
            "basis function centred on itself; weights are chosen so the "
            "interpolant matches the error values (with a small smoothing "
            "parameter to control oscillation)."
        ),
        "math": [
            "Basis:    φ(r) = √( r² + c² )           (multiquadric, c = 1)",
            "",
            "Weights:  solve   ( Φ + λ·I ) w  =  e",
            "          where  Φᵢⱼ = φ( ||xᵢ − xⱼ|| ),  λ = 0.05",
            "",
            "Surface:  S(g) = max( 0,  Σᵢ wᵢ · φ( ||g − xᵢ|| ) )",
        ],
        "algorithm": [
            "1. Sub-sample X, e to ≤ 400 points.",
            "2. Build pairwise Φ matrix.",
            "3. Solve regularised linear system for w.",
            "4. Evaluate the weighted sum at every grid pixel.",
        ],
        "approach": (
            "Acts as a smooth interpolant — close to data points it reproduces "
            "the observed errors; between them it interpolates. For weakspots "
            "near or within the training cloud this gives an accurate picture; "
            "but, like kNN/LOESS, when training points are absent from the "
            "weakspot the RBF surface there is determined by surrounding "
            "neighbours and may understate the true error."
        ),
        "strengths": [
            "Sharp at data points, smooth between them.",
            "No model assumption beyond the kernel shape.",
        ],
        "weaknesses": [
            "Can ring (oscillate) when smoothing λ is too small.",
            "No statistical uncertainty.",
            "Sensitive to the basis-function family (multiquadric vs Gaussian etc.).",
        ],
    },

    {
        "name": "4. kNN Performance Mapping",
        "tier": "Base",
        "overview": (
            "For each grid pixel, look up its k nearest training points and "
            "return the distance-weighted average of their errors."
        ),
        "math": [
            "Let  N_k(g) = indices of the k training points closest to g.",
            "",
            "    wᵢ(g) = 1 / d(g, xᵢ)         (i ∈ N_k(g))",
            "",
            "    S(g) = ( Σ wᵢ(g) · eᵢ ) / ( Σ wᵢ(g) )",
            "",
            "Default k = 10.",
        ],
        "algorithm": [
            "1. Build a KD-tree on X.",
            "2. For each grid pixel g, query k nearest neighbours.",
            "3. Return the inverse-distance-weighted mean of their errors.",
        ],
        "approach": (
            "kNN reflects local average error. In well-sampled regions it "
            "tracks the true error closely. Crucially, it has a structural "
            "blind spot: inside a data hole (the weakspot), its 'nearest "
            "neighbours' are at the perimeter of the hole, so the weighted "
            "average there is determined by the surrounding ring — which "
            "tends to under-state the true elevated error inside the hole. "
            "Several advanced ensembles below explicitly exploit this "
            "blindness as a signal."
        ),
        "strengths": [
            "Sharp local resolution where data is dense.",
            "Cheap, easy to tune.",
            "No model — robust to landscape shape.",
        ],
        "weaknesses": [
            "Goes blind inside data holes (the exact place we care about).",
            "Discontinuous derivatives.",
            "k must be picked.",
        ],
    },

    {
        "name": "5. LOESS Local Regression",
        "tier": "Base",
        "overview": (
            "Locally weighted linear regression. For each grid pixel, a small "
            "linear model is fit on its tricubic-weighted neighbourhood, and "
            "the prediction at the pixel is read out."
        ),
        "math": [
            "For a grid pixel g, adaptive bandwidth:",
            "    h(g) = distance from g to its k-th nearest training point",
            "           k = ⌈bandwidth_frac · N⌉,  default 0.30",
            "",
            "Tricubic weights:",
            "    u_i = clip( d(g, xᵢ) / h(g),  0, 1 )",
            "    w_i = ( 1 − u_i³ )³",
            "",
            "Local linear fit  (X_aug = [1, x₁, x₂]):",
            "    β(g) = ( X_augᵀ W X_aug + ε·I )⁻¹  X_augᵀ W e",
            "",
            "    S(g) = max( 0,  [1, g₁, g₂] · β(g) )",
        ],
        "algorithm": [
            "1. Sub-sample X, e to ≤ 400 points.",
            "2. Compute pairwise distances grid ↔ training.",
            "3. For each grid pixel: compute weights, solve weighted normal equations, predict.",
        ],
        "approach": (
            "LOESS is to kNN what a local linear fit is to a local average. "
            "It captures gradient information, so at the boundary of a "
            "weakspot it can correctly slope down rather than flatly average. "
            "It shares kNN's blindness inside data holes, but degrades more "
            "gracefully at edges."
        ),
        "strengths": [
            "Captures local gradients (kNN cannot).",
            "Adaptive bandwidth — handles uneven sampling.",
        ],
        "weaknesses": [
            "Hole-blindness inherited from local averaging.",
            "Costly — solves a 3×3 system per grid pixel.",
        ],
    },

    {
        "name": "6. Quantile Regression",
        "tier": "Base",
        "overview": (
            "Polynomial quantile regression: instead of fitting the conditional "
            "mean of error, fit a high conditional quantile (τ = 0.9). The "
            "resulting surface is an upper envelope of the error field."
        ),
        "math": [
            "Pinball (check) loss for quantile τ:",
            "    ρ_τ(u) = u · ( τ − 𝟙[u < 0] )",
            "",
            "Fit:  β̂_τ = argmin_β  Σᵢ  ρ_τ( eᵢ − βᵀ Φ(xᵢ) )  +  α·||β||₂²",
            "",
            "Feature map Φ(·) is a degree-2 polynomial; α = 0.01.",
            "",
            "Surface:  S(g) = max( 0, β̂_τᵀ Φ(g) )",
        ],
        "algorithm": [
            "1. Build polynomial features Φ(X).",
            "2. Solve the regularised linear-programming formulation of "
            "quantile regression (HiGHS solver).",
            "3. Evaluate β̂ᵀ Φ on the grid.",
        ],
        "approach": (
            "Where ordinary regression smears error over an averaged surface, "
            "quantile regression at τ = 0.9 keeps only the upper-tail signal "
            "— it answers 'where is error systematically large?' rather than "
            "'where is error on-average elevated?'. This makes its argmax a "
            "good anchor for the weakspot centre, but the low-degree polynomial "
            "basis means it cannot reproduce the shape finely. Hence it is "
            "used as a localiser in the advanced ensembles."
        ),
        "strengths": [
            "Naturally locates the centre of a systematic-error region.",
            "Robust to outliers in the lower tail.",
        ],
        "weaknesses": [
            "Polynomial basis is too rigid for sharp shapes.",
            "Edge artefacts on degree ≥ 2 at the corners of [0,1]².",
        ],
    },

    {
        "name": "7. Peaks over Threshold (EVT)",
        "tier": "Base",
        "overview": (
            "Extreme-value theory approach: select training points whose error "
            "is above a high threshold, then fit a kernel density to their "
            "spatial locations. The KDE is read out as the surface."
        ),
        "math": [
            "Threshold:   t = q_{0.85}( e )       (85th percentile of errors)",
            "",
            "Extreme set: X_ext = { xᵢ : eᵢ ≥ t }",
            "",
            "Bandwidth:   h = max( 0.05,  0.5 / √( |X_ext| ) )",
            "",
            "Kernel density:",
            "    S(g) = ( 1 / |X_ext| ) · Σ_{x ∈ X_ext}  K_h( g − x )",
            "    where K_h is the Gaussian kernel with bandwidth h.",
        ],
        "algorithm": [
            "1. Threshold errors at the 85th percentile.",
            "2. Collect the spatial locations of the survivors.",
            "3. Fit a Gaussian KDE on those locations.",
            "4. Score the KDE on the grid.",
        ],
        "approach": (
            "Models the spatial concentration of extremes as a point process. "
            "Where the worst errors physically cluster, the KDE produces a "
            "sharp blob — an excellent anchor for the weakspot centre. By "
            "design it ignores the baseline error landscape entirely, so its "
            "surface is unimodal and shape-poor. Pairs naturally with a "
            "landscape estimator (see methods 14)."
        ),
        "strengths": [
            "Very sharp localisation of where the worst errors lie.",
            "Simple, robust.",
        ],
        "weaknesses": [
            "Loses all baseline-landscape information.",
            "Unimodal KDE blob — cannot represent extended or multi-lobe weakspots.",
        ],
    },

    {
        "name": "8. Bayesian Optimization (Expected Improvement)",
        "tier": "Base",
        "overview": (
            "Borrows the acquisition-function machinery from Bayesian "
            "optimisation. Fits a GPR surrogate to the error, then highlights "
            "regions where a BO search seeking maximum error would invest "
            "its next query."
        ),
        "math": [
            "Fit GPR on (X, e) → posterior  μ(g), σ(g).",
            "",
            "Best-so-far:  f* = max_i  eᵢ",
            "",
            "Z(g) = ( μ(g) − f* ) / σ(g)",
            "",
            "Expected Improvement:",
            "    EI(g) = ( μ(g) − f* ) · Φ_n( Z(g) )  +  σ(g) · φ_n( Z(g) )",
            "    where Φ_n, φ_n are the standard normal CDF and pdf.",
            "",
            "Surface:  S(g) = max( 0, EI(g) )",
        ],
        "algorithm": [
            "1. Same GPR fit as method 1 but with Matérn ν = 5/2.",
            "2. Compute the EI formula for every grid pixel.",
        ],
        "approach": (
            "EI rewards points with high posterior mean and/or high posterior "
            "uncertainty. As a weakspot detector it is heuristic — it answers "
            "'where would I sample next' rather than 'where is the worst "
            "error?'. In practice it behaves like a smoothed combination of "
            "the GPR mean and the GPR variance, with both peaks tending to "
            "coincide with the weakspot."
        ),
        "strengths": [
            "Combines mean and uncertainty in a single number.",
            "Sensitive to data holes via σ.",
        ],
        "weaknesses": [
            "Not designed for the task — borrowed from BO sequential search.",
            "Inherits all GPR fragilities.",
        ],
    },

    # ─────────────────────────── SIMPLE ENSEMBLES ───────────────────────────

    {
        "name": "9. kNN + GPR (mean)",
        "tier": "Simple Ensemble",
        "overview": (
            "Arithmetic mean of unit-normalised kNN and GPR surfaces."
        ),
        "math": [
            "S̃_kNN(g) = unit-normalise( kNN(g) )",
            "S̃_GPR(g) = unit-normalise( GPR(g) )",
            "",
            "S(g) = ½ · ( S̃_kNN(g) + S̃_GPR(g) )",
        ],
        "algorithm": [
            "1. Run kNN; unit-normalise.",
            "2. Run GPR; unit-normalise.",
            "3. Average pixel-wise.",
        ],
        "approach": (
            "Combines kNN's local sharpness where data is dense with GPR's "
            "principled extrapolation in data holes. Because the surfaces are "
            "first scaled to [0, 1], neither dominates by raw magnitude. "
            "A coarse but reliable baseline against which more sophisticated "
            "ensembles are compared."
        ),
        "strengths": [
            "Patches kNN's hole-blindness with GPR.",
            "Trivial to implement and interpret.",
        ],
        "weaknesses": [
            "Equal weighting — does not exploit where each method is reliable.",
            "Loses calibration of the original numerical scales.",
        ],
    },

    {
        "name": "10. kNN + Quantile (mean)",
        "tier": "Simple Ensemble",
        "overview": (
            "Arithmetic mean of unit-normalised kNN and Quantile-Regression "
            "surfaces. Combines average-error and tail-error signals."
        ),
        "math": [
            "S(g) = ½ · ( unit-normalise(kNN(g))  +  unit-normalise(QR(g)) )",
        ],
        "algorithm": [
            "1. Run kNN; unit-normalise.",
            "2. Run QR;  unit-normalise.",
            "3. Average pixel-wise.",
        ],
        "approach": (
            "kNN reflects mean local error; QR reflects upper-tail behaviour. "
            "Averaging emphasises regions where both the central tendency "
            "and the tail of error are elevated — a stricter, more agreement-"
            "driven signal."
        ),
        "strengths": [
            "Combines two complementary statistics.",
        ],
        "weaknesses": [
            "kNN's hole-blindness is only half-mitigated by QR (still polynomial-rigid).",
        ],
    },

    # ────────────────────────── ADVANCED ENSEMBLES ──────────────────────────

    {
        "name": "11. Anchored GPR (QR prior)",
        "tier": "Advanced",
        "overview": (
            "GPR with a non-zero prior mean function set to the Quantile-"
            "Regression surface. The GPR posterior is fit to the *residuals* "
            "between observed errors and the QR prior, so it refines QR's "
            "shape rather than starting from zero."
        ),
        "math": [
            "1. Prior mean:  m₀(x) = QR-surface(x)   (fit once on X)",
            "",
            "2. Residuals:    rᵢ = eᵢ − m₀(xᵢ)",
            "",
            "3. Residual GPR: r(x) ~ GP( 0, k(x, x′) )   fit to (X, r)",
            "",
            "4. Composite posterior mean on the grid:",
            "    S(g) = max( 0,  m₀(g)  +  μ_residual(g) )",
        ],
        "algorithm": [
            "1. Run QR; predict on both the training points and the grid in "
            "one call to avoid a refit.",
            "2. Compute residuals at the training points.",
            "3. Fit a Matérn-3/2 GPR to the residuals.",
            "4. Sum QR-on-grid and residual posterior mean.",
        ],
        "approach": (
            "This is the textbook Bayesian way to inject prior structural "
            "knowledge into a GP. QR's polynomial surface anchors the centre "
            "and rough envelope; the residual GPR is then free to bend the "
            "shape locally to match the actual error field. Because the GPR "
            "operates on residuals, it can produce a surface that is smaller, "
            "sharper, or off-centre relative to QR alone — driven by the data, "
            "not by assumption."
        ),
        "strengths": [
            "Principled Bayesian formulation.",
            "QR provides location stability; GPR provides shape flexibility.",
            "Surface adapts to weakspots that QR alone would over-smooth.",
        ],
        "weaknesses": [
            "Computational cost ≈ QR + GPR.",
            "Sensitive to QR's quantile choice (τ = 0.9 by default).",
        ],
    },

    {
        "name": "12. Gated GPR (QR anchor)",
        "tier": "Advanced",
        "overview": (
            "Multiplicative gating: GPR's surface is multiplied by a Gaussian "
            "mask centred on QR's argmax. Sharper than averaging — landscape "
            "signal far from the QR anchor is suppressed exponentially."
        ),
        "math": [
            "1. Anchor:  c* = argmax_g  QR(g)",
            "",
            "2. Spread:  τ = √( weighted mean of d(g, c*)² over the top-15% "
            "QR pixels )",
            "            clamped to [0.05, 0.35]",
            "",
            "3. Gate:    G(g) = exp(  − d(g, c*)² / ( 2 τ² )  )",
            "",
            "4. Output:  S(g) = unit-normalise( GPR(g) )  ·  G(g)",
        ],
        "algorithm": [
            "1. Run QR; locate its argmax and extract a characteristic spread "
            "from the high-QR pixel cluster.",
            "2. Run GPR; unit-normalise.",
            "3. Build the Gaussian gate and multiply.",
        ],
        "approach": (
            "Where Anchored GPR is additive, Gated GPR is multiplicative — "
            "the gate hard-suppresses anything far from the QR anchor while "
            "leaving the GPR landscape untouched near it. This produces very "
            "tight, sharply-defined surfaces. The risk is that the gate is "
            "too narrow and clips real weakspot extent; the adaptive τ "
            "(derived from QR's own spread) mitigates this."
        ),
        "strengths": [
            "Visually tight surfaces — extraction ellipse hugs the weakspot.",
            "Computationally cheap.",
        ],
        "weaknesses": [
            "Gate bandwidth depends on QR's spread estimate, which can be wide.",
            "Multiplicative form can zero out small but real off-centre lobes.",
        ],
    },

    {
        "name": "13. QR × GPR (geometric mean)",
        "tier": "Advanced",
        "overview": (
            "Pixel-wise geometric mean of the unit-normalised QR and GPR "
            "surfaces. A consensus filter: only pixels rated high by both "
            "methods survive."
        ),
        "math": [
            "S(g) = √(  S̃_QR(g)  ·  S̃_GPR(g)  )",
            "",
            "Equivalent log form:",
            "S(g) = exp(  ½ · ( log S̃_QR(g) + log S̃_GPR(g) )  )",
        ],
        "algorithm": [
            "1. Run QR; unit-normalise.",
            "2. Run GPR; unit-normalise.",
            "3. Multiply and square-root pixel-wise.",
        ],
        "approach": (
            "The geometric mean penalises disagreement — a pixel with values "
            "(1, 0) averages to 0 instead of 0.5. So a pixel survives only if "
            "*both* the localiser (QR) and the landscape estimator (GPR) flag "
            "it. This produces conservative, agreement-driven surfaces that "
            "are well-suited to extraction: false-positive lobes from either "
            "method tend to be cancelled."
        ),
        "strengths": [
            "Symmetric, no anchor needed.",
            "Naturally conservative — fewer false positives.",
        ],
        "weaknesses": [
            "A single method going to zero kills the consensus, "
            "even if the other is very confident.",
        ],
    },

    {
        "name": "14. EVT × GPR (geometric mean)",
        "tier": "Advanced",
        "overview": (
            "Same construction as method 13 but with the EVT KDE as the "
            "localiser instead of QR. EVT is sharper than QR, so the "
            "consensus is tighter."
        ),
        "math": [
            "S(g) = √(  S̃_EVT(g)  ·  S̃_GPR(g)  )",
        ],
        "algorithm": [
            "1. Run EVT (KDE of extreme-error locations); unit-normalise.",
            "2. Run GPR; unit-normalise.",
            "3. Geometric mean pixel-wise.",
        ],
        "approach": (
            "EVT provides a sharp KDE blob over the locations of extreme "
            "errors; GPR provides the surrounding landscape. Their product "
            "is high only where the extreme-error point process is dense and "
            "the GPR mean is also elevated — a doubly-confirmed weakspot. "
            "Typically tighter than QR × GPR but slightly more sensitive to "
            "EVT's threshold and KDE bandwidth."
        ),
        "strengths": [
            "Tighter than QR × GPR.",
            "EVT naturally captures multi-lobe extreme structure (KDE).",
        ],
        "weaknesses": [
            "Inherits EVT's sensitivity to threshold and bandwidth.",
            "Can shrink to almost nothing if either factor is sparse.",
        ],
    },

    {
        "name": "15. GPR / kNN (variance-weighted mixture)",
        "tier": "Advanced",
        "overview": (
            "A convex combination of GPR and kNN, with the mixing weight "
            "driven by GPR's own posterior standard deviation. Where σ_GPR "
            "is high (sparse data → likely weakspot interior), the mixture "
            "leans on GPR; where σ_GPR is low (dense data), it leans on "
            "kNN's sharper local average."
        ),
        "math": [
            "1. Run GPR; obtain posterior (μ(g), σ(g)).",
            "",
            "2. Robust scaling of σ:",
            "    σ̄  = median_g  σ(g)",
            "    Δ  = IQR_g    σ(g)       (i.e. q₀.₇₅ − q₀.₂₅)",
            "",
            "3. Sigmoid weight:",
            "    α(g) = 1 / ( 1 + exp( − ( σ(g) − σ̄ ) / Δ ) )",
            "",
            "4. Mixture:",
            "    S(g) = α(g) · S̃_GPR(g)  +  ( 1 − α(g) ) · S̃_kNN(g)",
        ],
        "algorithm": [
            "1. Fit GPR; predict (μ, σ) on the grid.",
            "2. Run kNN; unit-normalise.",
            "3. Compute the sigmoid weight α from σ.",
            "4. Mix pixel-wise.",
        ],
        "approach": (
            "Directly attacks kNN's hole-blindness. In data-dense regions kNN "
            "is sharper and more accurate than GPR — so we trust it. In data "
            "holes the GPR posterior variance balloons (no nearby evidence), "
            "and we automatically switch over to GPR's smoothed extrapolation. "
            "The sigmoid centred at the median σ with width = IQR gives a "
            "data-adaptive transition with no free hyperparameters."
        ),
        "strengths": [
            "Adaptive — each region uses the right method.",
            "No hand-tuned switching threshold.",
            "Mathematically clean.",
        ],
        "weaknesses": [
            "Cost = GPR + kNN per run.",
            "GPR's σ can be miscalibrated when the kernel hyperparameters "
            "fit poorly.",
        ],
    },

    {
        "name": "16. GPR / kNN (disagreement-amplified)",
        "tier": "Advanced",
        "overview": (
            "Adds an asymmetric clip of the GPR-vs-kNN gap to the GPR "
            "surface, amplifying exactly the data-hole signature: GPR "
            "predicting high, kNN dropping low."
        ),
        "math": [
            "Δ(g) = max( 0,  S̃_GPR(g) − S̃_kNN(g) )",
            "",
            "S(g) = unit-normalise(  S̃_GPR(g)  +  λ · Δ(g)  )",
            "",
            "Default λ = 0.7.",
        ],
        "algorithm": [
            "1. Run GPR; unit-normalise.",
            "2. Run kNN; unit-normalise.",
            "3. Compute the clipped gap.",
            "4. Add λ-weighted gap to GPR; unit-normalise.",
        ],
        "approach": (
            "Data holes have a characteristic signature: GPR predicts an "
            "elevated value (it extrapolates from the high-error ring around "
            "the hole), while kNN drops (its 'nearest neighbours' are at the "
            "perimeter, where errors may be more moderate). The asymmetric "
            "max() ensures that only this direction of disagreement is "
            "amplified — symmetric noise where one is randomly higher than "
            "the other is left alone."
        ),
        "strengths": [
            "Selective: amplifies exactly the weakspot-suggestive disagreement.",
            "Single tunable parameter (λ).",
        ],
        "weaknesses": [
            "Assumes GPR's posterior is well-calibrated relative to kNN.",
            "λ is currently fixed at 0.7; tuning may help.",
        ],
    },

    {
        "name": "17. Local GPR (QR-localised)",
        "tier": "Advanced",
        "overview": (
            "Two-stage procedure. Stage A uses QR to localise the candidate "
            "weakspot region. Stage B refits a fresh GPR using only the "
            "training points inside that region, with a tighter length-scale "
            "prior so the kernel adapts to local geometry rather than the "
            "global mean spacing."
        ),
        "math": [
            "Stage A — localiser:",
            "    Run QR.  Let  c* = argmax_g QR(g),  τ = QR-spread.",
            "    Region radius  R = clip( 3·τ,  0.10,  0.35 ).",
            "",
            "Stage B — local GPR:",
            "    X_local = { xᵢ : d(xᵢ, c*) ≤ R }",
            "    If |X_local| < 20:  fall back to global GPR.",
            "    Else: fit GPR on (X_local, e_local) with",
            "          length-scale bounds  ℓ ∈ [10⁻³, 5].",
            "    Predict posterior mean on the full grid.",
            "",
            "Surface:  S(g) = max( 0, μ_local(g) ).",
        ],
        "algorithm": [
            "1. Run QR; extract (c*, τ).",
            "2. Compute the local subset of training points.",
            "3. Refit a GPR on that subset; tighter length-scale bounds.",
            "4. Predict on the full grid; clip.",
        ],
        "approach": (
            "Standard GPR fits a *single* length-scale to all of [0, 1]². If "
            "the weakspot is smaller than the global typical spacing, the "
            "fitted length-scale is too long and the weakspot is smoothed "
            "out. By localising the fit to points near the QR-suggested "
            "centre and tightening the length-scale prior, the kernel is "
            "free to discover a shorter, sharper scale tailored to the "
            "local geometry. The fall-back ensures stability when the "
            "candidate region is genuinely sparse."
        ),
        "strengths": [
            "Resolves weakspots that global GPR over-smooths.",
            "Adaptive: the second stage's kernel scale is data-driven.",
            "Graceful fallback when the local subset is too small.",
        ],
        "weaknesses": [
            "Sensitive to QR's localisation quality.",
            "Effectively runs GPR twice (once globally for QR-residual checks; "
            "we currently skip the global step but it's an option).",
            "Local subset must contain ≥ 20 points.",
        ],
    },
]


# ─────────────────────────────────────────────────────────────
# Doc builder
# ─────────────────────────────────────────────────────────────

OUT_PATH = Path("data/Weakspot_Detection_Methods.docx")


def _set_base_styles(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)


def _add_math(doc: Document, text: str) -> None:
    """Centred, monospaced block — used for equations and pseudocode."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(10)


def _add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)


def _add_strong(doc: Document, label: str, body: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run(label + ": ")
    r.bold = True
    p.add_run(body)


def _add_method(doc: Document, spec: dict) -> None:
    h = doc.add_heading(spec["name"], level=2)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    _add_strong(doc, "Tier", spec["tier"])

    doc.add_heading("Overview", level=3)
    doc.add_paragraph(spec["overview"])

    doc.add_heading("Mathematical formulation", level=3)
    _add_math(doc, "\n".join(spec["math"]))

    doc.add_heading("Algorithm", level=3)
    _add_bullets(doc, spec["algorithm"])

    doc.add_heading("How it approaches the weakspot problem", level=3)
    doc.add_paragraph(spec["approach"])

    doc.add_heading("Strengths", level=3)
    _add_bullets(doc, spec["strengths"])

    doc.add_heading("Weaknesses", level=3)
    _add_bullets(doc, spec["weaknesses"])


def build_doc() -> Path:
    doc = Document()
    _set_base_styles(doc)

    # Title
    title = doc.add_heading("Weakspot Detection Methods", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = subtitle.add_run("Mathematical reference for the 17 detectors "
                         "implemented in MS_Research_Demonstrator")
    r.italic = True
    r.font.size = Pt(12)

    doc.add_page_break()

    # Introduction
    doc.add_heading("1. Introduction & Notation", level=1)
    for para in INTRO.split("\n\n"):
        if any(para.lstrip().startswith(prefix) for prefix in
               ("A.", "B.", "C.", "N ", "M ", "x_i", "e_i", "g ",
                "d(a", "S̃", "q_p")):
            _add_math(doc, para)
        else:
            doc.add_paragraph(para)

    doc.add_heading("2. Stage-2 Extraction (common back-end)", level=1)
    doc.add_paragraph(
        "Every surface S returned by a detection method is fed into the same "
        "stage-2 extractor before being scored against the ground truth. "
        "The extractor turns the continuous surface into a parametric "
        "weakspot estimate: a centroid, per-axis sigmas, and an oriented "
        "2σ ellipse."
    )
    _add_math(doc,
        "1. Normalise S to [0, 1].\n"
        "2. Threshold:  M(g) = 1[ S(g) ≥ q_p(S) ]    (p = 0.85 by default)\n"
        "3. Connected components via 4-connectivity.\n"
        "4. Largest component, weighted by integrated mass Σ S(g) on M.\n"
        "5. Weighted centroid:\n"
        "       c = (1/W) Σ_{g∈M} S(g) · g           W = Σ S(g)\n"
        "6. Weighted covariance:\n"
        "       Σ = (1/W) Σ_{g∈M} S(g) · (g − c)(g − c)ᵀ\n"
        "7. Diagonal of Σ → per-axis sigmas (σ_x1, σ_x2).\n"
        "8. Eigendecomposition of Σ → (σ_major, σ_minor, angle).\n"
        "9. 2σ ellipse parameterised by (c, σ_major, σ_minor, angle).\n"
        "10. Scored vs the induced ground-truth region via IoU."
    )
    doc.add_paragraph(
        "Because every method shares this back-end, differences in IoU and "
        "centroid distance directly reflect differences in the upstream "
        "surface S — not in the parametric extraction step."
    )

    # Base methods
    doc.add_page_break()
    doc.add_heading("3. Base Methods", level=1)
    for m in METHODS:
        if m["tier"] == "Base":
            _add_method(doc, m)

    # Simple ensembles
    doc.add_page_break()
    doc.add_heading("4. Simple Ensembles", level=1)
    for m in METHODS:
        if m["tier"] == "Simple Ensemble":
            _add_method(doc, m)

    # Advanced ensembles
    doc.add_page_break()
    doc.add_heading("5. Advanced Ensembles", level=1)
    doc.add_paragraph(
        "Each advanced ensemble below is structurally motivated by the "
        "asymmetry observed empirically: localiser methods (QR, EVT) accurately "
        "find the weakspot centre but cannot model its shape, while landscape "
        "estimators (GPR, kNN, RBF, LOESS) approximate the shape well but "
        "either under-resolve the centre (GPR over-smooths) or go blind "
        "inside data holes (kNN, LOESS). The advanced ensembles combine "
        "these strengths without assuming any ground-truth information at "
        "inference time."
    )
    for m in METHODS:
        if m["tier"] == "Advanced":
            _add_method(doc, m)

    # Summary table
    doc.add_page_break()
    doc.add_heading("6. Summary table", level=1)
    table = doc.add_table(rows=1, cols=4)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, t in enumerate(["#", "Method", "Type", "Role"]):
        hdr[i].text = t
        for run in hdr[i].paragraphs[0].runs:
            run.bold = True

    roles = {
        "1. Gaussian Process Regression (GPR)": ("Base", "Landscape estimator (smooth, has σ)"),
        "2. Polynomial Response Surface":       ("Base", "Global trend"),
        "3. RBF Interpolation":                 ("Base", "Smooth interpolant"),
        "4. kNN Performance Mapping":           ("Base", "Local average (hole-blind)"),
        "5. LOESS Local Regression":            ("Base", "Local linear (hole-blind)"),
        "6. Quantile Regression":               ("Base", "Localiser (upper envelope)"),
        "7. Peaks over Threshold (EVT)":        ("Base", "Localiser (extreme-point KDE)"),
        "8. Bayesian Optimization (Expected Improvement)": ("Base", "Heuristic (mean + σ)"),
        "9. kNN + GPR (mean)":                  ("Simple ensemble", "Hole-patch via averaging"),
        "10. kNN + Quantile (mean)":            ("Simple ensemble", "Average + tail signal"),
        "11. Anchored GPR (QR prior)":          ("Advanced", "Bayesian: QR mean + GPR residuals"),
        "12. Gated GPR (QR anchor)":            ("Advanced", "Multiplicative anchoring"),
        "13. QR × GPR (geometric mean)":        ("Advanced", "Consensus filter (localiser × landscape)"),
        "14. EVT × GPR (geometric mean)":       ("Advanced", "Consensus filter, sharper localiser"),
        "15. GPR / kNN (variance-weighted mixture)": ("Advanced", "Adaptive: trust GPR in holes"),
        "16. GPR / kNN (disagreement-amplified)": ("Advanced", "Hole-signature amplifier"),
        "17. Local GPR (QR-localised)":         ("Advanced", "Two-stage: locate then refit"),
    }
    for i, (name, (typ, role)) in enumerate(roles.items(), start=1):
        row = table.add_row().cells
        row[0].text = str(i)
        row[1].text = name
        row[2].text = typ
        row[3].text = role

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_PATH)
    return OUT_PATH


if __name__ == "__main__":
    p = build_doc()
    print(f"Wrote {p.resolve()}  ({p.stat().st_size:,} bytes)")
