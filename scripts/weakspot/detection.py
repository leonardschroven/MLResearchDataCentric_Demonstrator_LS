"""
Weakspot identification methods.

Each method receives:
  X_test_norm : (N, 2)  test inputs in [0,1]^2
  errors      : (N,)    absolute prediction errors at test points
  grid_flat   : (M, 2)  evaluation grid in [0,1]^2

Each method returns:
  surface : (M,)  estimated error intensity on the grid (higher = more likely weakspot)
"""
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.neighbors import KNeighborsRegressor, KernelDensity
from scipy.interpolate import Rbf
from scipy.stats import norm


# ─────────────────────────────────────────────
# Grid helper
# ─────────────────────────────────────────────

def create_grid(resolution: int = 40):
    """Create a uniform grid over [0,1]^2."""
    lin = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(lin, lin)
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def _subsample(X, errors, max_n: int = 400, seed: int = 0):
    if len(X) <= max_n:
        return X, errors
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), max_n, replace=False)
    return X[idx], errors[idx]


# ─────────────────────────────────────────────
# 1. Gaussian Process Regression
# ─────────────────────────────────────────────

def detect_gpr(X_test_norm, errors, grid_flat, **kwargs):
    """GPR error surface — posterior mean predicts local error magnitude."""
    X, e = _subsample(X_test_norm, errors, max_n=300)
    kernel = Matern(nu=1.5, length_scale_bounds=(1e-2, 10)) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                   normalize_y=True, alpha=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(X, e)
    mean, _ = gpr.predict(grid_flat, return_std=True)
    return np.clip(mean, 0, None)


# ─────────────────────────────────────────────
# 2. Polynomial Response Surface
# ─────────────────────────────────────────────

def detect_polynomial(X_test_norm, errors, grid_flat, degree: int = 3, **kwargs):
    """Polynomial response surface fitted to the error values."""
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_p = poly.fit_transform(X_test_norm)
    G_p = poly.transform(grid_flat)
    lr = LinearRegression()
    lr.fit(X_p, errors)
    return np.clip(lr.predict(G_p), 0, None)


# ─────────────────────────────────────────────
# 3. Radial Basis Function Interpolation
# ─────────────────────────────────────────────

def detect_rbf(X_test_norm, errors, grid_flat, **kwargs):
    """RBF interpolation of the error surface."""
    X, e = _subsample(X_test_norm, errors, max_n=400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rbf = Rbf(X[:, 0], X[:, 1], e, function="multiquadric", smooth=0.05)
    surface = rbf(grid_flat[:, 0], grid_flat[:, 1])
    return np.clip(surface, 0, None)


# ─────────────────────────────────────────────
# 4. kNN Performance Mapping
# ─────────────────────────────────────────────

def detect_knn(X_test_norm, errors, grid_flat, k: int = 10, **kwargs):
    """kNN weighted average of errors for each grid point."""
    k = min(k, len(X_test_norm) - 1)
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
    knn.fit(X_test_norm, errors)
    return knn.predict(grid_flat)


# ─────────────────────────────────────────────
# 5. LOESS (Local Regression)
# ─────────────────────────────────────────────

def detect_loess(X_test_norm, errors, grid_flat, bandwidth_frac: float = 0.3, **kwargs):
    """
    2D LOESS — vectorized local weighted regression.
    bandwidth_frac: fraction of N used as local neighbourhood.
    """
    X, e = _subsample(X_test_norm, errors, max_n=400)
    N = len(X)
    M = len(grid_flat)
    k = max(5, int(bandwidth_frac * N))

    # Pairwise distances (M, N)
    diff = grid_flat[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    # Adaptive bandwidth per grid point
    sorted_d = np.sort(dists, axis=1)
    bw = sorted_d[:, min(k, N - 1)][:, np.newaxis] + 1e-10  # (M, 1)

    # Tricubic weights (M, N)
    u = np.clip(dists / bw, 0, 1)
    W = (1 - u ** 3) ** 3

    # Weighted local linear regression: design matrix (N, 3) = [1, x1, x2]
    X_aug = np.column_stack([np.ones(N), X])  # (N, 3)
    predictions = np.zeros(M)

    for i in range(M):
        w = W[i]
        if w.sum() < 1e-10:
            predictions[i] = e.mean()
            continue
        Xw = X_aug * w[:, np.newaxis]      # (N, 3)
        XtWX = X_aug.T @ Xw               # (3, 3)
        XtWy = Xw.T @ e                   # (3,)
        try:
            beta = np.linalg.solve(XtWX + np.eye(3) * 1e-6, XtWy)
            predictions[i] = np.array([1, grid_flat[i, 0], grid_flat[i, 1]]) @ beta
        except np.linalg.LinAlgError:
            predictions[i] = np.average(e, weights=w)

    return np.clip(predictions, 0, None)


# ─────────────────────────────────────────────
# 6. Quantile Regression
# ─────────────────────────────────────────────

def detect_quantile(X_test_norm, errors, grid_flat, quantile: float = 0.90,
                    degree: int = 2, **kwargs):
    """Polynomial quantile regression at high quantile → upper error envelope."""
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_p = poly.fit_transform(X_test_norm)
    G_p = poly.transform(grid_flat)
    qr = QuantileRegressor(quantile=quantile, alpha=0.01, solver="highs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qr.fit(X_p, errors)
    return np.clip(qr.predict(G_p), 0, None)


# ─────────────────────────────────────────────
# 7. Peaks over Threshold (EVT)
# ─────────────────────────────────────────────

def detect_pot(X_test_norm, errors, grid_flat, threshold_quantile: float = 0.85, **kwargs):
    """
    Peaks Over Threshold EVT.
    Fits a kernel density to extreme-error locations to map their concentration.
    """
    threshold = np.quantile(errors, threshold_quantile)
    mask = errors >= threshold
    if mask.sum() < 3:
        return np.zeros(len(grid_flat))

    X_extreme = X_test_norm[mask]
    bandwidth = max(0.05, 0.5 / np.sqrt(mask.sum()))
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(X_extreme)
    log_dens = kde.score_samples(grid_flat)
    dens = np.exp(log_dens - log_dens.max())  # normalized
    return dens


# ─────────────────────────────────────────────
# 8. Bayesian Optimization (EI acquisition)
# ─────────────────────────────────────────────

def detect_bayesopt(X_test_norm, errors, grid_flat, **kwargs):
    """
    Bayesian Optimization perspective: GP surrogate of error + Expected Improvement.
    EI highlights where the BO search for maximum error would focus.
    """
    X, e = _subsample(X_test_norm, errors, max_n=300)
    kernel = Matern(nu=2.5, length_scale_bounds=(1e-2, 10)) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                   normalize_y=True, alpha=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(X, e)

    mu, sigma = gpr.predict(grid_flat, return_std=True)
    f_best = e.max()
    Z = (mu - f_best) / (sigma + 1e-9)
    EI = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
    EI = np.clip(EI, 0, None)
    return EI


# ─────────────────────────────────────────────
# 9 & 10. Ensembles — average of two component surfaces
# ─────────────────────────────────────────────

def _unit_normalize(s: np.ndarray) -> np.ndarray:
    """Min-max scale a surface to [0,1] so component averages aren't dominated
    by whichever method happens to have the larger raw range."""
    s = np.asarray(s, dtype=float)
    lo, hi = float(s.min()), float(s.max())
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def detect_knn_gpr(X_test_norm, errors, grid_flat, **kwargs):
    """Mean of unit-normalized kNN and GPR surfaces.

    kNN is local & jagged; GPR is smooth & global. Averaging them keeps the
    local sharpness of kNN while letting the GPR posterior fill in regions
    with sparse coverage.
    """
    s_knn = _unit_normalize(detect_knn(X_test_norm, errors, grid_flat, **kwargs))
    s_gpr = _unit_normalize(detect_gpr(X_test_norm, errors, grid_flat, **kwargs))
    return 0.5 * (s_knn + s_gpr)


def detect_knn_quantile(X_test_norm, errors, grid_flat, **kwargs):
    """Mean of unit-normalized kNN and Quantile-Regression surfaces.

    kNN responds to *average* local error; quantile regression to the *upper
    envelope*. Averaging emphasises regions where both the average and the
    tail of the error distribution are elevated.
    """
    s_knn = _unit_normalize(detect_knn(X_test_norm, errors, grid_flat, **kwargs))
    s_qr  = _unit_normalize(detect_quantile(X_test_norm, errors, grid_flat, **kwargs))
    return 0.5 * (s_knn + s_qr)


# ─────────────────────────────────────────────
# Internal helpers reused by the advanced ensembles
# ─────────────────────────────────────────────

def _gpr_mean_std(X_test_norm, errors, grid_flat):
    """Fit a Matérn-1.5 GPR and return posterior (mean, std) on the grid.

    Separate helper because ``detect_gpr`` discards std and we need it for
    confidence-weighted ensembles.
    """
    X, e = _subsample(X_test_norm, errors, max_n=300)
    kernel = Matern(nu=1.5, length_scale_bounds=(1e-2, 10)) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                   normalize_y=True, alpha=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(X, e)
    mean, std = gpr.predict(grid_flat, return_std=True)
    return np.clip(mean, 0, None), std


def _peak_and_spread(surface, grid_flat, top_quantile: float = 0.85):
    """From an arbitrary positive surface defined on ``grid_flat``, return
    (peak location, characteristic spread) of the top-``top_quantile`` pixels.

    Used to pull a "where is the weakspot" anchor + radius estimate out of
    any localizer-style surface (QR, EVT, ...).
    """
    surf = _unit_normalize(surface)
    peak = grid_flat[int(np.argmax(surface))]
    thresh = np.quantile(surf, top_quantile)
    high = surf >= thresh
    if high.sum() < 3:
        return peak, 0.15
    pts = grid_flat[high]
    w = surf[high]
    if w.sum() < 1e-9:
        w = np.ones_like(w)
    # Weighted centroid then weighted spread around the peak
    c = np.average(pts, axis=0, weights=w)
    # Anchor = surface argmax (sharper than the centroid for sharp peaks)
    spread = float(np.sqrt(
        np.average(np.sum((pts - peak) ** 2, axis=1), weights=w)
    ))
    return peak, max(0.05, min(0.35, spread))


# ─────────────────────────────────────────────
# 11. Anchored GPR — QR surface as GPR prior mean
# ─────────────────────────────────────────────

def detect_anchored_gpr(X_test_norm, errors, grid_flat, **kwargs):
    """GPR fit to *residuals* after subtracting a Quantile-Regression prior mean.

    Bayesian-style: use QR's upper-envelope surface as the prior mean function
    m₀(x). Fit the GPR to (errors − m₀(X)) so the posterior corrects QR's
    shape rather than starting from zero. Final surface = m₀(grid) + residual.

    Effect: anchored at QR's peak by construction, but the residual GPR is free
    to bend the shape to match the actual error landscape.
    """
    # Predict QR on the union of train points and grid — single fit, both outputs
    combined = np.vstack([X_test_norm, grid_flat])
    qr_combined = detect_quantile(X_test_norm, errors, combined, **kwargs)
    qr_train = qr_combined[:len(X_test_norm)]
    qr_grid  = qr_combined[len(X_test_norm):]

    # Fit GPR to residuals
    residuals = errors - qr_train
    Xs, rs = _subsample(X_test_norm, residuals, max_n=300)
    kernel = Matern(nu=1.5, length_scale_bounds=(1e-2, 10)) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                   normalize_y=True, alpha=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(Xs, rs)
    residual_grid = gpr.predict(grid_flat)

    return np.clip(qr_grid + residual_grid, 0, None)


# ─────────────────────────────────────────────
# 12. Gated GPR — Gaussian mask centred on the QR peak
# ─────────────────────────────────────────────

def detect_gated_gpr(X_test_norm, errors, grid_flat, **kwargs):
    """GPR surface multiplied by a Gaussian gate centred on the QR peak.

    QR/EVT locate the peak; GPR provides the shape. The gate suppresses
    landscape detail far from the localizer's anchor — sharper than
    averaging, keeps full GPR detail inside the gate.
    """
    qr_surface = detect_quantile(X_test_norm, errors, grid_flat, **kwargs)
    c_star, tau = _peak_and_spread(qr_surface, grid_flat)

    gpr_surface = _unit_normalize(detect_gpr(X_test_norm, errors, grid_flat, **kwargs))
    d2 = np.sum((grid_flat - c_star) ** 2, axis=1)
    gate = np.exp(-d2 / (2 * tau ** 2))
    return gpr_surface * gate


# ─────────────────────────────────────────────
# 13 & 14. Geometric-mean consensus ensembles
# ─────────────────────────────────────────────

def detect_geomean_qr_gpr(X_test_norm, errors, grid_flat, **kwargs):
    """sqrt( QR · GPR )  —  consensus filter: pixels need to be flagged
    by **both** the localizer and the landscape estimator to survive.

    More conservative than the arithmetic mean — penalises any pixel
    where the two methods disagree.
    """
    s_qr  = _unit_normalize(detect_quantile(X_test_norm, errors, grid_flat, **kwargs))
    s_gpr = _unit_normalize(detect_gpr(X_test_norm, errors, grid_flat, **kwargs))
    return np.sqrt(s_qr * s_gpr)


def detect_geomean_evt_gpr(X_test_norm, errors, grid_flat, **kwargs):
    """sqrt( EVT · GPR )  —  same idea with extreme-value localizer."""
    s_pot = _unit_normalize(detect_pot(X_test_norm, errors, grid_flat, **kwargs))
    s_gpr = _unit_normalize(detect_gpr(X_test_norm, errors, grid_flat, **kwargs))
    return np.sqrt(s_pot * s_gpr)


# ─────────────────────────────────────────────
# 15. Variance-weighted GPR ⇆ kNN mixture
# ─────────────────────────────────────────────

def detect_variance_weighted(X_test_norm, errors, grid_flat, **kwargs):
    """Blend GPR and kNN by GPR's own posterior variance.

    Where σ_GPR is small (dense data) → trust kNN's sharp local average.
    Where σ_GPR is large (data hole — exactly where the weakspot lives) →
    trust GPR's principled extrapolation.

    α(x) = sigmoid( (σ_GPR(x) − median) / IQR )
    S    = α · GPR + (1−α) · kNN
    """
    mu, sigma = _gpr_mean_std(X_test_norm, errors, grid_flat)
    s_gpr = _unit_normalize(mu)
    s_knn = _unit_normalize(detect_knn(X_test_norm, errors, grid_flat, **kwargs))

    s_med = float(np.median(sigma))
    s_iqr = max(
        float(np.percentile(sigma, 75) - np.percentile(sigma, 25)),
        1e-6
    )
    alpha = 1.0 / (1.0 + np.exp(-(sigma - s_med) / s_iqr))
    return alpha * s_gpr + (1.0 - alpha) * s_knn


# ─────────────────────────────────────────────
# 16. Disagreement-amplified GPR vs kNN
# ─────────────────────────────────────────────

def detect_disagreement_gpr_knn(X_test_norm, errors, grid_flat,
                                lam: float = 0.7, **kwargs):
    """Add λ·max(0, GPR − kNN) to GPR.

    Data holes are exactly where GPR predicts high error but kNN drops
    (no neighbours). The asymmetric clip amplifies that gap *only* in the
    direction that indicates a weakspot — symmetric noise is ignored.
    """
    s_gpr = _unit_normalize(detect_gpr(X_test_norm, errors, grid_flat, **kwargs))
    s_knn = _unit_normalize(detect_knn(X_test_norm, errors, grid_flat, **kwargs))
    disagreement = np.maximum(0.0, s_gpr - s_knn)
    return _unit_normalize(s_gpr + lam * disagreement)


# ─────────────────────────────────────────────
# 17. Local-refined GPR — QR-localised second-stage GPR
# ─────────────────────────────────────────────

def detect_local_refined_gpr(X_test_norm, errors, grid_flat, **kwargs):
    """Two-stage: QR finds the region, a *locally refit* GPR shapes it.

    Stage A: run QR on the full data → peak ``c*`` and spread ``τ``.
    Stage B: subset training points within ~3τ of ``c*`` and refit GPR with
             tighter length-scale bounds on that subset, so the kernel
             length-scale is dominated by *local* geometry instead of the
             global mean spacing. Predict on the full grid.

    Falls back to the global GPR if the local subset is too small (< 20).
    """
    # Stage A — localize via QR
    combined = np.vstack([X_test_norm, grid_flat])
    qr_combined = detect_quantile(X_test_norm, errors, combined, **kwargs)
    qr_grid = qr_combined[len(X_test_norm):]
    c_star, tau = _peak_and_spread(qr_grid, grid_flat)
    radius = float(np.clip(3.0 * tau, 0.10, 0.35))

    # Stage B — local subset
    d = np.linalg.norm(X_test_norm - c_star, axis=1)
    local_mask = d <= radius
    if local_mask.sum() < 20:
        return detect_gpr(X_test_norm, errors, grid_flat, **kwargs)

    X_local = X_test_norm[local_mask]
    e_local = errors[local_mask]
    Xs, es = _subsample(X_local, e_local, max_n=300)

    # Tighter length-scale bounds — encourage finer local shape
    kernel = Matern(nu=1.5, length_scale_bounds=(1e-3, 5.0)) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                   normalize_y=True, alpha=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(Xs, es)
    return np.clip(gpr.predict(grid_flat), 0, None)


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

DETECTION_METHODS = {
    # Base methods
    "Gaussian Process Regression": detect_gpr,
    "Polynomial Response Surface": detect_polynomial,
    "RBF Interpolation": detect_rbf,
    "kNN Performance Mapping": detect_knn,
    "LOESS Local Regression": detect_loess,
    "Quantile Regression": detect_quantile,
    "Peaks over Threshold (EVT)": detect_pot,
    "Bayesian Optimization (EI)": detect_bayesopt,
    # Simple averages
    "kNN + GPR (mean)": detect_knn_gpr,
    "kNN + Quantile (mean)": detect_knn_quantile,
    # Advanced ensembles
    "Anchored GPR (QR prior)": detect_anchored_gpr,
    "Gated GPR (QR anchor)": detect_gated_gpr,
    "QR × GPR (geometric)": detect_geomean_qr_gpr,
    "EVT × GPR (geometric)": detect_geomean_evt_gpr,
    "GPR/kNN (variance-weighted)": detect_variance_weighted,
    "GPR/kNN (disagreement-amplified)": detect_disagreement_gpr_knn,
    "Local GPR (QR-localised)": detect_local_refined_gpr,
}
