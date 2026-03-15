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
# Registry
# ─────────────────────────────────────────────

DETECTION_METHODS = {
    "Gaussian Process Regression": detect_gpr,
    "Polynomial Response Surface": detect_polynomial,
    "RBF Interpolation": detect_rbf,
    "kNN Performance Mapping": detect_knn,
    "LOESS Local Regression": detect_loess,
    "Quantile Regression": detect_quantile,
    "Peaks over Threshold (EVT)": detect_pot,
    "Bayesian Optimization (EI)": detect_bayesopt,
}
