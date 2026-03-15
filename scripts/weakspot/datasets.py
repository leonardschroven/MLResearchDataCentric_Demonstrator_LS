"""
Dataset loading and exclusion zone utilities for the weakspot experiment.

Recommended open-license 2D regression datasets:
- California Housing (lat/lon → house value)  — sklearn built-in, StatLib/CC0
- Synthetic Peaks function                     — generated, no download needed
- Fish Market (Length+Height → Weight)         — Kaggle CC0: aungpyaeap/fish-market
- Forest Fires (temp+RH → area)                — UCI CC BY 4.0
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

AVAILABLE_DATASETS = {
    "California Housing (Lat/Lon → House Value)": "california",
    "Synthetic Peaks Function": "synthetic",
    "Synthetic Gaussian Bumps": "gaussian_bumps",
}

def load_dataset(key: str, n_samples: int = 1500, random_state: int = 42):
    """
    Load a 2D regression dataset.
    Returns: X (N,2), y (N,), x1_name, x2_name, y_name
    """
    rng = np.random.RandomState(random_state)

    if key == "california":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        X = df[["Longitude", "Latitude"]].values
        y = data.target
        # Subsample
        idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
        return X[idx], y[idx], "Longitude", "Latitude", "Median House Value ($100k)"

    elif key == "synthetic":
        n = min(n_samples, 2000)
        x1 = rng.uniform(-3, 3, n)
        x2 = rng.uniform(-3, 3, n)
        y = (3*(1 - x1)**2 * np.exp(-x1**2 - (x2+1)**2)
             - 10*(x1/5 - x1**3 - x2**5) * np.exp(-x1**2 - x2**2)
             - 1/3 * np.exp(-(x1+1)**2 - x2**2))
        y += rng.normal(0, 0.15, n)
        X = np.column_stack([x1, x2])
        return X, y, "x₁", "x₂", "f(x₁, x₂)"

    elif key == "gaussian_bumps":
        # Five Gaussian bumps at fixed centres + noise — clear local structure,
        # great for weakspot detection because removing one bump creates a clean gap.
        n = min(n_samples, 2000)
        x1 = rng.uniform(0, 6, n)
        x2 = rng.uniform(0, 6, n)
        centres = np.array([[1.0, 1.0], [1.5, 4.5], [3.0, 3.0], [4.5, 1.0], [5.0, 5.0]])
        heights = np.array([3.0, 2.5, 4.0, 2.0, 3.5])
        widths  = np.array([0.6, 0.7, 0.8, 0.5, 0.6])
        y = np.zeros(n)
        for (cx, cy), h, w in zip(centres, heights, widths):
            y += h * np.exp(-((x1 - cx)**2 + (x2 - cy)**2) / (2 * w**2))
        y += rng.normal(0, 0.1, n)
        X = np.column_stack([x1, x2])
        return X, y, "x₁", "x₂", "Bump Height"

    else:
        raise ValueError(f"Unknown dataset key: {key}")


def normalize(X: np.ndarray):
    """Normalize X to [0,1]. Returns (X_norm, scaler)."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(X), scaler


def apply_exclusion_zone(X_norm: np.ndarray, y: np.ndarray,
                          center: np.ndarray, radius: float):
    """
    Remove points within `radius` of `center` (circle exclusion, normalized space).
    Returns: X_keep, y_keep, X_excl, y_excl, dists
    """
    dists = np.sqrt(np.sum((X_norm - center) ** 2, axis=1))
    mask_keep = dists > radius
    mask_excl = ~mask_keep
    return (X_norm[mask_keep], y[mask_keep],
            X_norm[mask_excl], y[mask_excl], dists)


def apply_exclusion_knn(X_norm: np.ndarray, y: np.ndarray,
                         center: np.ndarray, n_exclude: int):
    """
    Remove the `n_exclude` nearest points to `center` (kNN exclusion).
    Returns: X_keep, y_keep, X_excl, y_excl, dists
    """
    dists = np.sqrt(np.sum((X_norm - center) ** 2, axis=1))
    n_exclude = min(n_exclude, len(X_norm) - 1)
    excl_idx = np.argsort(dists)[:n_exclude]
    mask_excl = np.zeros(len(X_norm), dtype=bool)
    mask_excl[excl_idx] = True
    mask_keep = ~mask_excl
    return (X_norm[mask_keep], y[mask_keep],
            X_norm[mask_excl], y[mask_excl], dists)


def make_train_test_split(X_norm, y, excl_mask, test_size=0.2, random_state=42):
    """
    Split into train (outside exclusion zone) and test (full dataset).
    """
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_norm, y, test_size=test_size, random_state=random_state
    )
    return X_tr, X_te, y_tr, y_te
