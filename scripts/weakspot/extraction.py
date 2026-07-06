"""
Stage-2 weakspot extraction.

Each detection method produces a continuous error surface. This module turns that
surface into a *parametric* weakspot estimate:

    threshold the surface  →  connected components  →  largest component  →
    weighted centroid + covariance  →  per-axis sigmas + oriented ellipse.

Returned sigmas describe the spatial extent of the weakspot in each input
dimension, so a wide-in-x1 / narrow-in-x2 weakspot is captured directly.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay


def induced_mask(xx, yy, excl_mode: str, center, radius: float, X_excl):
    """Binary ground-truth mask of the *induced* weakspot region on the grid.

    Circle mode → pixels within `radius` of `center`.
    kNN mode    → pixels inside the convex hull of the excluded points.
    """
    if excl_mode == "Circle (radius)":
        return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2

    if X_excl is None or len(X_excl) < 3:
        return np.zeros(xx.shape, dtype=bool)
    try:
        hull = ConvexHull(X_excl)
        deln = Delaunay(X_excl[hull.vertices])
    except Exception:
        return np.zeros(xx.shape, dtype=bool)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return (deln.find_simplex(grid) >= 0).reshape(xx.shape)


def ellipse_mask(xx, yy, ext) -> np.ndarray:
    """Boolean mask of pixels inside the extraction's 2σ ellipse."""
    if ext is None:
        return np.zeros(xx.shape, dtype=bool)
    cx, cy = ext["center"]
    a = 2.0 * ext["sigma_major"]
    b = 2.0 * ext["sigma_minor"]
    if a <= 0 or b <= 0:
        return np.zeros(xx.shape, dtype=bool)
    ang = ext["angle_rad"]
    dx = xx - cx
    dy = yy - cy
    # Rotate into ellipse-aligned frame
    xr =  dx * np.cos(ang) + dy * np.sin(ang)
    yr = -dx * np.sin(ang) + dy * np.cos(ang)
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def iou_ellipse_at_thresholds(surface, xx, yy, gt_mask,
                              quantiles=(0.80, 0.85, 0.90, 0.95)):
    """IoU between the extracted ellipse region (per quantile) and the induced mask.

        IoU(q) = | ellipse(q) ∩ induced | / | ellipse(q) ∪ induced |

    The ellipse is rebuilt at each quantile by re-running ``extract_weakspot`` —
    the threshold defines the connected component, which defines the ellipse.

    Returns
    -------
    dict[float, float] — quantile → IoU in [0, 1]. NaN if either region is empty.
    """
    out = {}
    gt_sum = int(gt_mask.sum())
    for q in quantiles:
        if gt_sum == 0:
            out[q] = float("nan")
            continue
        ext = extract_weakspot(surface, xx, yy, threshold_quantile=q)
        em = ellipse_mask(xx, yy, ext)
        em_sum = int(em.sum())
        if em_sum == 0:
            out[q] = 0.0
            continue
        inter = int(np.logical_and(em, gt_mask).sum())
        union = int(np.logical_or(em, gt_mask).sum())
        out[q] = float(inter / union) if union > 0 else 0.0
    return out


def extract_weakspot(surface, xx, yy, threshold_quantile: float = 0.85):
    """
    Parameters
    ----------
    surface : (M,) array — flat surface values (already normalized to [0,1]).
    xx, yy  : (R, R) meshgrid arrays produced by ``create_grid``.
    threshold_quantile : float in (0, 1).
        Pixels with surface >= this quantile are considered "weakspot pixels".

    Returns
    -------
    dict or None — None if no pixel exceeds the threshold.
        center           : (cx, cy)  weighted centroid of the largest component
        sigma_x1, sigma_x2 : axis-aligned per-dimension standard deviations
        sigma_major, sigma_minor : eigenvalues (sqrt) of the covariance — i.e.
                                   spread along the principal axes of the blob
        angle_rad        : orientation of the major axis (radians from +x₁)
        area_frac        : fraction of [0,1]^2 covered by the largest component
        n_components     : how many distinct above-threshold regions were found
        threshold_value  : the numeric surface threshold used
        ellipse_x, ellipse_y : (200,) arrays tracing the 2σ ellipse for plotting
        mask             : (R, R) bool — pixels of the largest component
    """
    Z = surface.reshape(xx.shape)
    threshold_value = float(np.quantile(surface, threshold_quantile))
    mask = Z >= threshold_value

    if mask.sum() == 0:
        return None

    labels, n_components = ndimage.label(mask)
    if n_components == 0:
        return None

    # Largest component, weighted by integrated surface mass (not just pixel count)
    component_mass = ndimage.sum(Z, labels, index=range(1, n_components + 1))
    largest = int(np.argmax(component_mass)) + 1
    keep = labels == largest

    pts = np.column_stack([xx[keep], yy[keep]])
    weights = Z[keep].astype(float)
    if weights.sum() < 1e-12:
        weights = np.ones_like(weights)

    # Weighted centroid
    cx = float(np.average(pts[:, 0], weights=weights))
    cy = float(np.average(pts[:, 1], weights=weights))

    # Weighted covariance (no Bessel correction — N is grid pixel count, not a sample)
    centered = pts - np.array([cx, cy])
    W = weights / weights.sum()
    cov = (centered * W[:, None]).T @ centered

    sigma_x1 = float(np.sqrt(max(cov[0, 0], 0.0)))
    sigma_x2 = float(np.sqrt(max(cov[1, 1], 0.0)))

    # Principal axes
    evals, evecs = np.linalg.eigh(cov)  # ascending
    sigma_minor = float(np.sqrt(max(evals[0], 0.0)))
    sigma_major = float(np.sqrt(max(evals[1], 0.0)))
    major_vec = evecs[:, 1]
    angle = float(np.arctan2(major_vec[1], major_vec[0]))

    # 2σ ellipse points (covers ~95% of an isotropic Gaussian blob)
    t = np.linspace(0, 2 * np.pi, 200)
    base = np.column_stack([2 * sigma_major * np.cos(t),
                            2 * sigma_minor * np.sin(t)])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    ellipse = base @ R.T + np.array([cx, cy])

    return {
        "center": (cx, cy),
        "sigma_x1": sigma_x1,
        "sigma_x2": sigma_x2,
        "sigma_major": sigma_major,
        "sigma_minor": sigma_minor,
        "angle_rad": angle,
        "area_frac": float(keep.sum() / keep.size),
        "n_components": int(n_components),
        "threshold_value": threshold_value,
        "ellipse_x": ellipse[:, 0],
        "ellipse_y": ellipse[:, 1],
        "mask": keep,
    }
