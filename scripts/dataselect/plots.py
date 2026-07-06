"""Plotly helpers specific to the data-selective-training page.

Kept small: the page reuses the weakspot plotting utilities
(``plot_data_overview``, ``plot_ground_truth_error``, ``plot_comparison``, …)
for everything those already cover, and only adds the two views that are new
here — the true landscape and the Gaussian selection kernel.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _circle_xy(center, radius, n=200):
    theta = np.linspace(0, 2 * np.pi, n)
    return center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)


def plot_landscape(xx, yy, true_surface, center, radius,
                   title="Ground-truth landscape f(x₁, x₂)"):
    """Filled contour of the noiseless target with the induced weakspot overlaid."""
    Z = np.asarray(true_surface).reshape(xx.shape)
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z, colorscale="Viridis",
        colorbar=dict(title="f(x₁, x₂)"), contours=dict(showlines=False),
        opacity=0.9,
    ))
    if radius and radius > 0:
        cx, cy = _circle_xy(center, radius)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Induced weakspot",
        ))
    fig.update_layout(
        title=title, xaxis_title="x₁ (norm)", yaxis_title="x₂ (norm)",
        height=460, template="plotly_white",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


def plot_selection(X_pool, weights, X_sel, sel_center, sel_sigma,
                   induced_center, induced_radius,
                   title="Weakspot-guided data selection", show_kernel=True):
    """Candidate pool with the chosen (newly selected) points highlighted.

    ``show_kernel=True`` colours the pool by Gaussian selection weight and draws
    the 1σ / 2σ selection rings around the detected weakspot centre (weakspot-
    guided view). ``show_kernel=False`` draws a neutral pool with no kernel —
    used for the random baseline, where selection ignores the weakspot.
    """
    fig = go.Figure()

    # Candidate pool — coloured by weight (guided) or neutral grey (baseline)
    if show_kernel and weights is not None:
        pool_marker = dict(color=weights, colorscale="Blues", size=5,
                           colorbar=dict(title="Selection<br>weight"),
                           line=dict(width=0.3, color="lightgray"))
    else:
        pool_marker = dict(color="#c7c7c7", size=5,
                           line=dict(width=0.3, color="lightgray"))
    fig.add_trace(go.Scatter(
        x=X_pool[:, 0], y=X_pool[:, 1], mode="markers",
        marker=pool_marker, name="Candidate pool (available)", opacity=0.6,
    ))

    # Selected points (the new datapoints that get added to the training set)
    if X_sel is not None and len(X_sel) > 0:
        fig.add_trace(go.Scatter(
            x=X_sel[:, 0], y=X_sel[:, 1], mode="markers",
            marker=dict(color="#d62728", size=7, symbol="circle",
                        line=dict(width=0.8, color="black")),
            name=f"Newly selected ({len(X_sel)})",
        ))

    # Selection kernel rings (1σ, 2σ) around the detected weakspot centroid
    if show_kernel and sel_center is not None and sel_sigma is not None:
        for k, dash in ((1, "dot"), (2, "dash")):
            rx, ry = _circle_xy(sel_center, k * sel_sigma)
            fig.add_trace(go.Scatter(
                x=rx, y=ry, mode="lines",
                line=dict(color="#7f00ff", width=1.6, dash=dash),
                name=f"Selection {k}σ",
            ))
        fig.add_trace(go.Scatter(
            x=[sel_center[0]], y=[sel_center[1]], mode="markers",
            marker=dict(color="#7f00ff", size=11, symbol="x",
                        line=dict(width=1.5, color="black")),
            name="Detected weakspot centre",
        ))

    # Induced weakspot for reference
    if induced_radius and induced_radius > 0:
        cx, cy = _circle_xy(induced_center, induced_radius)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Induced weakspot",
        ))

    fig.update_layout(
        title=title, xaxis_title="x₁ (norm)", yaxis_title="x₂ (norm)",
        height=520, template="plotly_white",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


def plot_training_points(X_train, y_train, X_sel, center, radius,
                         title="Training set"):
    """Scatter of the training set (optionally with newly added selected points)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X_train[:, 0], y=X_train[:, 1], mode="markers",
        marker=dict(color=y_train, colorscale="Viridis", size=5,
                    colorbar=dict(title="y (noisy)"),
                    line=dict(width=0.3, color="lightgray")),
        name="Training points", opacity=0.8,
    ))
    if X_sel is not None and len(X_sel) > 0:
        fig.add_trace(go.Scatter(
            x=X_sel[:, 0], y=X_sel[:, 1], mode="markers",
            marker=dict(color="#d62728", size=7, symbol="diamond",
                        line=dict(width=0.6, color="black")),
            name=f"Newly selected points ({len(X_sel)})",
        ))
    if radius and radius > 0:
        cx, cy = _circle_xy(center, radius)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Induced weakspot",
        ))
    fig.update_layout(
        title=title, xaxis_title="x₁ (norm)", yaxis_title="x₂ (norm)",
        height=480, template="plotly_white",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig
