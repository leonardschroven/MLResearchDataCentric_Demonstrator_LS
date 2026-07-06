"""Visualization utilities for the weakspot experiment."""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def _circle_trace(center, radius, name="Induced Weakspot", color="red", dash="dash"):
    """Return a plotly trace drawing a circle in [0,1]^2 space."""
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = center
    x_c = cx + radius * np.cos(theta)
    y_c = cy + radius * np.sin(theta)
    return go.Scatter(x=x_c, y=y_c, mode="lines",
                      line=dict(color=color, width=2, dash=dash),
                      name=name, showlegend=True)


def plot_data_overview(X_norm, y, X_excl_norm, center, radius,
                       x1_name="x₁", x2_name="x₂", y_name="y",
                       excl_mode="Circle (radius)"):
    """Scatter of full data + excluded zone highlighted."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X_norm[:, 0], y=X_norm[:, 1],
        mode="markers",
        marker=dict(color=y, colorscale="Viridis", size=4,
                    colorbar=dict(title=y_name), showscale=True),
        name="Training data", text=[f"{y_name}: {v:.2f}" for v in y]
    ))

    if len(X_excl_norm) > 0:
        fig.add_trace(go.Scatter(
            x=X_excl_norm[:, 0], y=X_excl_norm[:, 1],
            mode="markers",
            marker=dict(color="red", size=6, symbol="x", line=dict(width=1.5)),
            name=f"Excluded ({len(X_excl_norm)} pts)"
        ))

    # Draw boundary: circle for radius mode, convex hull for kNN mode
    if excl_mode == "Circle (radius)" and radius and radius > 0:
        fig.add_trace(_circle_trace(center, radius))
    elif len(X_excl_norm) >= 3:
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(X_excl_norm)
            hull_pts = np.vstack([X_excl_norm[hull.vertices],
                                  X_excl_norm[hull.vertices[0]]])
            fig.add_trace(go.Scatter(
                x=hull_pts[:, 0], y=hull_pts[:, 1], mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="Exclusion boundary"
            ))
        except Exception:
            pass

    fig.update_layout(
        title="Dataset with Induced Weakspot Zone",
        xaxis_title=f"{x1_name} (normalized)",
        yaxis_title=f"{x2_name} (normalized)",
        height=500, template="plotly_white",
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)", bordercolor="lightgray", borderwidth=1
        )
    )
    return fig


def plot_error_surface(xx, yy, surface, X_test_norm, errors,
                       center, radius, title="Error Surface"):
    """2D heatmap of error surface + test point scatter + weakspot circle."""
    Z = surface.reshape(xx.shape)

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        colorscale="YlOrRd", opacity=0.85,
        colorbar=dict(title="Abs. Prediction Error<br>|y_pred − y_true|"),
        contours=dict(showlines=False),
        name="Error surface"
    ))

    fig.add_trace(go.Scatter(
        x=X_test_norm[:, 0], y=X_test_norm[:, 1],
        mode="markers",
        marker=dict(
            color="#00BFFF",  # bright cyan — clearly distinct from YlOrRd heatmap
            size=5,
            line=dict(width=0.8, color="black"),
            symbol="circle",
        ),
        name="Test points", text=[f"err: {e:.3f}" for e in errors],
        opacity=0.75,
    ))

    fig.add_trace(_circle_trace(center, radius))

    fig.update_layout(
        title=title,
        xaxis_title="x₁ (norm)", yaxis_title="x₂ (norm)",
        height=400, template="plotly_white",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1])
    )
    return fig


def plot_ground_truth_error(X_test_norm, errors, center, radius):
    """Show actual model errors as a scatter at real test point locations (no interpolation)."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X_test_norm[:, 0], y=X_test_norm[:, 1],
        mode="markers",
        marker=dict(
            color=errors, colorscale="YlOrRd", size=6,
            colorbar=dict(title="Abs. Prediction Error<br>|y_pred − y_true|"),
            showscale=True,
            line=dict(width=0.5, color="black"),
        ),
        name="Test errors",
        text=[f"err: {e:.3f}" for e in errors],
    ))

    if radius and radius > 0:
        fig.add_trace(_circle_trace(center, radius))

    fig.update_layout(
        title="Ground-Truth Error Surface (Actual Test Points — No Interpolation)",
        xaxis_title="x₁ (norm)", yaxis_title="x₂ (norm)",
        height=400, template="plotly_white",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)")
    )
    return fig


def _smooth_errors(X, errors, sigma):
    """Gaussian kernel smoothing of scattered error values.
    sigma controls neighbourhood size in normalised [0,1]^2 space."""
    if sigma <= 0:
        return errors
    from scipy.spatial.distance import cdist
    D = cdist(X, X)
    W = np.exp(-0.5 * (D / sigma) ** 2)
    W /= W.sum(axis=1, keepdims=True)
    return W @ errors


def plot_ground_truth_error_3d(X_test_norm, errors, center, radius, sigma=0.0):
    """3D triangulated surface of actual model errors using Delaunay triangulation."""
    from scipy.spatial import Delaunay
    z = _smooth_errors(X_test_norm, errors, sigma)
    tri = Delaunay(X_test_norm)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=X_test_norm[:, 0],
        y=X_test_norm[:, 1],
        z=z,
        i=tri.simplices[:, 0],
        j=tri.simplices[:, 1],
        k=tri.simplices[:, 2],
        intensity=z,
        intensitymode="vertex",
        colorscale="YlOrRd",
        colorbar=dict(title="Abs. Prediction Error<br>|y_pred − y_true|"),
        opacity=0.9,
        name="Error surface",
    ))

    # Draw weakspot circle on the base plane (skip when no weakspot induced)
    if radius and radius > 0:
        theta = np.linspace(0, 2 * np.pi, 200)
        cx, cy = center
        x_c = cx + radius * np.cos(theta)
        y_c = cy + radius * np.sin(theta)
        fig.add_trace(go.Scatter3d(
            x=x_c, y=y_c, z=np.zeros(200),
            mode="lines",
            line=dict(color="red", width=4),
            name="Induced weakspot zone",
        ))

    smooth_label = f"  (σ={sigma:.2f})" if sigma > 0 else "  (no smoothing)"
    fig.update_layout(
        title=f"Ground-Truth Error Surface — 3D Polygon Mesh{smooth_label}",
        scene=dict(
            xaxis_title="x₁ (norm)",
            yaxis_title="x₂ (norm)",
            zaxis_title="Abs. Prediction Error",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
        ),
        height=520, template="plotly_white",
    )
    return fig


def plot_all_methods(xx, yy, surfaces: dict, center, radius,
                     X_test_norm, errors, extractions: dict | None = None):
    """Grid of subplots — one per detection method.

    If ``extractions`` is provided, the extracted 2σ ellipse and centroid are
    overlaid on each subplot.
    """
    methods = list(surfaces.keys())
    n = len(methods)
    cols = 4
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=methods,
        vertical_spacing=0.08, horizontal_spacing=0.06
    )

    grid_flat = np.column_stack([xx.ravel(), yy.ravel()])
    theta = np.linspace(0, 2 * np.pi, 120)
    cx, cy = center
    x_c = cx + radius * np.cos(theta)
    y_c = cy + radius * np.sin(theta)

    for i, name in enumerate(methods):
        r = i // cols + 1
        c = i % cols + 1
        Z = surfaces[name].reshape(xx.shape)

        fig.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0], z=Z,
            colorscale="YlOrRd", showscale=False,
            contours=dict(showlines=False), opacity=0.85
        ), row=r, col=c)

        fig.add_trace(go.Scatter(
            x=x_c, y=y_c, mode="lines",
            line=dict(color="red", width=1.5, dash="dash"),
            showlegend=(i == 0), name="Induced zone"
        ), row=r, col=c)

        ext = extractions.get(name) if extractions else None
        if ext is not None:
            fig.add_trace(go.Scatter(
                x=ext["ellipse_x"], y=ext["ellipse_y"],
                mode="lines",
                line=dict(color="cyan", width=2),
                showlegend=(i == 0), name="Detected 2σ ellipse",
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=[ext["center"][0]], y=[ext["center"][1]],
                mode="markers",
                marker=dict(color="cyan", size=8, symbol="x",
                            line=dict(width=1.5, color="black")),
                showlegend=(i == 0), name="Detected centroid",
            ), row=r, col=c)

    fig.update_layout(
        title="Weakspot Detection — All Methods",
        height=max(350, rows * 320),
        template="plotly_white"
    )
    return fig


def plot_comparison(xx, yy, gt_surface, best_surface, best_name,
                    center, radius, X_test_norm, errors, sigma=0.0,
                    extraction=None):
    """Side-by-side: ground truth errors (Delaunay polygon surface) vs detected weakspot."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Ground-Truth Errors", f"Detected: {best_name}"],
                        horizontal_spacing=0.12)
    theta = np.linspace(0, 2 * np.pi, 120)
    cx, cy = center
    x_c = cx + radius * np.cos(theta)
    y_c = cy + radius * np.sin(theta)

    # ── Left: Delaunay polygon surface (linear interp within triangles) + scatter ──
    z = _smooth_errors(X_test_norm, errors, sigma)
    grid_flat = np.column_stack([xx.ravel(), yy.ravel()])
    interp = LinearNDInterpolator(X_test_norm, z)
    gt_grid = interp(grid_flat)
    nan_mask = np.isnan(gt_grid)
    if nan_mask.any():
        fill = NearestNDInterpolator(X_test_norm, z)
        gt_grid[nan_mask] = fill(grid_flat[nan_mask])
    Z_gt = gt_grid.reshape(xx.shape)

    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z_gt,
        colorscale="YlOrRd", opacity=0.85,
        showscale=True,
        colorbar=dict(
            title="Abs. Error<br>|y_pred−y_true|",
            x=0.44, len=0.85, thickness=14,
        ),
        contours=dict(showlines=False),
        name="GT surface",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=X_test_norm[:, 0], y=X_test_norm[:, 1],
        mode="markers",
        marker=dict(color="#00BFFF", size=4, line=dict(width=0.5, color="black")),
        name="Test points", showlegend=True,
        text=[f"err: {e:.3f}" for e in errors],
        opacity=0.7,
    ), row=1, col=1)

    if radius and radius > 0:
        fig.add_trace(go.Scatter(
            x=x_c, y=y_c, mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Induced zone", showlegend=True,
        ), row=1, col=1)

    # ── Right: detected surface as contour ──
    Z = best_surface.reshape(xx.shape)
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        colorscale="YlOrRd",
        showscale=True,
        colorbar=dict(
            title="Estimated Error Score<br>(normalized)",
            x=1.01, len=0.85, thickness=14,
        ),
        contours=dict(showlines=False), opacity=0.9,
    ), row=1, col=2)

    if radius and radius > 0:
        fig.add_trace(go.Scatter(
            x=x_c, y=y_c, mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Induced zone", showlegend=False,
        ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=X_test_norm[:, 0], y=X_test_norm[:, 1],
        mode="markers",
        marker=dict(color="#00BFFF", size=4, line=dict(width=0.6, color="black")),
        name="Test points", showlegend=False, opacity=0.7,
    ), row=1, col=2)

    if extraction is not None:
        fig.add_trace(go.Scatter(
            x=extraction["ellipse_x"], y=extraction["ellipse_y"],
            mode="lines",
            line=dict(color="cyan", width=2.5),
            name="Detected 2σ ellipse", showlegend=True,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[extraction["center"][0]], y=[extraction["center"][1]],
            mode="markers",
            marker=dict(color="cyan", size=10, symbol="x",
                        line=dict(width=1.5, color="black")),
            name="Detected centroid", showlegend=True,
        ), row=1, col=2)

    fig.update_layout(
        height=460, template="plotly_white",
        title="Induced Weakspot vs Detected Weakspot",
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


def plot_metrics_bar(metrics: dict):
    """Bar chart of model evaluation metrics."""
    fig = go.Figure(go.Bar(
        x=list(metrics.keys()), y=list(metrics.values()),
        marker_color=["#636EFA", "#EF553B", "#00CC96"]
    ))
    fig.update_layout(title="Model Performance Metrics",
                      yaxis_title="Value", height=300,
                      template="plotly_white")
    return fig
