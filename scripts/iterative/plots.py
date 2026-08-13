"""Plotly helpers for the iterative study.

Only the views the loop makes necessary live here; everything the single-round
pages already draw (landscape, selection kernel, error surface) is imported from
``scripts.dataselect.plots`` and ``scripts.weakspot.plotting`` unchanged.

One visual grammar is used throughout, so any chart can be read without a
legend lookup: **colour = strategy** (green guided, red random) and
**line style = regime** (solid accumulative, dotted new-only, dashed
scaled-accum).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scripts.iterative.loop import TRACKS

STRAT_COLOUR = {"guided": "#2ca02c", "random": "#d62728"}
REGIME_DASH = {"accumulative": "solid", "new-only": "dot", "scaled-accum": "dash"}
SMOOTHERS = ("None", "Rolling mean", "Exponential (EWMA)")


def track_label(key: str) -> str:
    strat, regime = TRACKS[key]
    return f"{strat} · {regime}"


def smooth(y, method: str, win: int) -> np.ndarray:
    """Smooth one iteration series.

    ``min_periods=1`` / ``adjust=False`` keep the endpoints defined so short
    trajectories still plot end to end. Purely cosmetic: every table and CSV
    below uses the raw numbers.
    """
    s = pd.Series(np.asarray(y, dtype=float))
    if method == "Rolling mean":
        return s.rolling(max(int(win), 1), center=True, min_periods=1).mean().to_numpy()
    if method == "Exponential (EWMA)":
        return s.ewm(span=max(int(win), 1), adjust=False).mean().to_numpy()
    return s.to_numpy()


def trend_figure(iters, hist: dict, metric: str, ylab: str, title: str,
                 method: str = "None", win: int = 3,
                 refs: dict | None = None) -> go.Figure:
    """Per-iteration trend for every active track.

    When smoothing is on the bold line is the trend and the raw values stay
    visible as faint dots, so nothing is hidden. ``refs`` draws horizontal
    reference lines (used for the budget-matched single-shot control).
    """
    fig = go.Figure()
    for key, series in hist.items():
        if metric not in series:
            continue
        strat, regime = TRACKS[key]
        colour, dash = STRAT_COLOUR[strat], REGIME_DASH[regime]
        name, y = track_label(key), np.asarray(series[metric], dtype=float)
        if method != "None":
            fig.add_trace(go.Scatter(
                x=iters, y=y, mode="markers", marker=dict(color=colour, size=5),
                opacity=0.25, legendgroup=name, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=iters, y=smooth(y, method, win), mode="lines", name=name,
                legendgroup=name, line=dict(color=colour, dash=dash, width=2.5)))
        else:
            fig.add_trace(go.Scatter(
                x=iters, y=y, mode="lines+markers", name=name, legendgroup=name,
                line=dict(color=colour, dash=dash, width=2),
                marker=dict(color=colour, size=6)))
    for label, (value, colour) in (refs or {}).items():
        if value is None or not np.isfinite(value):
            continue
        fig.add_hline(y=value, line_dash="dashdot", line_color=colour,
                      annotation_text=label, annotation_position="top left",
                      annotation_font_size=11)
    fig.update_layout(height=460, title=title, legend_title_text="",
                      template="plotly_white",
                      xaxis_title="iteration", yaxis_title=ylab)
    return fig


def gap_figure(iters, hist: dict, metric: str, title: str,
               method: str = "None", win: int = 3) -> go.Figure:
    """Guided − random per iteration, one line per regime (negative = guided wins).

    The single number the companion paper reported for one round, now as a
    trajectory: it shows not just whether guidance won but *when*, and whether an
    early advantage survives to the end of the loop.
    """
    fig = go.Figure()
    for regime, dash in REGIME_DASH.items():
        g = next((k for k, (s, r) in TRACKS.items()
                  if s == "guided" and r == regime and k in hist), None)
        r_ = next((k for k, (s, r) in TRACKS.items()
                   if s == "random" and r == regime and k in hist), None)
        if g is None or r_ is None:
            continue
        d = (np.asarray(hist[g][metric], dtype=float)
             - np.asarray(hist[r_][metric], dtype=float))
        fig.add_trace(go.Scatter(
            x=iters, y=smooth(d, method, win) if method != "None" else d,
            mode="lines+markers", name=regime,
            line=dict(color="#1f77b4", dash=dash, width=2.2),
            marker=dict(size=6)))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(height=400, title=title, template="plotly_white",
                      legend_title_text="regime", xaxis_title="iteration",
                      yaxis_title="guided − random (negative = guided wins)")
    return fig


def schedule_figure(sched: dict) -> go.Figure:
    """The three scheduled controls actually applied, round by round.

    Learning rate is drawn on its own log axis; the rehearsal mix α and kernel
    width σ share the left axis. The detected weakspot severity is overlaid so
    an adaptive schedule can be read against the signal that drove it.
    """
    k = list(range(1, len(sched["lr"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=sched["mix"], name="rehearsal mix α",
                             mode="lines+markers", line=dict(color="#2ca02c", width=2.5)))
    fig.add_trace(go.Scatter(x=k, y=sched["sigma"], name="kernel width σ",
                             mode="lines+markers", line=dict(color="#9467bd", width=2.5)))
    fig.add_trace(go.Scatter(x=k, y=sched["severity"], name="weakspot severity (in/out)",
                             mode="lines+markers", line=dict(color="#8c564b", width=1.6,
                                                             dash="dot")))
    fig.add_trace(go.Scatter(x=k, y=sched["lr"], name="learning rate", yaxis="y2",
                             mode="lines+markers", line=dict(color="#ff7f0e", width=2.5)))
    fig.update_layout(
        height=380, template="plotly_white", title="Scheduled controls per iteration",
        xaxis_title="iteration", yaxis_title="α · σ · severity",
        yaxis2=dict(title="learning rate", overlaying="y", side="right", type="log"),
        legend=dict(orientation="h", y=1.12))
    return fig


def detection_figure(det: dict) -> go.Figure:
    """Detection quality and weakspot migration per iteration.

    ``distance`` (to the induced centre) and ``drift`` (from the previous
    round's detection) share the left axis; IoU is on the right. A rising drift
    with a falling IoU is the loop losing its target as the gap heals.
    """
    k = list(range(1, len(det["distance"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=det["distance"], name="centroid distance ↓",
                             mode="lines+markers", line=dict(color="#d62728", width=2.4)))
    fig.add_trace(go.Scatter(x=k, y=det["drift"], name="drift from previous round",
                             mode="lines+markers", line=dict(color="#7f7f7f", width=1.8,
                                                             dash="dot")))
    fig.add_trace(go.Scatter(x=k, y=det["iou"], name="IoU @ q=0.90 ↑", yaxis="y2",
                             mode="lines+markers", line=dict(color="#1f77b4", width=2.4)))
    fig.update_layout(
        height=380, template="plotly_white",
        title="Weakspot identification across the loop",
        xaxis_title="iteration", yaxis_title="distance (unit square)",
        yaxis2=dict(title="IoU", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", y=1.12))
    return fig


def trajectory_figure(det: dict, center, radius: float) -> go.Figure:
    """Path of the detected weakspot centre over the loop, on the input square.

    Makes migration legible: a guided loop that works keeps re-aiming at the gap
    until it heals, then wanders off into whatever is now weakest.
    """
    cx, cy = det["cx"], det["cy"]
    fig = go.Figure()
    if radius and radius > 0:
        th = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(go.Scatter(
            x=center[0] + radius * np.cos(th), y=center[1] + radius * np.sin(th),
            mode="lines", line=dict(color="red", width=2, dash="dash"),
            name="induced weakspot"))
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="lines+markers+text",
        text=[str(i + 1) for i in range(len(cx))], textposition="top center",
        marker=dict(size=11, color=list(range(1, len(cx) + 1)), colorscale="Viridis",
                    showscale=True, colorbar=dict(title="iteration"),
                    line=dict(width=1, color="black")),
        line=dict(color="#7f00ff", width=1.4), name="detected centre"))
    fig.update_layout(height=470, template="plotly_white",
                      title="Detected weakspot centre, iteration by iteration",
                      xaxis=dict(range=[0, 1], title="x₁"),
                      yaxis=dict(range=[0, 1], title="x₂", scaleanchor="x"))
    return fig
