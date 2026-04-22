"""Plotly chart builders for reports."""

from __future__ import annotations

from typing import Any

from art_style_search.report_data import ReportData
from art_style_search.scoring import composite_score

_COLOR_BG = "#0e0e0c"
_COLOR_INK = "#f2eee6"
_COLOR_INK_MUTED = "#908a7c"
_COLOR_RULE = "#2a2823"
_COLOR_ACCENT = "#d9543a"
_COLOR_GOLD = "#c9a961"

_METRIC_SPECS: list[tuple[str, str]] = [
    ("dreamsim_similarity_mean", "DreamSim"),
    ("color_histogram_mean", "Color Histogram"),
    ("ssim_mean", "SSIM"),
    ("hps_score_mean", "HPS v2"),
    ("aesthetics_score_mean", "Aesthetics"),
    ("megastyle_similarity_mean", "MegaStyle"),
    ("style_consistency", "Style Consistency"),
    ("vision_style", "Vision · Style"),
    ("vision_subject", "Vision · Subject"),
    ("vision_composition", "Vision · Composition"),
    ("vision_medium", "Vision · Medium"),
    ("vision_proportions", "Vision · Proportions"),
]


def _metric_series(data: ReportData, extractor) -> tuple[list[int], list[float], list[float]]:
    """Return (iterations, best_per_iter, mean_per_iter) for a given metric extractor."""
    iters: list[int] = []
    best: list[float] = []
    mean: list[float] = []
    for iteration in data.iteration_numbers():
        results = data.iteration_logs[iteration]
        values = [extractor(result.aggregated) for result in results]
        if not values:
            continue
        iters.append(iteration)
        best.append(max(values))
        mean.append(sum(values) / len(values))
    return iters, best, mean


def _editorial_layout(**overrides) -> dict:
    """Shared Plotly layout for all charts."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='"IBM Plex Sans", system-ui, sans-serif', color=_COLOR_INK, size=12),
        title=dict(font=dict(family='"Fraunces", Georgia, serif', size=22, color=_COLOR_INK), x=0.0, xanchor="left"),
        xaxis=dict(
            gridcolor=_COLOR_RULE,
            linecolor=_COLOR_RULE,
            zerolinecolor=_COLOR_RULE,
            tickfont=dict(family='"IBM Plex Mono", monospace', size=11, color=_COLOR_INK_MUTED),
            title=dict(font=dict(family='"IBM Plex Sans", sans-serif', size=11, color=_COLOR_INK_MUTED)),
        ),
        yaxis=dict(
            gridcolor=_COLOR_RULE,
            linecolor=_COLOR_RULE,
            zerolinecolor=_COLOR_RULE,
            tickfont=dict(family='"IBM Plex Mono", monospace', size=11, color=_COLOR_INK_MUTED),
            title=dict(font=dict(family='"IBM Plex Sans", sans-serif', size=11, color=_COLOR_INK_MUTED)),
        ),
        legend=dict(
            font=dict(family='"IBM Plex Sans", sans-serif', size=11, color=_COLOR_INK),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=40, t=70, b=60),
        hoverlabel=dict(
            bgcolor=_COLOR_BG,
            bordercolor=_COLOR_ACCENT,
            font=dict(family='"IBM Plex Mono", monospace', color=_COLOR_INK),
        ),
    )
    base.update(overrides)
    return base


def _build_composite_trajectory(data: ReportData) -> str:
    """Hero chart: composite_score best + mean per iteration as JSON."""
    import plotly.graph_objects as go

    iters, best, mean = _metric_series(data, composite_score)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=best,
            mode="lines+markers",
            name="Best per iteration",
            line=dict(color=_COLOR_ACCENT, width=2.5, shape="spline", smoothing=0.4),
            marker=dict(size=9, color=_COLOR_ACCENT, line=dict(color=_COLOR_BG, width=2)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=mean,
            mode="lines+markers",
            name="Mean per iteration",
            line=dict(color=_COLOR_INK_MUTED, width=1.5, dash="2px,4px"),
            marker=dict(size=5, color=_COLOR_INK_MUTED, symbol="circle-open"),
        )
    )
    if data.state.global_best_metrics is not None:
        fig.add_hline(
            y=composite_score(data.state.global_best_metrics),
            line_color=_COLOR_GOLD,
            line_dash="dot",
            line_width=1,
            annotation_text="global best",
            annotation_position="top right",
            annotation_font=dict(family='"Fraunces", serif', size=11, color=_COLOR_GOLD),
        )
    fig.update_layout(
        **_editorial_layout(
            title="I. Composite Score",
            xaxis_title="iteration",
            yaxis_title="composite score",
            height=440,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(family='"IBM Plex Sans", sans-serif', size=11, color=_COLOR_INK),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    )
    return fig.to_json() or ""


def _build_per_metric_trajectories(data: ReportData) -> str:
    """4x3 subplot grid of the 12 component metrics (best + mean per iter)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=4,
        cols=3,
        subplot_titles=[title for _, title in _METRIC_SPECS],
        vertical_spacing=0.11,
        horizontal_spacing=0.09,
    )
    for idx, (attr, _title) in enumerate(_METRIC_SPECS):
        row = idx // 3 + 1
        col = idx % 3 + 1
        iters, best, mean = _metric_series(data, lambda metrics, attr_name=attr: getattr(metrics, attr_name))
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=best,
                mode="lines+markers",
                name="best",
                line=dict(color=_COLOR_ACCENT, width=2),
                marker=dict(size=5, color=_COLOR_ACCENT),
                showlegend=(idx == 0),
                legendgroup="best",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=mean,
                mode="lines+markers",
                name="mean",
                line=dict(color=_COLOR_INK_MUTED, width=1.2, dash="2px,4px"),
                marker=dict(size=4, color=_COLOR_INK_MUTED, symbol="circle-open"),
                showlegend=(idx == 0),
                legendgroup="mean",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        **_editorial_layout(
            title="II. Per-metric trajectories",
            height=860,
            margin=dict(l=60, r=40, t=100, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="left",
                x=0,
                font=dict(family='"IBM Plex Sans", sans-serif', size=11, color=_COLOR_INK),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    )
    fig.update_xaxes(
        gridcolor=_COLOR_RULE,
        linecolor=_COLOR_RULE,
        zerolinecolor=_COLOR_RULE,
        tickfont=dict(family='"IBM Plex Mono", monospace', size=10, color=_COLOR_INK_MUTED),
    )
    fig.update_yaxes(
        gridcolor=_COLOR_RULE,
        linecolor=_COLOR_RULE,
        zerolinecolor=_COLOR_RULE,
        tickfont=dict(family='"IBM Plex Mono", monospace', size=10, color=_COLOR_INK_MUTED),
    )
    layout: Any = fig["layout"]
    for annotation in layout["annotations"]:
        annotation["font"] = dict(family='"Fraunces", Georgia, serif', size=13, color=_COLOR_INK)
    fig.update_xaxes(title_text="iteration", row=3)
    return fig.to_json() or ""
