"""Post-run HTML report generator.

Builds a self-contained ``runs/<name>/report.html`` for a finished or paused
run.  Loads ``state.json`` plus all per-experiment iteration logs, then
renders four sections:

1. **Header** — run metadata, convergence status, best score, best prompt.
2. **Metric trajectories** — interactive Plotly charts of ``composite_score``
   and all 9 component metrics across iterations.
3. **Iteration drill-down** — one collapsible block per iteration containing
   an experiment table (hypothesis, category, score, kept) plus an image
   comparison grid of the winning experiment.
4. **Knowledge Base** — hypothesis tree with outcome colouring, category
   progress bars, and the current open-problems list.

Images are referenced via relative paths (no base64), so the report stays
small and fast to open.  Plotly is imported lazily so the ``list``/``clean``
commands don't pay the import cost.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

from art_style_search.scoring import adaptive_composite_score, composite_score
from art_style_search.state import load_iteration_log, load_state
from art_style_search.types import (
    CategoryProgress,
    Hypothesis,
    IterationResult,
    KnowledgeBase,
    LoopState,
    MetricScores,
    OpenProblem,
)

logger = logging.getLogger(__name__)

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
_FONTS_CDN = (
    "https://fonts.googleapis.com/css2"
    "?family=Fraunces:ital,opsz,wght@0,9..144,300..900;1,9..144,300..900"
    "&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400"
    "&family=IBM+Plex+Mono:wght@400;500"
    "&display=swap"
)
_MAX_TREE_DEPTH = 6
_LOG_PATTERN = re.compile(r"iter_(\d+)_branch_(\d+)\.json$")

# Editorial palette — used in CSS *and* cascaded into Plotly charts so the
# two visual languages stay consistent.
_COLOR_BG = "#0e0e0c"  # warm near-black
_COLOR_INK = "#f2eee6"  # cream primary text
_COLOR_INK_MUTED = "#908a7c"  # faded cream
_COLOR_RULE = "#2a2823"  # hairline dividers
_COLOR_ACCENT = "#d9543a"  # vermilion — winners, highlights
_COLOR_GOLD = "#c9a961"  # muted gold — global best context


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class ReportData:
    """Everything the renderer needs for one run."""

    run_name: str
    run_dir: Path
    state: LoopState
    iteration_logs: dict[int, list[IterationResult]] = field(default_factory=dict)

    def iteration_numbers(self) -> list[int]:
        """Sorted list of iteration indices that have at least one log."""
        return sorted(self.iteration_logs.keys())

    def winner_of(self, iteration: int) -> IterationResult | None:
        """Return the experiment with the highest ``composite_score`` for *iteration*."""
        results = self.iteration_logs.get(iteration, [])
        if not results:
            return None
        return max(results, key=lambda r: composite_score(r.aggregated))


def _load_iteration_logs(log_dir: Path) -> dict[int, list[IterationResult]]:
    """Parse every ``iter_NNN_branch_M.json`` under *log_dir*.

    Malformed files are logged and skipped — a partial crash mid-write
    shouldn't break the whole report.
    """
    result: dict[int, list[IterationResult]] = {}
    if not log_dir.is_dir():
        return result

    for path in sorted(log_dir.glob("iter_*_branch_*.json")):
        match = _LOG_PATTERN.search(path.name)
        if not match:
            continue
        try:
            record = load_iteration_log(path)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Skipping malformed iteration log %s: %s", path, exc)
            continue
        result.setdefault(record.iteration, []).append(record)

    for iteration_results in result.values():
        iteration_results.sort(key=lambda r: r.branch_id)
    return result


def load_report_data(run_dir: Path) -> ReportData:
    """Load *state.json* and all iteration logs from *run_dir*.

    Raises ``FileNotFoundError`` if the state file is missing (run not
    started).
    """
    state_file = run_dir / "state.json"
    state = load_state(state_file)
    if state is None:
        raise FileNotFoundError(f"No state.json found in {run_dir} — run not started yet")

    return ReportData(
        run_name=run_dir.name,
        run_dir=run_dir,
        state=state,
        iteration_logs=_load_iteration_logs(run_dir / "logs"),
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _rel(target: Path, report_dir: Path) -> str:
    """Return an ``<img src>``-safe relative path from *report_dir* to *target*.

    Falls back to a ``file://`` URI for cross-volume paths (Windows) where
    ``relpath`` raises ``ValueError``.
    """
    try:
        rel = os.path.relpath(target.resolve(), report_dir.resolve())
    except ValueError:
        return target.resolve().as_uri()
    # Use forward slashes even on Windows so the HTML is portable.
    return rel.replace(os.sep, "/")


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _metric_series(
    data: ReportData,
    extractor,
) -> tuple[list[int], list[float], list[float]]:
    """Return (iterations, best_per_iter, mean_per_iter) for a given metric extractor."""
    iters: list[int] = []
    best: list[float] = []
    mean: list[float] = []
    for i in data.iteration_numbers():
        results = data.iteration_logs[i]
        values = [extractor(r.aggregated) for r in results]
        if not values:
            continue
        iters.append(i)
        best.append(max(values))
        mean.append(sum(values) / len(values))
    return iters, best, mean


def _editorial_layout(**overrides) -> dict:
    """Shared Plotly layout for all charts — transparent bg, serif titles, cream ink."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family='"IBM Plex Sans", system-ui, sans-serif',
            color=_COLOR_INK,
            size=12,
        ),
        title=dict(
            font=dict(family='"Fraunces", Georgia, serif', size=22, color=_COLOR_INK),
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(
            gridcolor=_COLOR_RULE,
            linecolor=_COLOR_RULE,
            zerolinecolor=_COLOR_RULE,
            tickfont=dict(family='"IBM Plex Mono", monospace', size=11, color=_COLOR_INK_MUTED),
            title=dict(
                font=dict(family='"IBM Plex Sans", sans-serif', size=11, color=_COLOR_INK_MUTED),
            ),
        ),
        yaxis=dict(
            gridcolor=_COLOR_RULE,
            linecolor=_COLOR_RULE,
            zerolinecolor=_COLOR_RULE,
            tickfont=dict(family='"IBM Plex Mono", monospace', size=11, color=_COLOR_INK_MUTED),
            title=dict(
                font=dict(family='"IBM Plex Sans", sans-serif', size=11, color=_COLOR_INK_MUTED),
            ),
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
    return fig.to_json()


_METRIC_SPECS: list[tuple[str, str]] = [
    ("dreamsim_similarity_mean", "DreamSim"),
    ("color_histogram_mean", "Color Histogram"),
    ("ssim_mean", "SSIM"),
    ("hps_score_mean", "HPS v2"),
    ("aesthetics_score_mean", "Aesthetics"),
    ("style_consistency", "Style Consistency"),
    ("vision_style", "Vision · Style"),
    ("vision_subject", "Vision · Subject"),
    ("vision_composition", "Vision · Composition"),
]


def _build_per_metric_trajectories(data: ReportData) -> str:
    """3x3 subplot grid of the 9 component metrics (best + mean per iter)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    titles = [title for _, title in _METRIC_SPECS]
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=titles,
        vertical_spacing=0.14,
        horizontal_spacing=0.09,
    )

    for idx, (attr, _title) in enumerate(_METRIC_SPECS):
        row = idx // 3 + 1
        col = idx % 3 + 1
        iters, best, mean = _metric_series(data, lambda m, a=attr: getattr(m, a))
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
    # Cascade editorial axis styling into every subplot (update_layout only hits
    # the first by default for make_subplots figures).
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
    # Retitle subplot annotations with Fraunces italic, small caps feel.
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(
            family='"Fraunces", Georgia, serif',
            size=13,
            color=_COLOR_INK,
        )
    fig.update_xaxes(title_text="iteration", row=3)
    return fig.to_json()


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _h(text: str | None) -> str:
    """HTML-escape user-supplied text, treating ``None`` as empty."""
    return html.escape(text or "", quote=True)


def _fmt_score(value: float) -> str:
    return f"{value:.3f}"


def _metric_scores_tooltip(scores: MetricScores) -> str:
    parts = [
        f"DS={scores.dreamsim_similarity:.3f}",
        f"Color={scores.color_histogram:.3f}",
        f"SSIM={scores.ssim:.3f}",
        f"HPS={scores.hps_score:.3f}",
        f"Aes={scores.aesthetics_score:.1f}",
        f"V[S={scores.vision_style:.1f} Su={scores.vision_subject:.1f} Co={scores.vision_composition:.1f}]",
    ]
    return " ".join(parts)


def _per_image_score_for(result: IterationResult, gen_path: Path) -> MetricScores | None:
    """Find the ``MetricScores`` entry whose index matches the generated file's stem."""
    try:
        idx = int(gen_path.stem)
    except ValueError:
        return None
    if 0 <= idx < len(result.per_image_scores):
        return result.per_image_scores[idx]
    return None


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_header(data: ReportData) -> str:
    state = data.state
    iteration_count = max(data.iteration_numbers(), default=-1) + 1
    status = "in progress"
    if state.converged and state.convergence_reason is not None:
        status = f"converged · {state.convergence_reason.value.replace('_', ' ')}"

    best_score = "—"
    if state.global_best_metrics is not None:
        best_score = _fmt_score(composite_score(state.global_best_metrics))

    profile = state.style_profile
    profile_rows = [
        ("Color palette", profile.color_palette),
        ("Composition", profile.composition),
        ("Technique", profile.technique),
        ("Mood / atmosphere", profile.mood_atmosphere),
        ("Subject matter", profile.subject_matter),
        ("Influences", profile.influences),
    ]
    profile_html = "".join(
        f"<div class='kv-row'><dt>{_h(label)}</dt><dd>{_h(text)}</dd></div>" for label, text in profile_rows
    )

    return f"""
<header class="masthead">
  <div class="masthead-meta">
    <span>Art Style Search</span>
    <span class="meta-sep">·</span>
    <span>Post-run Report</span>
  </div>
  <h1 class="masthead-title">{_h(data.run_name)}</h1>
  <div class="masthead-rule"></div>
  <dl class="stats">
    <div class="stat stat--anim" style="--delay: 0ms">
      <dd class="stat-value">{iteration_count}</dd>
      <dt class="stat-label">iterations</dt>
    </div>
    <div class="stat stat--anim" style="--delay: 80ms">
      <dd class="stat-value stat-value--score">{best_score}</dd>
      <dt class="stat-label">best composite</dt>
    </div>
    <div class="stat stat--anim" style="--delay: 160ms">
      <dd class="stat-value stat-value--status">{_h(status)}</dd>
      <dt class="stat-label">status</dt>
    </div>
    <div class="stat stat--anim" style="--delay: 240ms">
      <dd class="stat-value">{len(state.fixed_references)}</dd>
      <dt class="stat-label">fixed references</dt>
    </div>
  </dl>
  <div class="preamble">
    <details class="fold">
      <summary><span class="fold-cue">§</span> Style profile</summary>
      <dl class="kv">{profile_html}</dl>
    </details>
    <details class="fold">
      <summary><span class="fold-cue">§</span> Best meta-prompt</summary>
      <pre class="prompt-block">{_h(state.global_best_prompt)}</pre>
    </details>
  </div>
</header>
"""


def _render_trajectories_section(composite_json: str, multi_json: str) -> str:
    if not composite_json:
        return (
            '<section class="trajectories">'
            '<div class="section-head"><span class="section-numeral">I</span>'
            "<h2>Metric trajectories</h2></div>"
            '<p class="empty">No iteration logs available yet.</p>'
            "</section>"
        )
    return f"""
<section class="trajectories">
  <div class="section-head">
    <span class="section-numeral">I</span>
    <h2>Metric trajectories</h2>
    <p class="section-kicker">Composite score and nine component metrics over the run.</p>
  </div>
  <figure class="chart-figure">
    <div id="composite-chart" class="chart"></div>
    <figcaption>Figure 1 · Composite score, best and mean per iteration. Gold hairline marks the global best.</figcaption>
  </figure>
  <figure class="chart-figure">
    <div id="metrics-chart" class="chart"></div>
    <figcaption>Figure 2 · All nine components of the composite score. Vermilion = best-of-iteration; cream = batch mean.</figcaption>
  </figure>
  <script type="application/json" id="composite-data">{composite_json}</script>
  <script type="application/json" id="metrics-data">{multi_json}</script>
</section>
"""


def _render_experiment_table(results: list[IterationResult]) -> str:
    if not results:
        return "<p class='empty'>No experiments logged.</p>"

    batch = [r.aggregated for r in results]
    winner_id = max(results, key=lambda r: composite_score(r.aggregated)).branch_id
    rows: list[str] = []
    for r in results:
        score = composite_score(r.aggregated)
        adaptive = adaptive_composite_score(r.aggregated, batch) if len(batch) >= 2 else None
        adaptive_cell = _fmt_score(adaptive) if adaptive is not None else "—"
        winner_mark = "<span class='winner-star'>✦</span>" if r.branch_id == winner_id else ""
        kept_cell = "kept" if r.kept else "cut"
        kept_class = "kept-yes" if r.kept else "kept-no"
        row_class = " class='winner-row'" if r.branch_id == winner_id else ""
        hyp = _h(r.hypothesis)
        truncated = hyp[:160] + ("…" if len(r.hypothesis) > 160 else "")
        rows.append(
            f"<tr{row_class}>"
            f"<td class='num-col'>{winner_mark}<span class='branch-id'>{r.branch_id:02d}</span></td>"
            f"<td class='hypothesis-col' title='{hyp}'>{truncated}</td>"
            f"<td class='num-col numeric'>{_fmt_score(score)}</td>"
            f"<td class='num-col numeric'>{adaptive_cell}</td>"
            f"<td class='kept-col {kept_class}'>{kept_cell}</td>"
            "</tr>"
        )
    return f"""
<table class="experiment-table">
  <thead>
    <tr>
      <th class="num-col">#</th>
      <th>Hypothesis</th>
      <th class="num-col numeric">composite</th>
      <th class="num-col numeric">adaptive</th>
      <th class="kept-col">verdict</th>
    </tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>
"""


def _render_image_grid(winner: IterationResult, report_dir: Path) -> str:
    if not winner.image_paths:
        return "<p class='empty'>No generated images recorded for the winning experiment.</p>"

    caption_by_idx = {i: c.image_path for i, c in enumerate(winner.iteration_captions)}
    pairs: list[tuple[int, Path, Path]] = []
    for gen_path in winner.image_paths:
        try:
            idx = int(gen_path.stem)
        except ValueError:
            continue
        ref = caption_by_idx.get(idx)
        if ref is not None:
            pairs.append((idx, ref, gen_path))

    if not pairs:
        return "<p class='empty'>Could not reconstruct reference/generated pairs.</p>"

    cards: list[str] = []
    for idx, ref, gen in pairs:
        scores = _per_image_score_for(winner, gen)
        tooltip = _metric_scores_tooltip(scores) if scores else ""
        ref_rel = _rel(ref, report_dir)
        gen_rel = _rel(gen, report_dir)
        ref_name = ref.stem[:28] + ("…" if len(ref.stem) > 28 else "")
        cards.append(
            f"""
<figure class="pair">
  <div class="pair-plate">
    <div class="plate-label">ref</div>
    <img src="{_h(ref_rel)}" alt="reference {idx}" loading="lazy">
  </div>
  <div class="pair-plate">
    <div class="plate-label">gen</div>
    <img src="{_h(gen_rel)}" alt="generated {idx}" title="{_h(tooltip)}" loading="lazy">
  </div>
  <figcaption>
    <span class="pair-num">{idx:02d}</span>
    <span class="pair-name">{_h(ref_name)}</span>
  </figcaption>
  <code class="score">{_h(tooltip)}</code>
</figure>"""
        )
    return f'<div class="ref-gen-grid">{"".join(cards)}</div>'


def _render_iteration_drilldown(data: ReportData, report_dir: Path) -> str:
    iterations = data.iteration_numbers()
    if not iterations:
        return (
            '<section class="iterations">'
            '<div class="section-head"><span class="section-numeral">II</span>'
            "<h2>Iterations</h2></div>"
            '<p class="empty">No iteration logs available yet.</p>'
            "</section>"
        )

    latest = iterations[-1]
    blocks: list[str] = []
    for i in iterations:
        results = data.iteration_logs[i]
        winner = data.winner_of(i)
        winner_score = _fmt_score(composite_score(winner.aggregated)) if winner else "—"
        experiment_table = _render_experiment_table(results)
        grid = _render_image_grid(winner, report_dir) if winner else ""
        narrative_blocks: list[str] = []
        if winner:
            for label, text in (
                ("Winning hypothesis", winner.hypothesis),
                ("Experiment", winner.experiment),
                ("Template changes", winner.template_changes),
                ("Claude analysis", winner.claude_analysis),
                ("Vision feedback", winner.vision_feedback),
                ("Round-trip feedback", winner.roundtrip_feedback),
            ):
                if text and text.strip():
                    narrative_blocks.append(
                        f"<details class='narrative'><summary>{_h(label)}</summary><pre>{_h(text)}</pre></details>"
                    )
        open_attr = " open" if i == latest else ""
        blocks.append(
            f"""
<details class="iteration"{open_attr}>
  <summary>
    <span class="iter-number">Iteration {i:02d}</span>
    <span class="iter-sep">—</span>
    <span class="iter-count">{len(results)} experiments</span>
    <span class="iter-score">{winner_score}</span>
  </summary>
  <div class="iteration-body">
    {experiment_table}
    {"".join(narrative_blocks)}
    {grid}
  </div>
</details>"""
        )
    return f"""
<section class="iterations">
  <div class="section-head">
    <span class="section-numeral">II</span>
    <h2>Iterations</h2>
    <p class="section-kicker">Per-iteration experiments with their hypotheses, scores, and the winner's reference / generated pairs.</p>
  </div>
  {"".join(blocks)}
</section>
"""


def _render_category_progress(kb: KnowledgeBase) -> str:
    if not kb.categories:
        return "<p class='empty'>No category progress recorded yet.</p>"

    rows: list[str] = []
    for name in sorted(kb.categories):
        cat: CategoryProgress = kb.categories[name]
        n_total = len(cat.hypothesis_ids)
        n_confirmed = len(cat.confirmed_insights)
        n_rejected = len(cat.rejected_approaches)
        confirmed_pct = (n_confirmed / max(n_total, 1)) * 100
        rejected_pct = (n_rejected / max(n_total, 1)) * 100
        delta_str = f"Δ {cat.best_perceptual_delta:+.3f}" if cat.best_perceptual_delta is not None else "—"
        display_name = name.replace("_", " ")
        rows.append(
            f"""
<div class="cat-row">
  <div class="cat-name">{_h(display_name)}</div>
  <div class="cat-bar" role="img" aria-label="{n_confirmed} confirmed, {n_rejected} rejected of {n_total}">
    <div class="cat-bar-confirmed" style="width: {confirmed_pct:.0f}%"></div>
    <div class="cat-bar-rejected" style="width: {rejected_pct:.0f}%"></div>
  </div>
  <div class="cat-meta">
    <span class="cat-count"><b>{n_total}</b> hyp</span>
    <span class="cat-count"><b>{n_confirmed}</b> confirmed</span>
    <span class="cat-count"><b>{n_rejected}</b> rejected</span>
    <span class="cat-delta">{_h(delta_str)}</span>
  </div>
</div>"""
        )
    return f'<div class="category-bars">{"".join(rows)}</div>'


def _render_hypothesis_tree(kb: KnowledgeBase) -> str:
    if not kb.hypotheses:
        return "<p class='empty'>No hypotheses recorded yet.</p>"

    by_id: dict[str, Hypothesis] = {h.id: h for h in kb.hypotheses}
    children_map: dict[str, list[Hypothesis]] = {}
    roots: list[Hypothesis] = []
    for h in kb.hypotheses:
        if h.parent_id and h.parent_id in by_id:
            children_map.setdefault(h.parent_id, []).append(h)
        else:
            roots.append(h)

    def _render_meta(h: Hypothesis) -> str:
        return (
            f"<span class='hyp-id'>{_h(h.id)}</span>"
            f"<span class='hyp-sep'>·</span>"
            f"<span class='hyp-iter'>iter {h.iteration}</span>"
            f"<span class='hyp-sep'>·</span>"
            f"<span class='hyp-category'>{_h(h.category.replace('_', ' '))}</span>"
            f"<span class='hyp-sep'>·</span>"
            f"<span class='hyp-outcome'>{_h(h.outcome)}</span>"
        )

    def _render_node(h: Hypothesis, depth: int) -> str:
        children = children_map.get(h.id, [])
        css_class = f"hyp hyp-{_h(h.outcome)}"
        statement = _h(h.statement)
        lesson = f"<div class='hyp-lesson'>{_h(h.lesson)}</div>" if h.lesson else ""
        meta = _render_meta(h)

        if depth >= _MAX_TREE_DEPTH and children:
            descendant_count = _count_descendants(h.id, children_map)
            return (
                f"<div class='{css_class}'>"
                f"<div class='hyp-meta'>{meta}</div>"
                f"<div class='hyp-statement'>{statement}</div>"
                f"{lesson}"
                f"<div class='hyp-deeper'>(+{descendant_count} deeper — collapsed)</div>"
                "</div>"
            )
        inner = "".join(_render_node(c, depth + 1) for c in children)
        if children:
            return (
                f"<details class='{css_class}' open>"
                f"<summary><span class='hyp-meta'>{meta}</span></summary>"
                f"<div class='hyp-statement'>{statement}</div>"
                f"{lesson}"
                f"<div class='hyp-children'>{inner}</div>"
                "</details>"
            )
        return (
            f"<div class='{css_class}'>"
            f"<div class='hyp-meta'>{meta}</div>"
            f"<div class='hyp-statement'>{statement}</div>"
            f"{lesson}"
            "</div>"
        )

    tree_html = "".join(_render_node(r, 0) for r in roots)
    return f'<div class="hypothesis-tree">{tree_html}</div>'


def _count_descendants(node_id: str, children_map: dict[str, list[Hypothesis]]) -> int:
    total = 0
    stack = list(children_map.get(node_id, []))
    while stack:
        child = stack.pop()
        total += 1
        stack.extend(children_map.get(child.id, []))
    return total


def _render_open_problems(problems: list[OpenProblem]) -> str:
    if not problems:
        return "<p class='empty'>No open problems.</p>"

    items: list[str] = []
    for idx, p in enumerate(problems, start=1):
        gap = f"{p.metric_gap:+.3f}" if p.metric_gap is not None else "—"
        priority = p.priority or "LOW"
        items.append(
            f"<li class='prio-{_h(priority.lower())}'>"
            f"<span class='prob-num'>{idx:02d}</span>"
            f"<span class='prio-chip'>{_h(priority)}</span>"
            f"<span class='prob-text'>{_h(p.text)}</span>"
            f"<span class='prob-meta'>"
            f"<span>{_h(p.category.replace('_', ' '))}</span>"
            f"<span>iter {p.since_iteration}</span>"
            f"<span class='prob-gap'>gap {gap}</span>"
            "</span>"
            "</li>"
        )
    return f"<ol class='open-problems'>{''.join(items)}</ol>"


def _render_kb_section(data: ReportData) -> str:
    kb = data.state.knowledge_base
    return f"""
<section class="kb-section">
  <div class="section-head">
    <span class="section-numeral">III</span>
    <h2>Knowledge Base</h2>
    <p class="section-kicker">Hypotheses tried, what confirmed or rejected them, and the open problems still worth attacking.</p>
  </div>
  <div class="kb-sub">
    <h3>Category progress</h3>
    {_render_category_progress(kb)}
  </div>
  <div class="kb-sub">
    <h3>Hypothesis tree</h3>
    {_render_hypothesis_tree(kb)}
  </div>
  <div class="kb-sub">
    <h3>Open problems</h3>
    {_render_open_problems(kb.open_problems)}
  </div>
</section>
"""


# ---------------------------------------------------------------------------
# HTML assembler
# ---------------------------------------------------------------------------


_CSS = """
:root {
  --bg: #0e0e0c;
  --bg-sunk: #0a0a09;
  --panel: rgba(242, 238, 230, 0.02);
  --ink: #f2eee6;
  --ink-muted: #908a7c;
  --ink-faint: #5c5850;
  --rule: #2a2823;
  --rule-strong: #3d3a33;
  --accent: #d9543a;
  --accent-dim: #8a3324;
  --gold: #c9a961;
  --max-w: 1200px;
  --serif: "Fraunces", Georgia, "Times New Roman", serif;
  --sans: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Consolas, monospace;
}

* { box-sizing: border-box; }

html, body {
  background: var(--bg);
  color: var(--ink);
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--sans);
  font-weight: 400;
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
  background-image:
    radial-gradient(ellipse 1200px 600px at 20% -10%, rgba(217, 84, 58, 0.06), transparent 60%),
    radial-gradient(ellipse 900px 500px at 100% 10%, rgba(201, 169, 97, 0.035), transparent 55%);
  background-attachment: fixed;
  min-height: 100vh;
}

.page {
  max-width: var(--max-w);
  margin: 0 auto;
  padding: 72px 56px 96px;
}

::selection {
  background: var(--accent);
  color: var(--bg);
}

/* ---------- Masthead ---------- */

.masthead {
  padding-bottom: 56px;
  margin-bottom: 64px;
  border-bottom: 1px solid var(--rule);
  position: relative;
}

.masthead-meta {
  font-family: var(--sans);
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-bottom: 28px;
  display: flex;
  gap: 12px;
  align-items: center;
}
.meta-sep { color: var(--ink-faint); }

.masthead-title {
  font-family: var(--serif);
  font-optical-sizing: auto;
  font-variation-settings: "opsz" 144, "wght" 500;
  font-size: clamp(64px, 11vw, 140px);
  line-height: 0.92;
  letter-spacing: -0.035em;
  margin: 0 0 32px;
  color: var(--ink);
  word-break: break-word;
  font-style: italic;
}

.masthead-rule {
  height: 1px;
  background: linear-gradient(to right, var(--accent) 0%, var(--accent) 80px, var(--rule) 80px, var(--rule) 100%);
  margin: 32px 0 40px;
}

.stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 32px;
  margin: 0 0 48px;
}
.stat {
  display: flex;
  flex-direction: column;
  border-left: 1px solid var(--rule);
  padding-left: 20px;
}
.stat:first-child { border-left-color: var(--accent); }

.stat-value {
  font-family: var(--serif);
  font-variation-settings: "opsz" 72, "wght" 400;
  font-size: 48px;
  line-height: 1;
  color: var(--ink);
  margin: 0;
  letter-spacing: -0.02em;
}
.stat-value--score {
  font-family: var(--mono);
  font-size: 40px;
  font-weight: 500;
  letter-spacing: -0.01em;
  color: var(--accent);
}
.stat-value--status {
  font-family: var(--serif);
  font-style: italic;
  font-size: 24px;
  color: var(--gold);
  line-height: 1.2;
  font-variation-settings: "opsz" 24, "wght" 400;
}

.stat-label {
  font-family: var(--sans);
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-top: 14px;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
.stat--anim {
  opacity: 0;
  animation: rise 0.7s cubic-bezier(0.2, 0.7, 0.2, 1) forwards;
  animation-delay: var(--delay, 0ms);
}

/* ---------- Preamble (foldable sections) ---------- */

.preamble { display: flex; flex-direction: column; gap: 4px; }

.fold {
  border-top: 1px solid var(--rule);
  padding: 18px 0;
}
.fold:last-child { border-bottom: 1px solid var(--rule); }
.fold > summary {
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: baseline;
  gap: 14px;
  font-family: var(--serif);
  font-style: italic;
  font-size: 18px;
  color: var(--ink);
  font-variation-settings: "opsz" 18, "wght" 400;
  transition: color 0.2s;
}
.fold > summary::-webkit-details-marker { display: none; }
.fold > summary:hover { color: var(--accent); }
.fold-cue {
  font-family: var(--serif);
  font-style: normal;
  font-size: 16px;
  color: var(--accent);
  font-weight: 500;
}
.fold[open] > summary { margin-bottom: 16px; }

.kv {
  margin: 0;
  display: grid;
  grid-template-columns: 200px 1fr;
  gap: 12px 32px;
}
.kv-row { display: contents; }
.kv-row dt {
  font-family: var(--sans);
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-muted);
  padding-top: 5px;
  margin: 0;
}
.kv-row dd {
  font-family: var(--serif);
  font-size: 15px;
  color: var(--ink);
  margin: 0;
  line-height: 1.5;
}

.prompt-block {
  background: var(--bg-sunk);
  padding: 24px 28px;
  border-left: 2px solid var(--accent);
  overflow-x: auto;
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.65;
  max-height: 560px;
  color: var(--ink);
  white-space: pre-wrap;
  margin: 0;
}

/* ---------- Section heads ---------- */

section {
  margin-bottom: 96px;
}

.section-head {
  display: grid;
  grid-template-columns: 100px 1fr;
  grid-template-rows: auto auto;
  gap: 8px 28px;
  align-items: baseline;
  margin-bottom: 48px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--rule);
}
.section-numeral {
  grid-row: 1 / 3;
  font-family: var(--serif);
  font-variation-settings: "opsz" 144, "wght" 300;
  font-style: italic;
  font-size: 88px;
  line-height: 0.9;
  color: var(--accent);
  text-align: right;
}
.section-head h2 {
  font-family: var(--serif);
  font-variation-settings: "opsz" 72, "wght" 500;
  font-size: 44px;
  line-height: 1;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 0;
}
.section-kicker {
  grid-column: 2;
  font-family: var(--serif);
  font-style: italic;
  font-size: 16px;
  color: var(--ink-muted);
  margin: 0;
  max-width: 640px;
  line-height: 1.5;
}

h3 {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin: 0 0 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--rule);
}

.empty {
  font-family: var(--serif);
  font-style: italic;
  color: var(--ink-muted);
  font-size: 16px;
}

/* ---------- Charts ---------- */

.chart-figure { margin: 0 0 48px; padding: 0; }
.chart { margin-bottom: 12px; }
.chart-figure figcaption {
  font-family: var(--serif);
  font-style: italic;
  font-size: 13px;
  color: var(--ink-muted);
  padding: 0 8px;
  border-left: 2px solid var(--rule);
  margin-left: 8px;
  line-height: 1.5;
  max-width: 780px;
}

/* ---------- Experiment table ---------- */

.experiment-table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 32px;
  font-family: var(--sans);
}
.experiment-table thead th {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-muted);
  text-align: left;
  padding: 14px 18px 14px 0;
  border-bottom: 1px solid var(--rule-strong);
}
.experiment-table th.numeric,
.experiment-table td.numeric { text-align: right; font-family: var(--mono); }
.experiment-table th.num-col,
.experiment-table td.num-col { width: 1%; white-space: nowrap; padding-right: 24px; }
.experiment-table th.kept-col,
.experiment-table td.kept-col { width: 1%; text-align: right; padding-left: 24px; padding-right: 0; }

.experiment-table tbody td {
  padding: 18px 18px 18px 0;
  border-bottom: 1px solid var(--rule);
  vertical-align: baseline;
  font-size: 14px;
  color: var(--ink);
}
.experiment-table tbody tr:last-child td { border-bottom: none; }
.experiment-table td.hypothesis-col {
  font-family: var(--serif);
  font-size: 15px;
  line-height: 1.5;
  color: var(--ink);
  max-width: 620px;
}
.experiment-table tr.winner-row td {
  color: var(--ink);
  background: linear-gradient(to right, rgba(217, 84, 58, 0.08), transparent 60%);
}
.experiment-table tr.winner-row td.hypothesis-col { font-style: italic; }

.branch-id {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--ink-muted);
  letter-spacing: 0.04em;
}
.winner-star {
  display: inline-block;
  color: var(--accent);
  margin-right: 8px;
  font-size: 13px;
}
.kept-yes {
  font-family: var(--sans);
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 500;
}
.kept-no {
  font-family: var(--sans);
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-faint);
}

/* ---------- Iteration drilldown ---------- */

.iteration {
  margin-bottom: 1px;
  border-top: 1px solid var(--rule);
}
.iterations > .iteration:last-of-type { border-bottom: 1px solid var(--rule); }
.iteration > summary {
  cursor: pointer;
  list-style: none;
  padding: 26px 0;
  display: grid;
  grid-template-columns: auto auto 1fr auto;
  gap: 18px;
  align-items: baseline;
  transition: padding-left 0.25s;
}
.iteration > summary:hover { padding-left: 16px; }
.iteration > summary::-webkit-details-marker { display: none; }
.iteration[open] > summary { padding-left: 0; }
.iteration[open] > summary:hover { padding-left: 0; }

.iter-number {
  font-family: var(--serif);
  font-variation-settings: "opsz" 48, "wght" 500;
  font-style: italic;
  font-size: 32px;
  line-height: 1;
  color: var(--ink);
  letter-spacing: -0.01em;
}
.iter-sep { color: var(--ink-faint); font-size: 22px; }
.iter-count {
  font-family: var(--sans);
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-muted);
}
.iter-score {
  font-family: var(--mono);
  font-size: 18px;
  color: var(--accent);
  font-weight: 500;
  justify-self: end;
}
.iteration-body {
  padding: 8px 0 40px;
  display: flex;
  flex-direction: column;
  gap: 32px;
}

.narrative { margin: 0; }
.narrative > summary {
  cursor: pointer;
  list-style: none;
  font-family: var(--sans);
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-muted);
  padding: 10px 0;
  border-top: 1px solid var(--rule);
  transition: color 0.2s;
}
.narrative > summary::-webkit-details-marker { display: none; }
.narrative > summary::before { content: "— "; color: var(--accent); }
.narrative > summary:hover { color: var(--accent); }
.narrative pre {
  background: var(--bg-sunk);
  border-left: 2px solid var(--rule);
  padding: 16px 20px;
  margin: 8px 0 16px;
  font-family: var(--mono);
  font-size: 11px;
  line-height: 1.6;
  color: var(--ink);
  white-space: pre-wrap;
  overflow-x: auto;
}

/* ---------- Image pair grid ---------- */

.ref-gen-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 28px 24px;
  margin-top: 8px;
}
.pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  margin: 0;
}
.pair-plate {
  position: relative;
  background: var(--bg-sunk);
  border: 1px solid var(--rule);
  aspect-ratio: 1 / 1;
  overflow: hidden;
}
.plate-label {
  position: absolute;
  top: 8px;
  left: 8px;
  font-family: var(--sans);
  font-size: 9px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink);
  background: rgba(14, 14, 12, 0.82);
  padding: 3px 8px;
  z-index: 1;
  backdrop-filter: blur(4px);
}
.pair-plate:first-child .plate-label { color: var(--gold); }
.pair-plate:nth-child(2) .plate-label { color: var(--accent); }
.pair img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  transition: transform 0.5s cubic-bezier(0.2, 0.7, 0.2, 1);
}
.pair:hover img { transform: scale(1.02); }
.pair figcaption {
  grid-column: 1 / 3;
  display: flex;
  gap: 12px;
  align-items: baseline;
  padding-top: 10px;
  border-top: 1px solid var(--rule);
  margin-top: 2px;
}
.pair-num {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--accent);
  font-weight: 500;
}
.pair-name {
  font-family: var(--serif);
  font-style: italic;
  font-size: 13px;
  color: var(--ink);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}
.pair .score {
  grid-column: 1 / 3;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--ink-muted);
  padding: 2px 0 0;
  line-height: 1.6;
  letter-spacing: 0.02em;
}

/* ---------- KB section ---------- */

.kb-sub { margin-bottom: 56px; }
.kb-sub:last-child { margin-bottom: 0; }

.category-bars { display: flex; flex-direction: column; gap: 20px; }
.cat-row {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 24px;
  align-items: center;
}
.cat-name {
  font-family: var(--serif);
  font-style: italic;
  font-size: 17px;
  color: var(--ink);
  font-variation-settings: "opsz" 18, "wght" 400;
}
.cat-bar {
  height: 3px;
  background: var(--rule);
  display: flex;
  position: relative;
}
.cat-bar-confirmed {
  background: var(--accent);
  height: 100%;
}
.cat-bar-rejected {
  background: var(--ink-faint);
  height: 100%;
}
.cat-meta {
  grid-column: 2;
  display: flex;
  gap: 24px;
  font-family: var(--sans);
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-top: 8px;
}
.cat-count b {
  font-family: var(--mono);
  font-weight: 500;
  color: var(--ink);
  margin-right: 4px;
  letter-spacing: 0;
}
.cat-delta {
  font-family: var(--mono);
  color: var(--accent);
  letter-spacing: 0.02em;
  text-transform: none;
  margin-left: auto;
}

/* ---------- Hypothesis tree ---------- */

.hypothesis-tree { display: flex; flex-direction: column; gap: 2px; }
.hyp, details.hyp {
  border-left: 1px solid var(--rule);
  padding: 14px 0 14px 24px;
  margin: 0;
  position: relative;
}
.hyp-confirmed, details.hyp-confirmed { border-left-color: var(--accent); }
.hyp-rejected, details.hyp-rejected { border-left-color: var(--ink-faint); }
.hyp-partial, details.hyp-partial { border-left-color: var(--gold); }

details.hyp > summary {
  cursor: pointer;
  list-style: none;
  padding: 0;
  margin-bottom: 8px;
}
details.hyp > summary::-webkit-details-marker { display: none; }

.hyp-meta {
  font-family: var(--sans);
  font-size: 9px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--ink-muted);
  display: flex;
  gap: 8px;
  align-items: baseline;
  margin-bottom: 8px;
}
.hyp-id {
  font-family: var(--mono);
  color: var(--accent);
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: none;
}
.hyp-sep { color: var(--ink-faint); }
.hyp-iter { font-family: var(--mono); text-transform: none; letter-spacing: 0.04em; }
.hyp-category { color: var(--ink); }
.hyp-outcome { color: var(--gold); }
.hyp-confirmed .hyp-outcome { color: var(--accent); }
.hyp-rejected .hyp-outcome { color: var(--ink-faint); }

.hyp-statement {
  font-family: var(--serif);
  font-size: 15px;
  line-height: 1.5;
  color: var(--ink);
  margin-bottom: 6px;
}
.hyp-lesson {
  font-family: var(--serif);
  font-style: italic;
  font-size: 13px;
  color: var(--ink-muted);
  padding-top: 4px;
}
.hyp-deeper {
  font-family: var(--sans);
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin-top: 10px;
}
.hyp-children {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

/* ---------- Open problems ---------- */

.open-problems {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
}
.open-problems li {
  display: grid;
  grid-template-columns: 56px 88px 1fr;
  gap: 24px;
  padding: 22px 0;
  border-bottom: 1px solid var(--rule);
  align-items: baseline;
}
.open-problems li:first-child { border-top: 1px solid var(--rule); }
.prob-num {
  font-family: var(--serif);
  font-variation-settings: "opsz" 48, "wght" 300;
  font-style: italic;
  font-size: 36px;
  color: var(--ink-faint);
  line-height: 1;
  text-align: right;
}
.prio-chip {
  font-family: var(--sans);
  font-size: 9px;
  letter-spacing: 0.22em;
  font-weight: 500;
  text-transform: uppercase;
  padding: 4px 10px;
  border: 1px solid currentColor;
  display: inline-block;
  text-align: center;
  justify-self: start;
}
.prio-high .prio-chip { color: var(--accent); }
.prio-high .prob-num { color: var(--accent); }
.prio-med .prio-chip { color: var(--gold); }
.prio-low .prio-chip { color: var(--ink-muted); }
.prob-text {
  font-family: var(--serif);
  font-size: 17px;
  line-height: 1.45;
  color: var(--ink);
  grid-column: 3;
  grid-row: 1;
}
.prob-meta {
  grid-column: 3;
  grid-row: 2;
  display: flex;
  gap: 16px;
  font-family: var(--sans);
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin-top: 8px;
}
.prob-gap {
  font-family: var(--mono);
  text-transform: none;
  letter-spacing: 0.02em;
  color: var(--accent);
  margin-left: auto;
}

/* ---------- Responsive ---------- */

@media (max-width: 880px) {
  .page { padding: 48px 28px 64px; }
  .stats { grid-template-columns: repeat(2, 1fr); gap: 32px 20px; }
  .kv { grid-template-columns: 1fr; gap: 4px 0; }
  .kv-row { display: block; margin-bottom: 16px; }
  .kv-row dt { padding-top: 0; margin-bottom: 4px; }
  .section-head { grid-template-columns: 56px 1fr; gap: 8px 16px; }
  .section-numeral { font-size: 56px; }
  .section-head h2 { font-size: 32px; }
  .cat-row { grid-template-columns: 1fr; gap: 8px; }
  .cat-meta { margin-top: 4px; }
  .open-problems li { grid-template-columns: 40px 1fr; gap: 16px; }
  .prob-text, .prob-meta { grid-column: 2; }
  .prio-chip { grid-column: 2; grid-row: 2; justify-self: start; }
  .prob-text { grid-row: 1; }
  .prob-meta { grid-row: 3; }
  .iter-number { font-size: 24px; }
  .iteration > summary { grid-template-columns: auto auto; gap: 12px; row-gap: 4px; }
  .iter-sep { display: none; }
}

/* --- Scientific rigor sections --- */
.protocol-badge-row {
  display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
  padding: 16px 0; margin-bottom: 16px;
}
.protocol-badge {
  font-family: var(--mono); font-size: 13px; font-weight: 700;
  padding: 4px 12px; border-radius: 4px; letter-spacing: 0.05em;
}
.badge-classic { background: var(--panel); color: var(--ink-muted); border: 1px solid var(--rule); }
.badge-rigorous { background: rgba(201,169,97,0.15); color: var(--gold); border: 1px solid var(--gold); }
.manifest-item { font-family: var(--mono); font-size: 13px; color: var(--ink-muted); }
.manifest-item code { color: var(--ink); }

.promotion-table .promo-yes td:last-child { color: var(--accent); font-weight: 600; }
.promotion-table .promo-explore td:last-child { color: var(--gold); font-weight: 600; }
.promotion-table .promo-no td:last-child { color: var(--ink-muted); }
.decision-cell { text-transform: uppercase; font-family: var(--mono); font-size: 12px; letter-spacing: 0.05em; }

.holdout-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;
  padding: 24px 0;
}
.holdout-card {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 8px;
  padding: 20px; text-align: center;
}
.holdout-label { font-family: var(--sans); font-size: 13px; color: var(--ink-muted); margin-bottom: 8px; }
.holdout-value { font-family: var(--mono); font-size: 24px; color: var(--ink); }
.holdout-up { color: var(--accent); }
.holdout-down { color: var(--ink-muted); }
.holdout-images { font-family: var(--mono); font-size: 12px; color: var(--ink-muted); word-break: break-all; }
"""


# ---------------------------------------------------------------------------
# Scientific-rigor report sections (Phase 2C + Phase 5)
# ---------------------------------------------------------------------------


def _render_protocol_section(data: ReportData) -> str:
    """Render protocol metadata badge + manifest info (Phase 5D)."""
    from art_style_search.state import load_manifest

    manifest_path = data.run_dir / "run_manifest.json"
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return ""

    badge_class = "badge-rigorous" if "rigorous" in manifest.protocol_version else "badge-classic"
    badge_label = manifest.protocol_version.upper().replace("_", " ")

    git_line = (
        f'<span class="manifest-item">Git: <code>{_h(manifest.git_sha[:10])}</code></span>' if manifest.git_sha else ""
    )

    return f"""
<section class="protocol-section">
  <div class="section-head">
    <span class="section-numeral">IV</span>
    <h2>Protocol</h2>
    <p class="section-kicker">Run provenance and scientific rigor settings.</p>
  </div>
  <div class="protocol-badge-row">
    <span class="protocol-badge {badge_class}">{badge_label}</span>
    <span class="manifest-item">Seed: <code>{manifest.seed}</code></span>
    {git_line}
    <span class="manifest-item">Refs: <code>{manifest.num_fixed_refs}</code></span>
  </div>
  <details class="fold">
    <summary>Models &amp; config</summary>
    <dl class="kv">
      <dt>Caption model</dt><dd>{_h(manifest.model_names.get("caption_model", ""))}</dd>
      <dt>Generator model</dt><dd>{_h(manifest.model_names.get("generator_model", ""))}</dd>
      <dt>Reasoning model</dt><dd>{_h(manifest.model_names.get("reasoning_model", ""))}</dd>
      <dt>Provider</dt><dd>{_h(manifest.reasoning_provider)}</dd>
      <dt>Platform</dt><dd>{_h(manifest.platform)}</dd>
      <dt>Python</dt><dd>{_h(manifest.python_version.split()[0] if manifest.python_version else "")}</dd>
      <dt>Timestamp</dt><dd>{_h(manifest.timestamp_utc)}</dd>
    </dl>
  </details>
</section>
"""


def _render_promotion_section(data: ReportData) -> str:
    """Render promotion decision table (Phase 5A)."""
    from art_style_search.state import load_promotion_log

    log_path = data.run_dir / "promotion_log.jsonl"
    decisions = load_promotion_log(log_path)
    if not decisions:
        return ""

    rows: list[str] = []
    for d in decisions:
        css_class = {"promoted": "promo-yes", "exploration": "promo-explore", "rejected": "promo-no"}.get(
            d.decision, ""
        )
        p_cell = f"{d.p_value:.4f}" if d.p_value is not None else "—"
        effect_cell = f"{d.delta:+.5f}"
        rows.append(
            f'<tr class="{css_class}">'
            f"<td>{d.iteration + 1}</td>"
            f"<td>{d.candidate_branch_id}</td>"
            f"<td>{_fmt_score(d.baseline_score)}</td>"
            f"<td>{_fmt_score(d.candidate_score)}</td>"
            f"<td>{effect_cell}</td>"
            f"<td>{_fmt_score(d.epsilon)}</td>"
            f"<td>{p_cell}</td>"
            f'<td class="decision-cell">{_h(d.decision)}</td>'
            f"</tr>"
        )

    n_promoted = sum(1 for d in decisions if d.decision == "promoted")
    n_explored = sum(1 for d in decisions if d.decision == "exploration")

    return f"""
<section class="promotion-section">
  <div class="section-head">
    <span class="section-numeral">V</span>
    <h2>Promotion Decisions</h2>
    <p class="section-kicker">{n_promoted} promoted, {n_explored} explorations, {len(decisions) - n_promoted - n_explored} rejected.</p>
  </div>
  <div class="table-wrap">
    <table class="experiment-table promotion-table">
      <thead>
        <tr><th>Iter</th><th>Exp</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>&epsilon;</th><th>p-value</th><th>Decision</th></tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
  </div>
</section>
"""


def _render_holdout_section(data: ReportData) -> str:
    """Render silent-image holdout summary (Phase 2C)."""
    holdout_path = data.run_dir / "holdout_summary.json"
    if not holdout_path.exists():
        return ""

    summary = json.loads(holdout_path.read_text(encoding="utf-8"))
    n_silent = summary.get("silent_image_count", 0)
    if n_silent == 0:
        return ""

    iter0_mean = summary.get("iteration_0_mean")
    final_mean = summary.get("final_mean")
    delta = summary.get("delta")

    iter0_str = f"{iter0_mean:.4f}" if iter0_mean is not None else "—"
    final_str = f"{final_mean:.4f}" if final_mean is not None else "—"
    if delta is not None:
        arrow = "&#9650;" if delta > 0 else "&#9660;" if delta < 0 else "="
        delta_str = f"{delta:+.4f} {arrow}"
        delta_class = "holdout-up" if delta > 0 else "holdout-down" if delta < 0 else ""
    else:
        delta_str = "—"
        delta_class = ""

    image_names = ", ".join(summary.get("silent_image_names", []))

    return f"""
<section class="holdout-section">
  <div class="section-head">
    <span class="section-numeral">VI</span>
    <h2>Silent-Image Holdout</h2>
    <p class="section-kicker">{n_silent} images were never shown to the optimizer — improvements here indicate genuine generalization.</p>
  </div>
  <div class="holdout-grid">
    <div class="holdout-card">
      <div class="holdout-label">Iteration 0</div>
      <div class="holdout-value">{iter0_str}</div>
    </div>
    <div class="holdout-card">
      <div class="holdout-label">Final</div>
      <div class="holdout-value">{final_str}</div>
    </div>
    <div class="holdout-card">
      <div class="holdout-label">Delta</div>
      <div class="holdout-value {delta_class}">{delta_str}</div>
    </div>
  </div>
  <details class="fold">
    <summary>Silent images</summary>
    <p class="holdout-images">{_h(image_names)}</p>
  </details>
</section>
"""


def _assemble_html(data: ReportData, report_dir: Path) -> str:
    composite_json = ""
    multi_json = ""
    if data.iteration_logs:
        composite_json = _build_composite_trajectory(data)
        multi_json = _build_per_metric_trajectories(data)

    header = _render_header(data)
    trajectories = _render_trajectories_section(composite_json, multi_json)
    iterations_section = _render_iteration_drilldown(data, report_dir)
    kb_section = _render_kb_section(data)
    protocol_section = _render_protocol_section(data)
    promotion_section = _render_promotion_section(data)
    holdout_section = _render_holdout_section(data)

    plot_script = ""
    if composite_json and multi_json:
        plot_script = """
<script>
(function() {
  function parseAndPlot(elemId, dataId) {
    var el = document.getElementById(dataId);
    if (!el) return;
    try {
      var fig = JSON.parse(el.textContent);
      Plotly.newPlot(elemId, fig.data, fig.layout, {responsive: true, displaylogo: false});
    } catch (e) {
      console.error("Failed to render " + elemId, e);
    }
  }
  parseAndPlot("composite-chart", "composite-data");
  parseAndPlot("metrics-chart", "metrics-data");
})();
</script>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Art Style Search · {_h(data.run_name)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="{_FONTS_CDN}">
  <script src="{_PLOTLY_CDN}"></script>
  <style>{_CSS}</style>
</head>
<body>
  <main class="page">
    {header}
    {trajectories}
    {iterations_section}
    {kb_section}
    {protocol_section}
    {promotion_section}
    {holdout_section}
  </main>
  {plot_script}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_report(run_dir: Path, *, open_browser: bool = False) -> Path:
    """Generate ``runs/<run_dir>/report.html`` and return its path."""
    data = load_report_data(run_dir)
    report_path = run_dir / "report.html"
    html_doc = _assemble_html(data, report_path.parent)
    report_path.write_text(html_doc, encoding="utf-8")
    logger.info("Report written to %s", report_path)
    if open_browser:
        webbrowser.open(report_path.resolve().as_uri())
    return report_path


def build_all_reports(runs_dir: Path) -> list[Path]:
    """Regenerate reports for every run under *runs_dir* that has a state.json."""
    if not runs_dir.is_dir():
        return []
    results: list[Path] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "state.json").is_file():
            logger.info("Skipping %s: no state.json", run_dir.name)
            continue
        try:
            results.append(build_report(run_dir))
        except Exception:
            logger.exception("Failed to build report for %s", run_dir.name)
    return results
