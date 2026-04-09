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
_MAX_TREE_DEPTH = 6
_LOG_PATTERN = re.compile(r"iter_(\d+)_branch_(\d+)\.json$")


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
            name="Best (per iter)",
            line=dict(color="#4ade80", width=3),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=mean,
            mode="lines+markers",
            name="Mean (per iter)",
            line=dict(color="#60a5fa", width=2, dash="dot"),
            marker=dict(size=6),
        )
    )
    if data.state.global_best_metrics is not None:
        fig.add_hline(
            y=composite_score(data.state.global_best_metrics),
            line_color="#fbbf24",
            line_dash="dash",
            annotation_text="Global best",
            annotation_position="top right",
        )
    fig.update_layout(
        template="plotly_dark",
        title="Composite Score",
        xaxis_title="Iteration",
        yaxis_title="composite_score",
        margin=dict(l=50, r=30, t=50, b=50),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
    fig = make_subplots(rows=3, cols=3, subplot_titles=titles, vertical_spacing=0.12, horizontal_spacing=0.08)

    for idx, (attr, _title) in enumerate(_METRIC_SPECS):
        row = idx // 3 + 1
        col = idx % 3 + 1
        iters, best, mean = _metric_series(data, lambda m, a=attr: getattr(m, a))
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=best,
                mode="lines+markers",
                name="Best",
                line=dict(color="#4ade80", width=2),
                marker=dict(size=5),
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
                name="Mean",
                line=dict(color="#60a5fa", width=1.5, dash="dot"),
                marker=dict(size=4),
                showlegend=(idx == 0),
                legendgroup="mean",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        template="plotly_dark",
        title="Per-metric trajectories",
        height=820,
        margin=dict(l=50, r=30, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Iteration", row=3)
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
        status = f"converged ({state.convergence_reason.value})"

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
    profile_html = "".join(f"<tr><th>{_h(label)}</th><td>{_h(text)}</td></tr>" for label, text in profile_rows)

    return f"""
<header class="run-header">
  <h1>{_h(data.run_name)}</h1>
  <div class="stat-row">
    <div class="stat"><span class="stat-label">Iterations</span><span class="stat-value">{iteration_count}</span></div>
    <div class="stat"><span class="stat-label">Status</span><span class="stat-value">{_h(status)}</span></div>
    <div class="stat"><span class="stat-label">Best composite</span><span class="stat-value">{best_score}</span></div>
    <div class="stat"><span class="stat-label">Fixed refs</span><span class="stat-value">{len(state.fixed_references)}</span></div>
  </div>
  <details class="collapsible">
    <summary>Style profile</summary>
    <table class="kv">{profile_html}</table>
  </details>
  <details class="collapsible">
    <summary>Best meta-prompt</summary>
    <pre class="prompt-block">{_h(state.global_best_prompt)}</pre>
  </details>
</header>
"""


def _render_trajectories_section(composite_json: str, multi_json: str) -> str:
    if not composite_json:
        return '<section><h2>Metric trajectories</h2><p class="empty">No iteration logs available yet.</p></section>'
    return f"""
<section class="trajectories">
  <h2>Metric trajectories</h2>
  <div id="composite-chart" class="chart"></div>
  <div id="metrics-chart" class="chart"></div>
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
        winner_mark = "★ " if r.branch_id == winner_id else ""
        kept_cell = "✓" if r.kept else "✗"
        kept_class = "kept-yes" if r.kept else "kept-no"
        row_class = " class='winner-row'" if r.branch_id == winner_id else ""
        hyp = _h(r.hypothesis)
        rows.append(
            f"<tr{row_class}>"
            f"<td class='num'>{winner_mark}{r.branch_id}</td>"
            f"<td class='hypothesis' title='{hyp}'>{hyp[:140]}{'…' if len(r.hypothesis) > 140 else ''}</td>"
            f"<td class='num'>{_fmt_score(score)}</td>"
            f"<td class='num'>{adaptive_cell}</td>"
            f"<td class='{kept_class}'>{kept_cell}</td>"
            "</tr>"
        )
    return f"""
<table class="experiment-table">
  <thead>
    <tr>
      <th>#</th>
      <th>Hypothesis</th>
      <th>Composite</th>
      <th>Adaptive</th>
      <th>Kept</th>
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
        cards.append(
            f"""
<figure class="pair">
  <img src="{_h(ref_rel)}" alt="reference {idx}">
  <img src="{_h(gen_rel)}" alt="generated {idx}" title="{_h(tooltip)}">
  <figcaption>{idx:02d} · {_h(ref.name)}</figcaption>
  <code class="score">{_h(tooltip)}</code>
</figure>"""
        )
    return f'<div class="ref-gen-grid">{"".join(cards)}</div>'


def _render_iteration_drilldown(data: ReportData, report_dir: Path) -> str:
    iterations = data.iteration_numbers()
    if not iterations:
        return '<section><h2>Iterations</h2><p class="empty">No iteration logs available yet.</p></section>'

    latest = iterations[-1]
    blocks: list[str] = []
    for i in iterations:
        results = data.iteration_logs[i]
        winner = data.winner_of(i)
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
  <summary>Iteration {i} · {len(results)} experiments</summary>
  {experiment_table}
  {"".join(narrative_blocks)}
  {grid}
</details>"""
        )
    return f"<section><h2>Iterations</h2>{''.join(blocks)}</section>"


def _render_category_progress(kb: KnowledgeBase) -> str:
    if not kb.categories:
        return "<p class='empty'>No category progress recorded yet.</p>"

    rows: list[str] = []
    for name in sorted(kb.categories):
        cat: CategoryProgress = kb.categories[name]
        n_total = len(cat.hypothesis_ids)
        n_confirmed = len(cat.confirmed_insights)
        n_rejected = len(cat.rejected_approaches)
        # Simple bar: proportion confirmed
        confirmed_pct = (n_confirmed / max(n_total, 1)) * 100
        rejected_pct = (n_rejected / max(n_total, 1)) * 100
        delta_str = f"Δ {cat.best_perceptual_delta:+.3f}" if cat.best_perceptual_delta is not None else "no Δ yet"
        rows.append(
            f"""
<div class="cat-row">
  <div class="cat-name">{_h(name)}</div>
  <div class="cat-bar">
    <div class="cat-bar-confirmed" style="width: {confirmed_pct:.0f}%"></div>
    <div class="cat-bar-rejected" style="width: {rejected_pct:.0f}%"></div>
  </div>
  <div class="cat-meta">{n_total} hyp · {n_confirmed} confirmed · {n_rejected} rejected · {_h(delta_str)}</div>
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

    def _render_node(h: Hypothesis, depth: int) -> str:
        children = children_map.get(h.id, [])
        css_class = f"hyp hyp-{_h(h.outcome)}"
        meta = f"{_h(h.id)} · iter {h.iteration} · {_h(h.category)} · {_h(h.outcome.upper())}"
        statement = _h(h.statement)
        lesson = f"<div class='hyp-lesson'>{_h(h.lesson)}</div>" if h.lesson else ""

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
                f"<summary>{meta}</summary>"
                f"<div class='hyp-statement'>{statement}</div>"
                f"{lesson}"
                f"{inner}"
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
    for p in problems:
        gap = f" (gap {p.metric_gap:+.3f})" if p.metric_gap is not None else ""
        priority = p.priority or "LOW"
        items.append(
            f"<li class='prio-{_h(priority.lower())}'>"
            f"<span class='prio-chip'>{_h(priority)}</span> "
            f"<span class='prob-text'>{_h(p.text)}</span>"
            f"<span class='prob-meta'>{_h(p.category)} · since iter {p.since_iteration}{gap}</span>"
            "</li>"
        )
    return f"<ul class='open-problems'>{''.join(items)}</ul>"


def _render_kb_section(data: ReportData) -> str:
    kb = data.state.knowledge_base
    return f"""
<section class="kb-section">
  <h2>Knowledge Base</h2>
  <h3>Category progress</h3>
  {_render_category_progress(kb)}
  <h3>Hypothesis tree</h3>
  {_render_hypothesis_tree(kb)}
  <h3>Open problems</h3>
  {_render_open_problems(kb.open_problems)}
</section>
"""


# ---------------------------------------------------------------------------
# HTML assembler
# ---------------------------------------------------------------------------


_CSS = """
:root {
  --bg: #0f172a;
  --panel: #1e293b;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --accent: #60a5fa;
  --good: #4ade80;
  --bad: #f87171;
  --warn: #fbbf24;
  --border: #334155;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }
h1 { font-size: 28px; margin: 0 0 12px; }
h2 { font-size: 22px; margin: 32px 0 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
h3 { font-size: 16px; margin: 24px 0 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
section { background: var(--panel); border-radius: 10px; padding: 20px; margin-bottom: 24px; border: 1px solid var(--border); }
.run-header { background: var(--panel); border-radius: 10px; padding: 24px; border: 1px solid var(--border); margin-bottom: 24px; }
.stat-row { display: flex; gap: 24px; margin: 16px 0; flex-wrap: wrap; }
.stat { display: flex; flex-direction: column; }
.stat-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-value { font-size: 22px; font-weight: 600; color: var(--text); }
.collapsible { margin-top: 12px; }
.collapsible summary { cursor: pointer; color: var(--accent); padding: 6px 0; }
.collapsible[open] summary { margin-bottom: 8px; }
table.kv { border-collapse: collapse; width: 100%; }
table.kv th { text-align: left; padding: 6px 12px 6px 0; color: var(--muted); font-weight: 500; width: 180px; vertical-align: top; }
table.kv td { padding: 6px 0; color: var(--text); vertical-align: top; }
.prompt-block { background: #0b1220; padding: 16px; border-radius: 6px; overflow-x: auto; font-size: 12px; line-height: 1.5; max-height: 500px; border: 1px solid var(--border); }
.chart { margin-bottom: 24px; }
.empty { color: var(--muted); font-style: italic; }
.experiment-table { width: 100%; border-collapse: collapse; margin-bottom: 16px; }
.experiment-table th, .experiment-table td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }
.experiment-table th { color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; }
.experiment-table td.num { font-family: monospace; text-align: right; width: 90px; }
.experiment-table td.hypothesis { max-width: 500px; }
.experiment-table tr.winner-row { background: rgba(74, 222, 128, 0.08); }
.kept-yes { color: var(--good); font-weight: 600; }
.kept-no { color: var(--muted); }
.iteration { background: #162032; border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
.iteration > summary { cursor: pointer; font-weight: 600; font-size: 15px; padding: 6px 0; }
.narrative { margin: 8px 0; }
.narrative summary { cursor: pointer; color: var(--accent); font-size: 13px; padding: 4px 0; }
.narrative pre { background: #0b1220; padding: 12px; border-radius: 4px; font-size: 12px; line-height: 1.5; overflow-x: auto; white-space: pre-wrap; }
.ref-gen-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 16px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; border: 1px solid var(--border); padding: 8px; border-radius: 6px; background: #0b1220; margin: 0; }
.pair img { width: 100%; height: auto; object-fit: cover; border-radius: 3px; }
.pair figcaption { grid-column: 1 / 3; font-size: 11px; color: var(--muted); padding-top: 4px; }
.pair .score { grid-column: 1 / 3; font-family: monospace; font-size: 11px; color: var(--muted); }
.category-bars { display: flex; flex-direction: column; gap: 8px; }
.cat-row { display: grid; grid-template-columns: 180px 1fr auto; gap: 12px; align-items: center; }
.cat-name { font-size: 13px; font-weight: 500; }
.cat-bar { height: 14px; background: #0b1220; border-radius: 7px; overflow: hidden; display: flex; border: 1px solid var(--border); }
.cat-bar-confirmed { background: var(--good); }
.cat-bar-rejected { background: var(--bad); }
.cat-meta { font-size: 11px; color: var(--muted); font-family: monospace; }
.hypothesis-tree { display: flex; flex-direction: column; gap: 6px; }
.hyp, details.hyp { border-left: 4px solid var(--border); padding: 8px 12px; background: #0b1220; border-radius: 0 4px 4px 0; }
.hyp-confirmed, details.hyp-confirmed { border-left-color: var(--good); }
.hyp-rejected, details.hyp-rejected { border-left-color: var(--bad); }
.hyp-partial, details.hyp-partial { border-left-color: var(--warn); }
.hyp-meta, details.hyp > summary { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 4px; cursor: pointer; }
.hyp-statement { font-size: 13px; color: var(--text); margin-bottom: 4px; }
.hyp-lesson { font-size: 11px; color: var(--muted); font-style: italic; }
.hyp-deeper { font-size: 11px; color: var(--muted); margin-top: 6px; }
details.hyp[open] { padding-bottom: 12px; }
details.hyp > summary + * { margin-left: 12px; margin-top: 6px; }
.open-problems { list-style: none; padding: 0; margin: 0; }
.open-problems li { padding: 10px 12px; background: #0b1220; border: 1px solid var(--border); border-radius: 4px; margin-bottom: 6px; display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: baseline; }
.prio-chip { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: 600; letter-spacing: 0.05em; }
.prio-high .prio-chip { background: rgba(248, 113, 113, 0.2); color: var(--bad); border: 1px solid var(--bad); }
.prio-med .prio-chip { background: rgba(251, 191, 36, 0.2); color: var(--warn); border: 1px solid var(--warn); }
.prio-low .prio-chip { background: rgba(148, 163, 184, 0.2); color: var(--muted); border: 1px solid var(--muted); }
.prob-text { font-size: 13px; }
.prob-meta { font-size: 11px; color: var(--muted); font-family: monospace; }
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
  <script src="{_PLOTLY_CDN}"></script>
  <style>{_CSS}</style>
</head>
<body>
  {header}
  {trajectories}
  {iterations_section}
  {kb_section}
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
