"""HTML helper and section renderers for reports."""

from __future__ import annotations

import html
import logging
from pathlib import Path

from art_style_search.report_data import ReportData, _rel
from art_style_search.scoring import adaptive_composite_score, composite_score
from art_style_search.types import (
    CategoryProgress,
    Hypothesis,
    IterationResult,
    KnowledgeBase,
    MetricScores,
    OpenProblem,
)

logger = logging.getLogger(__name__)

_MAX_TREE_DEPTH = 6
_PRIORITY_PREFIXES = ("[HIGH] ", "[MED] ", "[LOW] ", "[HIGH]", "[MED]", "[LOW]")


def _strip_priority_prefix(text: str) -> str:
    """Strip leading [HIGH]/[MED]/[LOW] prefix from problem text."""
    for prefix in _PRIORITY_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


_VERDICT_TAG_RE = __import__("re").compile(
    r"<(style|subject|composition)\s+verdict=\"(MATCH|PARTIAL|MISS)\">(.*?)</\1>",
    __import__("re").DOTALL,
)
_IMAGE_HEADER_RE = __import__("re").compile(r"^\*\*(.+?)\*\*\s*\[([^\]]+)\]:\s*", __import__("re").MULTILINE)

_VERDICT_CSS = {"MATCH": "verdict-match", "PARTIAL": "verdict-partial", "MISS": "verdict-miss"}
_VERDICT_LABEL = {"MATCH": "Match", "PARTIAL": "Partial", "MISS": "Miss"}
_DIM_LABEL = {"style": "Style", "subject": "Subject", "composition": "Composition"}


def _format_vision_feedback(raw: str) -> str:
    """Convert raw vision feedback with XML-like verdict tags into styled HTML."""
    # Split into per-image blocks by the **filename** [codes]: pattern
    parts = _IMAGE_HEADER_RE.split(raw)
    if len(parts) < 4:
        # No recognisable structure — fall back to escaped pre
        return f"<pre>{html.escape(raw)}</pre>"

    blocks: list[str] = []
    # parts[0] is text before first match (usually empty), then groups of 3
    preamble = parts[0].strip()
    if preamble:
        blocks.append(f"<p class='vision-preamble'>{html.escape(preamble)}</p>")

    for i in range(1, len(parts), 3):
        if i + 2 >= len(parts):
            break
        filename = parts[i]
        codes = parts[i + 1]
        body = parts[i + 2]

        # Parse verdict tags within this image's body
        verdicts: list[str] = []
        for match in _VERDICT_TAG_RE.finditer(body):
            dim = match.group(1) or ""
            verdict = match.group(2) or ""
            text = (match.group(3) or "").strip()
            css = _VERDICT_CSS.get(verdict, "")
            dim_label = _DIM_LABEL.get(dim) or dim.title()
            v_label = _VERDICT_LABEL.get(verdict) or verdict
            verdicts.append(
                f"<div class='vision-verdict {css}'>"
                f"<span class='vision-dim'>{html.escape(dim_label)}</span>"
                f"<span class='vision-badge'>{html.escape(v_label)}</span>"
                f"<span class='vision-text'>{html.escape(text)}</span>"
                f"</div>"
            )

        if not verdicts:
            # No tags parsed — show body as plain text
            verdicts.append(f"<p>{html.escape(body.strip())}</p>")

        # Truncate long filenames for display
        short_name = filename[:32] + ("…" if len(filename) > 32 else "")
        blocks.append(
            f"<div class='vision-image'>"
            f"<div class='vision-image-header'>"
            f"<span class='vision-filename'>{html.escape(short_name)}</span>"
            f"<span class='vision-codes'>{html.escape(codes)}</span>"
            f"</div>"
            f"{''.join(verdicts)}"
            f"</div>"
        )

    return f"<div class='vision-feedback'>{''.join(blocks)}</div>"


def _render_prompt_diff(old_prompt: str, new_prompt: str) -> str:
    """Render a unified diff between two prompts as styled HTML."""
    import difflib

    old_lines = old_prompt.splitlines(keepends=True)
    new_lines = new_prompt.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile="previous", tofile="current", n=2)
    diff_lines: list[str] = []
    for line in diff:
        escaped = html.escape(line.rstrip("\n"))
        if line.startswith("+") and not line.startswith("+++"):
            diff_lines.append(f"<span class='diff-add'>{escaped}</span>")
        elif line.startswith("-") and not line.startswith("---"):
            diff_lines.append(f"<span class='diff-del'>{escaped}</span>")
        elif line.startswith("@@"):
            diff_lines.append(f"<span class='diff-hunk'>{escaped}</span>")
        else:
            diff_lines.append(escaped)
    if not diff_lines:
        return "<p class='empty'>No changes.</p>"
    nl = "\n"
    return f"<pre class='diff-block'>{nl.join(diff_lines)}{nl}</pre>"


def _render_captions(winner: IterationResult) -> str:
    """Render the winner's captions as a collapsible list."""
    if not winner.iteration_captions:
        return ""
    items: list[str] = []
    for caption in winner.iteration_captions:
        name = caption.image_path.stem[:32] + ("…" if len(caption.image_path.stem) > 32 else "")
        items.append(
            f"<details class='caption-item'>"
            f"<summary><span class='caption-name'>{html.escape(name)}</span></summary>"
            f"<pre class='caption-text'>{html.escape(caption.text)}</pre>"
            f"</details>"
        )
    return f"<div class='captions-list'>{''.join(items)}</div>"


def _h(text: str | None) -> str:
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
    try:
        idx = int(gen_path.stem)
    except ValueError:
        return None
    if 0 <= idx < len(result.per_image_scores):
        return result.per_image_scores[idx]
    return None


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


def _render_summary_section(data: ReportData) -> str:
    """Render a concise run summary synthesizing KB learnings and score trajectory."""
    kb = data.state.knowledge_base
    iterations = data.iteration_numbers()
    if not iterations:
        return ""

    # Score trajectory
    first_winner = data.winner_of(iterations[0])
    first_score = composite_score(first_winner.aggregated) if first_winner else 0.0
    best_score = composite_score(data.state.global_best_metrics) if data.state.global_best_metrics else 0.0
    delta = best_score - first_score

    # Hypothesis stats
    n_total = len(kb.hypotheses)
    n_confirmed = sum(1 for h in kb.hypotheses if h.outcome == "confirmed")
    n_rejected = sum(1 for h in kb.hypotheses if h.outcome == "rejected")
    n_partial = n_total - n_confirmed - n_rejected

    # Confirmed insights
    confirmed = [h for h in kb.hypotheses if h.outcome == "confirmed" and h.lesson]
    insight_items = "".join(
        f"<li><strong>{_h(h.category.replace('_', ' '))}:</strong> {_h(h.lesson)}</li>" for h in confirmed[:5]
    )
    insights_html = f"<ul class='summary-insights'>{insight_items}</ul>" if insight_items else ""

    # Top open problems
    top_problems = kb.open_problems[:3]
    problem_items = "".join(f"<li>{_h(_strip_priority_prefix(p.text))}</li>" for p in top_problems)
    problems_html = f"<ul class='summary-problems'>{problem_items}</ul>" if problem_items else ""

    # Promotion stats
    decisions = data.promotion_decisions
    n_promoted = sum(1 for d in decisions if d.decision == "promoted")

    return f"""
<section class="summary-section">
  <div class="section-head">
    <span class="section-numeral">&Sigma;</span>
    <h2>Run Summary</h2>
    <p class="section-kicker">What this run learned, measured, and left unsolved.</p>
  </div>
  <div class="summary-grid">
    <div class="summary-card">
      <h3>Score trajectory</h3>
      <p>Started at <code>{_fmt_score(first_score)}</code>, reached
      <code class="summary-highlight">{_fmt_score(best_score)}</code>
      ({delta:+.3f}) over {len(iterations)} iterations.
      {n_promoted} of {len(decisions)} candidates promoted.</p>
    </div>
    <div class="summary-card">
      <h3>Hypothesis outcomes</h3>
      <p><strong>{n_confirmed}</strong> confirmed, <strong>{n_partial}</strong> partial,
      <strong>{n_rejected}</strong> rejected out of {n_total} tested
      ({n_confirmed * 100 // max(n_total, 1)}% confirmation rate).</p>
      {insights_html}
    </div>
    <div class="summary-card">
      <h3>Open problems</h3>
      <p>{len(kb.open_problems)} unresolved {"problem" if len(kb.open_problems) == 1 else "problems"} remain.</p>
      {problems_html}
    </div>
  </div>
</section>
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

    batch = [result.aggregated for result in results]
    # Highlight the kept experiment; fall back to highest composite if none was kept
    kept_ids = {r.branch_id for r in results if r.kept}
    if not kept_ids:
        kept_ids = {max(results, key=lambda r: composite_score(r.aggregated)).branch_id}
    rows: list[str] = []
    for result in results:
        score = composite_score(result.aggregated)
        adaptive = adaptive_composite_score(result.aggregated, batch) if len(batch) >= 2 else None
        adaptive_cell = _fmt_score(adaptive) if adaptive is not None else "—"
        is_highlighted = result.branch_id in kept_ids
        winner_mark = "<span class='winner-star'>✦</span>" if is_highlighted else ""
        kept_cell = "kept" if result.kept else "cut"
        kept_class = "kept-yes" if result.kept else "kept-no"
        row_class = " class='winner-row'" if is_highlighted else ""
        hyp = _h(result.hypothesis)
        truncated = hyp[:160] + ("…" if len(result.hypothesis) > 160 else "")
        rows.append(
            f"<tr{row_class}>"
            f"<td class='num-col'>{winner_mark}<span class='branch-id'>{result.branch_id:02d}</span></td>"
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

    caption_by_idx = {i: caption.image_path for i, caption in enumerate(winner.iteration_captions)}
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
    prev_winner: IterationResult | None = None
    blocks: list[str] = []
    for iteration in iterations:
        results = data.iteration_logs[iteration]
        winner = data.winner_of(iteration)
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
                    body = _format_vision_feedback(text) if label == "Vision feedback" else f"<pre>{_h(text)}</pre>"
                    narrative_blocks.append(
                        f"<details class='narrative'><summary>{_h(label)}</summary>{body}</details>"
                    )
            # Prompt diff vs previous iteration
            if prev_winner and winner.rendered_prompt and prev_winner.rendered_prompt:
                diff_html = _render_prompt_diff(prev_winner.rendered_prompt, winner.rendered_prompt)
                narrative_blocks.append(
                    f"<details class='narrative'><summary>Prompt diff vs previous</summary>{diff_html}</details>"
                )
            # Captions
            captions_html = _render_captions(winner)
            if captions_html:
                narrative_blocks.append(
                    f"<details class='narrative'><summary>Captions ({len(winner.iteration_captions)})</summary>"
                    f"{captions_html}</details>"
                )
        open_attr = " open" if iteration == latest else ""
        blocks.append(
            f"""
<details class="iteration"{open_attr}>
  <summary>
    <span class="iter-number">Iteration {iteration:02d}</span>
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
        if winner:
            prev_winner = winner
    legend = (
        '<details class="fold metric-legend">'
        "<summary><span class='fold-cue'>&sect;</span> Metric abbreviations</summary>"
        "<dl class='kv'>"
        "<div class='kv-row'><dt>DS</dt><dd>DreamSim perceptual similarity (human-aligned)</dd></div>"
        "<div class='kv-row'><dt>Color</dt><dd>HSV color histogram intersection</dd></div>"
        "<div class='kv-row'><dt>SSIM</dt><dd>Structural similarity index</dd></div>"
        "<div class='kv-row'><dt>HPS</dt><dd>Human Preference Score v2 (caption-image alignment)</dd></div>"
        "<div class='kv-row'><dt>Aes</dt><dd>LAION Aesthetics predictor (1-10 scale)</dd></div>"
        "<div class='kv-row'><dt>V[S]</dt><dd>Vision style fidelity (MATCH/PARTIAL/MISS)</dd></div>"
        "<div class='kv-row'><dt>V[Su]</dt><dd>Vision subject fidelity</dd></div>"
        "<div class='kv-row'><dt>V[Co]</dt><dd>Vision composition fidelity</dd></div>"
        "</dl></details>"
    )

    return f"""
<section class="iterations">
  <div class="section-head">
    <span class="section-numeral">II</span>
    <h2>Iterations</h2>
    <p class="section-kicker">Per-iteration experiments with their hypotheses, scores, and the winner's reference / generated pairs.</p>
  </div>
  {legend}
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


def _count_descendants(node_id: str, children_map: dict[str, list[Hypothesis]]) -> int:
    total = 0
    stack = list(children_map.get(node_id, []))
    while stack:
        child = stack.pop()
        total += 1
        stack.extend(children_map.get(child.id, []))
    return total


def _render_hypothesis_tree(kb: KnowledgeBase) -> str:
    if not kb.hypotheses:
        return "<p class='empty'>No hypotheses recorded yet.</p>"

    by_id: dict[str, Hypothesis] = {hypothesis.id: hypothesis for hypothesis in kb.hypotheses}
    children_map: dict[str, list[Hypothesis]] = {}
    roots: list[Hypothesis] = []
    for hypothesis in kb.hypotheses:
        if hypothesis.parent_id and hypothesis.parent_id in by_id:
            children_map.setdefault(hypothesis.parent_id, []).append(hypothesis)
        else:
            roots.append(hypothesis)

    def _render_meta(hypothesis: Hypothesis) -> str:
        return (
            f"<span class='hyp-id'>{_h(hypothesis.id)}</span>"
            f"<span class='hyp-sep'>·</span>"
            f"<span class='hyp-iter'>iter {hypothesis.iteration}</span>"
            f"<span class='hyp-sep'>·</span>"
            f"<span class='hyp-category'>{_h(hypothesis.category.replace('_', ' '))}</span>"
            f"<span class='hyp-sep'>·</span>"
            f"<span class='hyp-outcome'>{_h(hypothesis.outcome)}</span>"
        )

    def _render_node(hypothesis: Hypothesis, depth: int) -> str:
        children = children_map.get(hypothesis.id, [])
        css_class = f"hyp hyp-{_h(hypothesis.outcome)}"
        statement = _h(hypothesis.statement)
        lesson = f"<div class='hyp-lesson'>{_h(hypothesis.lesson)}</div>" if hypothesis.lesson else ""
        meta = _render_meta(hypothesis)

        if depth >= _MAX_TREE_DEPTH and children:
            descendant_count = _count_descendants(hypothesis.id, children_map)
            return (
                f"<div class='{css_class}'>"
                f"<div class='hyp-meta'>{meta}</div>"
                f"<div class='hyp-statement'>{statement}</div>"
                f"{lesson}"
                f"<div class='hyp-deeper'>(+{descendant_count} deeper — collapsed)</div>"
                "</div>"
            )
        inner = "".join(_render_node(child, depth + 1) for child in children)
        if children:
            return (
                f"<details class='{css_class}'>"
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

    return f'<div class="hypothesis-tree">{"".join(_render_node(root, 0) for root in roots)}</div>'


def _render_open_problems(problems: list[OpenProblem]) -> str:
    if not problems:
        return "<p class='empty'>No open problems.</p>"

    items: list[str] = []
    for idx, problem in enumerate(problems, start=1):
        gap = f"{problem.metric_gap:+.3f}" if problem.metric_gap is not None else "—"
        priority = problem.priority or "LOW"
        display_text = _strip_priority_prefix(problem.text)
        items.append(
            f"<li class='prio-{_h(priority.lower())}'>"
            f"<span class='prob-num'>{idx:02d}</span>"
            f"<span class='prio-chip'>{_h(priority)}</span>"
            f"<span class='prob-text'>{_h(display_text)}</span>"
            f"<span class='prob-meta'>"
            f"<span>{_h(problem.category.replace('_', ' '))}</span>"
            f"<span>iter {problem.since_iteration}</span>"
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


def _render_protocol_section(data: ReportData) -> str:
    manifest = data.manifest
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
    decisions = data.promotion_decisions
    if not decisions:
        return ""

    rows: list[str] = []
    for decision in decisions:
        css_class = {"promoted": "promo-yes", "exploration": "promo-explore", "rejected": "promo-no"}.get(
            decision.decision, ""
        )
        p_cell = f"{decision.p_value:.4f}" if decision.p_value is not None else "—"
        effect_cell = f"{decision.delta:+.5f}"
        rows.append(
            f'<tr class="{css_class}">'
            f"<td>{decision.iteration + 1}</td>"
            f"<td>{decision.candidate_branch_id}</td>"
            f"<td>{_fmt_score(decision.baseline_score)}</td>"
            f"<td>{_fmt_score(decision.candidate_score)}</td>"
            f"<td>{effect_cell}</td>"
            f"<td>{_fmt_score(decision.epsilon)}</td>"
            f"<td>{p_cell}</td>"
            f'<td class="decision-cell">{_h(decision.decision)}</td>'
            f"</tr>"
        )

    n_promoted = sum(1 for decision in decisions if decision.decision == "promoted")
    n_explored = sum(1 for decision in decisions if decision.decision == "exploration")
    n_rejected = len(decisions) - n_promoted - n_explored
    kicker = (
        f"{n_promoted} promoted, "
        f"{n_explored} {'exploration' if n_explored == 1 else 'explorations'}, "
        f"{n_rejected} rejected."
    )
    return f"""
<section class="promotion-section">
  <div class="section-head">
    <span class="section-numeral">V</span>
    <h2>Promotion Decisions</h2>
    <p class="section-kicker">{kicker}</p>
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
    summary = data.holdout_summary
    if summary is None:
        return (
            '<section class="holdout-section">'
            '<div class="section-head"><span class="section-numeral">VI</span>'
            "<h2>Silent-Image Holdout</h2></div>"
            '<p class="empty">No holdout data available for this run. '
            "Enable the rigorous protocol with an information barrier to generate holdout metrics.</p>"
            "</section>"
        )

    n_silent = summary.get("silent_image_count", 0)

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
