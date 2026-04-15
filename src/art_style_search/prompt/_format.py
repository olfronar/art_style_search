"""Formatting helpers — render dataclasses into text blocks for the reasoning model."""

from __future__ import annotations

from art_style_search.types import (
    AggregatedMetrics,
    KnowledgeBase,
    PromptTemplate,
    StyleProfile,
)


def _format_style_profile(profile: StyleProfile, compact: bool = False) -> str:
    """Render a StyleProfile into a text block for system prompts.

    When *compact* is True, omits the raw analyses to save tokens.
    """
    result = (
        "## Style Profile\n\n"
        f"**Color palette:** {profile.color_palette}\n"
        f"**Composition:** {profile.composition}\n"
        f"**Technique:** {profile.technique}\n"
        f"**Mood / atmosphere:** {profile.mood_atmosphere}\n"
        f"**Subject matter:** {profile.subject_matter}\n"
        f"**Influences:** {profile.influences}"
    )
    if not compact:
        result += (
            "\n\n### Gemini raw analysis\n"
            f"{profile.gemini_raw_analysis}\n\n"
            "### Reasoning-model raw analysis\n"
            f"{profile.claude_raw_analysis}"
        )
    return result


def _format_template(template: PromptTemplate) -> str:
    """Render a PromptTemplate into an XML block for the reasoning model to read."""
    parts: list[str] = ["<template>"]
    for section in template.sections:
        parts.append(f'  <section name="{section.name}" description="{section.description}">{section.value}</section>')
    if template.negative_prompt:
        parts.append(f"  <negative>{template.negative_prompt}</negative>")
    if template.caption_sections:
        parts.append(f"  <caption_sections>{', '.join(template.caption_sections)}</caption_sections>")
    if template.caption_length_target > 0:
        parts.append(f"  <caption_length>{template.caption_length_target}</caption_length>")
    parts.append("</template>")
    return "\n".join(parts)


def _format_metrics(metrics: AggregatedMetrics) -> str:
    """Render AggregatedMetrics as a readable summary."""
    d = metrics.summary_dict()
    lines = [f"- {k}: {v:.4f}" for k, v in d.items()]
    return "\n".join(lines)


def _truncate_words(text: str, max_words: int, *, suffix: str = "...") -> str:
    """Cap ``text`` to ``max_words`` whitespace-separated tokens.

    If the text is already short enough, it's returned unchanged; otherwise
    the first ``max_words`` tokens are joined with spaces and ``suffix`` is
    appended.  Used to keep per-image captions and feedback blocks from
    swamping the reasoning-model context.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + suffix


# ---------------------------------------------------------------------------
# KnowledgeBase rendering (moved from types.py)
# ---------------------------------------------------------------------------


def format_knowledge_base(kb: KnowledgeBase, max_words: int = 3000) -> str:
    """Produce a structured, budget-enforced prompt section for the reasoning model.

    Priority order (fill until budget exhausted):
    1. Open Problems (~60 words)
    2. Per-Category Status — count + latest insight only (not all insights)
    3. Hypothesis Chain — last 5 full, rest as one-liners
    Rejected hypotheses are already marked REJECTED in the tree — no standalone section.
    """
    from art_style_search.types import Hypothesis

    if not kb.hypotheses:
        return ""

    num_cats = len(kb.categories)
    header = f"## Knowledge Base ({len(kb.hypotheses)} hypotheses across {num_cats} categories)\n"

    # --- Section 1: Open Problems (always included) ---
    problems_text = ""
    if kb.open_problems:
        lines = ["### Open Problems"]
        for i, prob in enumerate(kb.open_problems, 1):
            gap_str = f" \u2014 gap: {prob.metric_gap:.3f}" if prob.metric_gap is not None else ""
            lines.append(f"{i}. [{prob.priority}] {prob.text}{gap_str}")
        problems_text = "\n".join(lines)

    # --- Section 2: Per-Category Status (compact: count + latest insight) ---
    cat_lines = ["### Per-Category Status"]
    for cat_name in sorted(kb.categories):
        cat = kb.categories[cat_name]
        delta_str = f", best Δ: {cat.best_perceptual_delta:+.3f}" if cat.best_perceptual_delta is not None else ""
        n_conf = len(cat.confirmed_insights)
        n_rej = len(cat.rejected_approaches)
        cat_lines.append(
            f"**{cat_name}** [{len(cat.hypothesis_ids)} hyp, {n_conf} confirmed, {n_rej} rejected{delta_str}]"
        )
        if cat.last_mechanism_tried:
            cat_lines.append(f"  Last mechanism: {cat.last_mechanism_tried[:120]}")
        if cat.last_confirmed_mechanism:
            cat_lines.append(f"  Last confirmed mechanism: {cat.last_confirmed_mechanism[:120]}")
        if cat.confirmed_insights:
            cat_lines.append(f"  Latest: {cat.confirmed_insights[-1][:120]}")
        if cat.rejected_approaches:
            cat_lines.append(f"  Last rejected: {cat.rejected_approaches[-1][:280]}")
    cat_text = "\n".join(cat_lines)

    # --- Section 3: Hypothesis Chain (last 5 full, rest collapsed) ---
    recent_ids = {h.id for h in kb.hypotheses[-5:]}

    def _render_hyp(h: Hypothesis, indent: int, full: bool) -> str:
        prefix = "  " * indent + ("\u2514\u2500 " if indent > 0 else "")
        builds = f", builds on {h.parent_id}" if h.parent_id else ""
        if full:
            stmt = h.statement[:160] + ("..." if len(h.statement) > 160 else "")
            return f'{prefix}{h.id} (iter {h.iteration}, {h.category}{builds}) \u2192 {h.outcome.upper()}: "{stmt}"'
        return f"{prefix}{h.id} (iter {h.iteration}, {h.category}) \u2192 {h.outcome.upper()}"

    roots: list[Hypothesis] = []
    children_map: dict[str, list[Hypothesis]] = {}
    for h in kb.hypotheses:
        if h.parent_id is None:
            roots.append(h)
        else:
            children_map.setdefault(h.parent_id, []).append(h)

    tree_lines: list[str] = ["### Hypothesis Chain"]

    def _walk(h: Hypothesis, indent: int) -> None:
        tree_lines.append(_render_hyp(h, indent, full=(h.id in recent_ids)))
        for child in children_map.get(h.id, []):
            _walk(child, indent + 1)

    for root in roots:
        _walk(root, 0)
    # Orphans
    known_ids = {h.id for h in kb.hypotheses}
    for h in kb.hypotheses:
        if h.parent_id and h.parent_id not in known_ids and h not in roots:
            tree_lines.append(_render_hyp(h, 0, full=(h.id in recent_ids)))

    tree_text = "\n".join(tree_lines)

    # --- Assemble with budget enforcement ---
    # Priority: header + problems + categories + tree
    result_parts = [header]
    budget_remaining = max_words

    if problems_text:
        pw = len(problems_text.split())
        result_parts.append(problems_text)
        budget_remaining -= pw

    cw = len(cat_text.split())
    if cw <= budget_remaining:
        result_parts.append("\n" + cat_text)
        budget_remaining -= cw

    tw = len(tree_text.split())
    if tw <= budget_remaining:
        result_parts.append("\n" + tree_text)
    elif budget_remaining > 100:
        # Truncate tree: only show last 5 hypotheses
        short_lines = ["### Hypothesis Chain (recent)"]
        for h in kb.hypotheses[-5:]:
            short_lines.append(_render_hyp(h, 0, full=True))
        result_parts.append("\n" + "\n".join(short_lines))

    return "\n".join(result_parts)


def suggest_target_categories(kb: KnowledgeBase, num_targets: int, categories: list[str]) -> list[str]:
    """Rank categories by improvement potential for diverse experiment targeting."""
    scored: list[tuple[str, float]] = []
    for cat in categories:
        progress = kb.categories.get(cat)
        if not progress:
            scored.append((cat, 1.0))  # unexplored = high priority
            continue
        n_confirmed = len(progress.confirmed_insights)
        n_rejected = len(progress.rejected_approaches)
        n_total = len(progress.hypothesis_ids)

        if n_rejected >= 3 and n_confirmed == 0:
            score = 0.3 if progress.last_mechanism_tried else 0.2
        elif n_confirmed > 0 and progress.best_perceptual_delta and progress.best_perceptual_delta > 0:
            score = 0.7  # partial success, room to build
        else:
            score = 0.5 / max(n_total, 1)
            if progress.last_mechanism_tried and not progress.last_confirmed_mechanism:
                score += 0.1
        scored.append((cat, score))

    scored.sort(key=lambda x: -x[1])
    return [cat for cat, _ in scored[:num_targets]]
