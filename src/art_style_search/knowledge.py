"""Knowledge Base maintenance — hypothesis tracking, open problems, caption diffs."""

from __future__ import annotations

import logging
import re
from typing import Literal

from art_style_search.caption_sections import parse_labeled_sections
from art_style_search.contracts import ExperimentProposal
from art_style_search.prompt._format import _truncate_words
from art_style_search.scoring import classify_hypothesis
from art_style_search.types import (
    AggregatedMetrics,
    Caption,
    IterationResult,
    KnowledgeBase,
    OpenProblem,
    PromptTemplate,
    get_category_names,
)

logger = logging.getLogger(__name__)

_PRIORITY_ORDER = {"HIGH": 0, "MED": 1, "LOW": 2}
_PRIORITY_PREFIX_RE = re.compile(r"^\[(HIGH|MED|LOW)\]\s*", re.IGNORECASE)
IterationDecision = Literal["promoted", "exploration", "rejected"]


def strip_priority_prefix(text: str) -> str:
    """Strip a leading [HIGH]/[MED]/[LOW] tag (case-insensitive) from problem text."""
    return _PRIORITY_PREFIX_RE.sub("", text)


_NEAR_DUP_THRESHOLD = 0.5
_CAPTION_DIFF_WORD_BUDGET = 1500
_CAPTION_DIFF_SECTION_PRIORITY = ("Subject", "Art Style", "Composition")


def _section_diff_summary(prev_text: str, current_text: str) -> str:
    prev_sections = parse_labeled_sections(prev_text)
    current_sections = parse_labeled_sections(current_text)
    if not prev_sections or not current_sections:
        return ""

    ordered_names: list[str] = []
    seen: set[str] = set()
    for name in _CAPTION_DIFF_SECTION_PRIORITY:
        if prev_sections.get(name, "").strip() != current_sections.get(name, "").strip():
            ordered_names.append(name)
            seen.add(name)
    for name in list(prev_sections) + list(current_sections):
        if name in seen:
            continue
        if prev_sections.get(name, "").strip() != current_sections.get(name, "").strip():
            ordered_names.append(name)
            seen.add(name)
    if not ordered_names:
        return ""

    lines = ["  Section changes:"]
    for name in ordered_names:
        prev_body = prev_sections.get(name, "").strip()
        current_body = current_sections.get(name, "").strip()
        lines.append(f"  - [{name}]")
        lines.append(f"    PREV: {_truncate_words(prev_body, 60)}")
        lines.append(f"    NOW:  {_truncate_words(current_body, 60)}")
    return "\n".join(lines)


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens for Jaccard comparison."""
    normalized = _PRIORITY_PREFIX_RE.sub("", text).lower()
    return {w for w in re.findall(r"[a-z0-9]+", normalized) if len(w) > 2}


def _find_near_duplicate(text: str, candidates: list[str]) -> int | None:
    """Return index of the first near-duplicate in *candidates*, or None."""
    tokens = _tokenize(text)
    if not tokens:
        return None
    for idx, candidate in enumerate(candidates):
        cand_tokens = _tokenize(candidate)
        if not cand_tokens:
            continue
        intersection = len(tokens & cand_tokens)
        union = len(tokens | cand_tokens)
        if intersection / union >= _NEAR_DUP_THRESHOLD:
            return idx
    return None


_STYLE_GAP_NOTE_MAX = 8
_STYLE_GAP_MIN_TOKENS = 4
_STYLE_GAP_KB_MAX = 20


def aggregate_style_gap_notes(raw_gaps: list[str]) -> tuple[str, ...]:
    """Deduplicate + cluster per-image style-gap observations into a bounded tuple.

    Short or empty strings are dropped; near-duplicates (Jaccard token overlap
    ≥ ``_NEAR_DUP_THRESHOLD``) are merged keeping the longer, more informative
    phrasing. Capped at ``_STYLE_GAP_NOTE_MAX`` notes to avoid diluting the
    reasoner's attention with redundant observations.
    """
    kept: list[str] = []
    for gap in raw_gaps:
        note = (gap or "").strip()
        if len(_tokenize(note)) < _STYLE_GAP_MIN_TOKENS:
            continue
        dup_idx = _find_near_duplicate(note, kept)
        if dup_idx is None:
            kept.append(note)
        elif len(note) > len(kept[dup_idx]):
            kept[dup_idx] = note
        if len(kept) >= _STYLE_GAP_NOTE_MAX:
            break
    return tuple(kept)


def append_kb_style_gap_observations(kb: KnowledgeBase, new_notes: tuple[str, ...] | list[str]) -> None:
    """Append this iteration's deduplicated style-gap observations to the KB ring buffer.

    Notes already present in the KB (near-duplicates) are skipped so the buffer
    reflects distinct gaps across iterations. The oldest notes are dropped once
    the buffer exceeds ``_STYLE_GAP_KB_MAX``.
    """
    for note in new_notes:
        candidate = (note or "").strip()
        if len(_tokenize(candidate)) < _STYLE_GAP_MIN_TOKENS:
            continue
        if _find_near_duplicate(candidate, kb.style_gap_observations) is not None:
            continue
        kb.style_gap_observations.append(candidate)
    overflow = len(kb.style_gap_observations) - _STYLE_GAP_KB_MAX
    if overflow > 0:
        del kb.style_gap_observations[:overflow]


def _merge_problem(existing: OpenProblem, incoming: OpenProblem) -> OpenProblem:
    """Merge two near-duplicate problems without losing the strongest history."""
    existing_rank = _PRIORITY_ORDER.get(existing.priority, 99)
    incoming_rank = _PRIORITY_ORDER.get(incoming.priority, 99)
    chosen = existing if existing_rank <= incoming_rank else incoming

    text = incoming.text if len(incoming.text) > len(existing.text) else existing.text

    metric_gap_candidates = [gap for gap in (existing.metric_gap, incoming.metric_gap) if gap is not None]
    metric_gap = max(metric_gap_candidates) if metric_gap_candidates else None

    return OpenProblem(
        text=text,
        category=existing.category if existing_rank <= incoming_rank else incoming.category,
        priority=chosen.priority,
        metric_gap=metric_gap,
        since_iteration=min(existing.since_iteration, incoming.since_iteration),
    )


def _resolve_category(
    text: str,
    category_names: list[str],
    *,
    explicit_category: str = "",
    fallback_category: str = "general",
) -> str:
    """Resolve a category with explicit tags taking precedence over keyword fallback."""
    if explicit_category:
        return explicit_category
    if not text:
        return fallback_category
    classified = classify_hypothesis(text, category_names)
    return classified if classified != "general" else fallback_category


def _decision_to_outcome(decision: IterationDecision) -> tuple[str, bool]:
    """Map the orchestration decision to a hypothesis outcome and progress update policy."""
    if decision == "promoted":
        return "confirmed", True
    if decision == "exploration":
        return "partial", False
    return "rejected", True


def _manage_open_problems(
    kb: KnowledgeBase,
    result: IterationResult,
    proposal: ExperimentProposal,
    category: str,
    category_names: list[str],
    iteration: int,
) -> None:
    """Process, auto-generate, merge, age, and cap open problems on the KB."""
    if proposal.open_problems:
        scores = result.per_image_scores
        best_cat_ds = sum(sc.dreamsim_similarity for sc in scores) / len(scores) if scores else 0.0

        prev_problem_texts = {p.text: p.since_iteration for p in kb.open_problems}

        new_problems: list[OpenProblem] = []
        for raw_prob_text in proposal.open_problems:
            # Strip LLM-emitted priority prefix (e.g. "[HIGH] ...") and use it
            prefix_match = _PRIORITY_PREFIX_RE.match(raw_prob_text)
            if prefix_match:
                llm_priority: str | None = prefix_match.group(1).upper()
                prob_text = raw_prob_text[prefix_match.end() :]
            else:
                llm_priority = None
                prob_text = raw_prob_text

            prob_cat = _resolve_category(
                prob_text,
                category_names,
                explicit_category=proposal.target_category,
                fallback_category=category,
            )
            cat_progress = kb.categories.get(prob_cat)

            if llm_priority and llm_priority in _PRIORITY_ORDER:
                priority = llm_priority
            elif cat_progress is None or not cat_progress.confirmed_insights:
                priority = "HIGH"
            elif cat_progress.rejected_approaches and len(cat_progress.rejected_approaches) >= len(
                cat_progress.confirmed_insights
            ):
                priority = "MED"
            else:
                priority = "LOW"

            gap = 1.0 - best_cat_ds
            since = prev_problem_texts.get(prob_text, iteration)

            new_problems.append(
                OpenProblem(text=prob_text, category=prob_cat, priority=priority, metric_gap=gap, since_iteration=since)
            )

        # Auto-add open problems from low Gemini vision dimension scores
        agg = result.aggregated
        vision_dims = [
            ("style", agg.vision_style, "technique"),
            ("subject", agg.vision_subject, "subject_matter"),
            ("composition", agg.vision_composition, "composition"),
        ]
        for dim_name, score, cat_name in vision_dims:
            if score < 0.5:
                label = "MISS" if score == 0.0 else "PARTIAL"
                prob_text = f"{dim_name.title()} fidelity: Vision {dim_name} verdict {label}"
                if not any(dim_name in p.text.lower() for p in new_problems):
                    new_problems.append(
                        OpenProblem(
                            text=prob_text,
                            category=cat_name,
                            priority="HIGH" if score == 0.0 else "MED",
                            metric_gap=float(1.0 - score),
                            since_iteration=iteration,
                        )
                    )

        # Merge with existing problems — newer version wins for exact or near-duplicates
        merged: list[OpenProblem] = list(kb.open_problems)
        for new_p in new_problems:
            dup_idx = _find_near_duplicate(new_p.text, [p.text for p in merged])
            if dup_idx is not None:
                merged[dup_idx] = _merge_problem(merged[dup_idx], new_p)
            else:
                merged.append(new_p)
        kb.open_problems = merged

    # Age stale problems — demote priority if they haven't been solved (always runs)
    for p in kb.open_problems:
        age = iteration - p.since_iteration
        if age > 10:
            p.priority = "LOW"
        elif age > 5 and p.priority == "HIGH":
            p.priority = "MED"

    kb.open_problems = sorted(kb.open_problems, key=lambda p: _PRIORITY_ORDER.get(p.priority, 3))[:10]


def update_knowledge_base(
    kb: KnowledgeBase,
    result: IterationResult,
    template: PromptTemplate,
    best_metrics: AggregatedMetrics | None,
    proposal: ExperimentProposal,
    iteration: int,
    decision: IterationDecision = "rejected",
) -> None:
    """Update the shared KB with one experiment's results."""
    parent_id: str | None = None
    if proposal.builds_on:
        parent_match = re.match(r"H(\d+)", proposal.builds_on)
        if parent_match:
            parent_id = f"H{parent_match.group(1)}"

    category_names = get_category_names(template)
    category = _resolve_category(
        result.hypothesis,
        category_names,
        explicit_category=proposal.target_category or result.target_category,
    )

    metric_delta: dict[str, float] = {}
    if best_metrics is not None:
        metric_delta = {
            "dreamsim": result.aggregated.dreamsim_similarity_mean - best_metrics.dreamsim_similarity_mean,
            "hps": result.aggregated.hps_score_mean - best_metrics.hps_score_mean,
            "aesthetics": result.aggregated.aesthetics_score_mean - best_metrics.aesthetics_score_mean,
            "color_histogram": result.aggregated.color_histogram_mean - best_metrics.color_histogram_mean,
            "ssim": result.aggregated.ssim_mean - best_metrics.ssim_mean,
            "vision_style": result.aggregated.vision_style - best_metrics.vision_style,
            "vision_subject": result.aggregated.vision_subject - best_metrics.vision_subject,
            "vision_composition": result.aggregated.vision_composition - best_metrics.vision_composition,
        }

    lessons = proposal.lessons
    lesson_text = lessons.confirmed or lessons.new_insight or lessons.rejected or ""
    outcome, update_progress = _decision_to_outcome(decision)

    if result.hypothesis:
        kb.add_hypothesis(
            iteration=iteration,
            parent_id=parent_id,
            statement=result.hypothesis,
            experiment=result.experiment,
            category=category,
            kept=result.kept,
            metric_delta=metric_delta,
            lesson=lesson_text,
            confirmed=lessons.confirmed,
            rejected=lessons.rejected,
            direction_id=proposal.direction_id,
            direction_summary=proposal.direction_summary,
            failure_mechanism=proposal.failure_mechanism,
            intervention_type=proposal.intervention_type,
            risk_level=proposal.risk_level,
            expected_primary_metric=proposal.expected_primary_metric,
            expected_tradeoff=proposal.expected_tradeoff,
            changed_sections=proposal.changed_sections or result.changed_sections,
            outcome=outcome,
            update_progress=update_progress,
        )

    _manage_open_problems(kb, result, proposal, category, category_names, iteration)


def build_caption_diffs(prev_captions: list[Caption], worst_captions: list[Caption]) -> str:
    """Show how captions changed for worst-performing images vs previous iteration."""
    if not prev_captions or not worst_captions:
        return ""
    prev_by_path = {c.image_path: c.text for c in prev_captions}

    diffs: list[str] = []
    for cap in worst_captions:
        prev_text = prev_by_path.get(cap.image_path)
        if prev_text is None:
            continue
        if prev_text == cap.text:
            diffs.append(
                f"**{cap.image_path.name}**: Caption UNCHANGED (meta-prompt change had no effect on this image)"
            )
        else:
            section_summary = _section_diff_summary(prev_text, cap.text)
            if section_summary:
                diffs.append(f"**{cap.image_path.name}**:\n{section_summary}")
            else:
                diffs.append(
                    f"**{cap.image_path.name}**:\n"
                    f"  PREV: {_truncate_words(prev_text, 60)}\n"
                    f"  NOW:  {_truncate_words(cap.text, 60)}"
                )
    if not diffs:
        return ""
    body = "## Caption Changes (worst 3 images, prev → current)\n" + "\n".join(diffs)
    return _truncate_words(body, _CAPTION_DIFF_WORD_BUDGET, suffix="\n[...truncated]")
