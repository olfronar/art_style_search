"""Knowledge Base maintenance — hypothesis tracking, open problems, caption diffs."""

from __future__ import annotations

import logging
import re

from art_style_search.experiment import ExperimentProposal, best_kept_result
from art_style_search.types import (
    AggregatedMetrics,
    Caption,
    IterationResult,
    KnowledgeBase,
    OpenProblem,
    PromptTemplate,
    classify_hypothesis,
    get_category_names,
)

logger = logging.getLogger(__name__)

_PRIORITY_ORDER = {"HIGH": 0, "MED": 1, "LOW": 2}


def update_knowledge_base(
    kb: KnowledgeBase,
    result: IterationResult,
    template: PromptTemplate,
    best_metrics: AggregatedMetrics | None,
    proposal: ExperimentProposal,
    iteration: int,
) -> None:
    """Update the shared KB with one experiment's results."""
    parent_id: str | None = None
    if proposal.builds_on:
        parent_match = re.match(r"H(\d+)", proposal.builds_on)
        if parent_match:
            parent_id = f"H{parent_match.group(1)}"

    category_names = get_category_names(template)
    category = classify_hypothesis(result.hypothesis, category_names) if result.hypothesis else "general"

    metric_delta: dict[str, float] = {}
    if best_metrics is not None:
        metric_delta = {
            "dino": result.aggregated.dino_similarity_mean - best_metrics.dino_similarity_mean,
            "lpips": result.aggregated.lpips_distance_mean - best_metrics.lpips_distance_mean,
            "hps": result.aggregated.hps_score_mean - best_metrics.hps_score_mean,
            "aesthetics": result.aggregated.aesthetics_score_mean - best_metrics.aesthetics_score_mean,
            "color_histogram": result.aggregated.color_histogram_mean - best_metrics.color_histogram_mean,
            "texture": result.aggregated.texture_mean - best_metrics.texture_mean,
            "ssim": result.aggregated.ssim_mean - best_metrics.ssim_mean,
            "vision_style": result.aggregated.vision_style - best_metrics.vision_style,
            "vision_subject": result.aggregated.vision_subject - best_metrics.vision_subject,
            "vision_composition": result.aggregated.vision_composition - best_metrics.vision_composition,
        }

    lessons = proposal.lessons
    lesson_text = lessons.confirmed or lessons.new_insight or lessons.rejected or ""

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
        )

    if proposal.open_problems:
        scores = result.per_image_scores
        best_cat_dino = sum(sc.dino_similarity for sc in scores) / len(scores) if scores else 0.0

        prev_problem_texts = {p.text: p.since_iteration for p in kb.open_problems}

        new_problems: list[OpenProblem] = []
        for prob_text in proposal.open_problems:
            prob_cat = classify_hypothesis(prob_text, category_names)
            cat_progress = kb.categories.get(prob_cat)

            if cat_progress is None or not cat_progress.confirmed_insights:
                priority = "HIGH"
            elif cat_progress.rejected_approaches and len(cat_progress.rejected_approaches) >= len(
                cat_progress.confirmed_insights
            ):
                priority = "MED"
            else:
                priority = "LOW"

            # Gap = distance from perfect DINO (1.0) for this experiment
            gap = 1.0 - best_cat_dino
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
            if score < 5.0:
                assessment = f"Vision {dim_name} score: {score:.0f}/10"
                prob_text = f"{dim_name.title()} fidelity: {assessment}"
                if not any(dim_name in p.text.lower() for p in new_problems):
                    new_problems.append(
                        OpenProblem(
                            text=prob_text,
                            category=cat_name,
                            priority="HIGH" if score < 3.0 else "MED",
                            metric_gap=float((5.0 - score) / 10.0),
                            since_iteration=iteration,
                        )
                    )

        # Merge with existing problems instead of replacing — keeps context across experiments
        existing_by_text = {p.text: p for p in kb.open_problems}
        for p in new_problems:
            existing_by_text[p.text] = p  # newer version wins for duplicates
        kb.open_problems = sorted(existing_by_text.values(), key=lambda p: _PRIORITY_ORDER.get(p.priority, 3))[:10]


def build_caption_diffs(last_results: list[IterationResult], worst_captions: list[Caption]) -> str:
    """Show how captions changed for worst-performing images vs previous iteration."""
    if not last_results or not worst_captions:
        return ""
    prev = best_kept_result(last_results)
    if not prev:
        return ""
    prev_by_path = {c.image_path: c.text for c in prev.iteration_captions}

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
            diffs.append(f"**{cap.image_path.name}**:\n  PREV: {prev_text[:200]}...\n  NOW:  {cap.text[:200]}...")
    if not diffs:
        return ""
    return "## Caption Changes (worst 3 images, prev → current)\n" + "\n".join(diffs)
