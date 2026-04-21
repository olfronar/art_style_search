"""Shared test factory helpers — plain functions (not fixtures) for building realistic test objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from art_style_search.retry import caption_circuit_breaker, generation_circuit_breaker, vision_circuit_breaker
from art_style_search.types import (
    AggregatedMetrics,
    Caption,
    ConvergenceReason,
    IterationResult,
    KnowledgeBase,
    LoopState,
    MetricScores,
    PromptSection,
    PromptTemplate,
    StyleProfile,
)

# ---------------------------------------------------------------------------
# Factory helpers — build realistic test fixtures
# ---------------------------------------------------------------------------


def make_caption(*, index: int = 0) -> Caption:
    return Caption(
        image_path=Path(f"/data/reference_images/painting_{index:03d}.png"),
        text=f"A moody watercolor landscape with muted earth tones, soft edges, and a low horizon line (image {index}).",
    )


def make_metric_scores(*, seed: float = 0.0) -> MetricScores:
    return MetricScores(
        dreamsim_similarity=0.72 + seed * 0.01,
        hps_score=0.26 + seed * 0.002,
        aesthetics_score=6.1 + seed * 0.1,
        megastyle_similarity=0.68 + seed * 0.01,
    )


def make_aggregated_metrics(*, seed: float = 0.0) -> AggregatedMetrics:
    return AggregatedMetrics(
        dreamsim_similarity_mean=0.71 + seed * 0.01,
        dreamsim_similarity_std=0.03,
        hps_score_mean=0.25 + seed * 0.002,
        hps_score_std=0.01,
        aesthetics_score_mean=5.9 + seed * 0.1,
        aesthetics_score_std=0.4,
        megastyle_similarity_mean=0.67 + seed * 0.01,
        megastyle_similarity_std=0.02,
    )


def make_prompt_section(*, index: int = 0) -> PromptSection:
    sections = [
        ("medium", "Overall artistic medium", "Watercolor painting on rough cold-pressed paper"),
        ("palette", "Color palette", "Muted earth tones: ochre, burnt sienna, Payne's grey, sap green"),
        ("composition", "Layout and framing", "Low horizon line, asymmetric balance, negative space in upper third"),
    ]
    name, desc, val = sections[index % len(sections)]
    return PromptSection(name=name, description=desc, value=val)


def make_prompt_template(*, n_sections: int = 3) -> PromptTemplate:
    return PromptTemplate(
        sections=[make_prompt_section(index=i) for i in range(n_sections)],
        negative_prompt="photorealistic, 3D render, digital art, sharp edges",
    )


def make_style_profile() -> StyleProfile:
    return StyleProfile(
        color_palette="Muted earth tones — ochre, burnt sienna, slate blue, sap green.",
        composition="Low horizon, asymmetric balance, generous negative space.",
        technique="Wet-on-wet watercolor with dry-brush texture accents.",
        mood_atmosphere="Contemplative, quiet, slightly melancholic.",
        subject_matter="Rural landscapes, fields, isolated structures.",
        influences="Andrew Wyeth, J.M.W. Turner, Japanese ink wash.",
        gemini_raw_analysis="Gemini vision analysis text for reference images...",
        claude_raw_analysis="Claude structured analysis text for reference images...",
    )


def make_iteration_result(*, branch_id: int = 0, iteration: int = 1) -> IterationResult:
    return IterationResult(
        branch_id=branch_id,
        iteration=iteration,
        template=make_prompt_template(),
        rendered_prompt=make_prompt_template().render(),
        image_paths=[Path(f"/data/outputs/iter_{iteration:03d}/img_{i}.png") for i in range(4)],
        per_image_scores=[make_metric_scores(seed=float(i)) for i in range(4)],
        aggregated=make_aggregated_metrics(seed=float(iteration)),
        claude_analysis="Good progress on tonal range; edges still too crisp compared to references.",
        template_changes="Increased wet-on-wet emphasis, added dry-brush texture note.",
        kept=True,
        n_images_attempted=4,
        n_images_succeeded=4,
    )


def make_loop_state(
    *,
    iteration: int = 5,
    converged: bool = False,
    convergence_reason: ConvergenceReason | None = None,
    global_best_metrics: AggregatedMetrics | None = None,
) -> LoopState:
    return LoopState(
        iteration=iteration,
        current_template=make_prompt_template(),
        best_template=make_prompt_template(n_sections=2),
        best_metrics=make_aggregated_metrics(seed=1.0),
        knowledge_base=KnowledgeBase(),
        captions=[make_caption(index=i) for i in range(3)],
        style_profile=make_style_profile(),
        experiment_history=[make_iteration_result(branch_id=i, iteration=i + 1) for i in range(2)],
        global_best_prompt="Watercolor painting on rough cold-pressed paper, muted earth tones...",
        global_best_metrics=global_best_metrics,
        converged=converged,
        convergence_reason=convergence_reason,
    )


# ---------------------------------------------------------------------------
# Auto-use fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """Reset all per-surface circuit breakers between tests to prevent state leakage."""
    yield
    for cb in (caption_circuit_breaker, generation_circuit_breaker, vision_circuit_breaker):
        cb._consecutive_failures = 0
        cb._open_until = 0.0
