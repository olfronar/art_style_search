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
    # Seed every numeric/bool/str field with a distinctive non-default value so any future
    # codec drop surfaces as a round-trip mismatch. (The original v9 MegaStyle codec bug
    # slipped past round-trip tests because the factory relied on dataclass defaults on
    # both sides of the encode/decode.)
    return MetricScores(
        dreamsim_similarity=0.72 + seed * 0.01,
        hps_score=0.26 + seed * 0.002,
        aesthetics_score=6.1 + seed * 0.1,
        color_histogram=0.64 + seed * 0.01,
        ssim=0.58 + seed * 0.01,
        # Ternary verdicts deliberately alternate MATCH (1.0) / MISS (0.0) across the five
        # dimensions rather than defaulting any to 0.5 — a default-value match would mask
        # a codec round-trip drop (see test_architecture_invariants.TestCodecReflection).
        vision_style=1.0,
        vision_subject=0.0,
        vision_composition=1.0,
        vision_medium=0.0,
        vision_proportions=1.0,
        megastyle_similarity=0.68 + seed * 0.01,
        style_gap="rim light reads hotter than the reference; warm it toward magenta",
        is_fallback=False,
    )


def make_aggregated_metrics(*, seed: float = 0.0) -> AggregatedMetrics:
    # Same seeding discipline as `make_metric_scores` — every field distinctive so codec
    # drops surface via round-trip tests.
    return AggregatedMetrics(
        dreamsim_similarity_mean=0.71 + seed * 0.01,
        dreamsim_similarity_std=0.03,
        hps_score_mean=0.25 + seed * 0.002,
        hps_score_std=0.01,
        aesthetics_score_mean=5.9 + seed * 0.1,
        aesthetics_score_std=0.4,
        color_histogram_mean=0.63 + seed * 0.01,
        color_histogram_std=0.05,
        ssim_mean=0.57 + seed * 0.01,
        ssim_std=0.04,
        style_consistency=0.82,
        completion_rate=0.95,
        vision_style=0.75,
        vision_style_std=0.15,
        vision_subject=0.55,
        vision_subject_std=0.2,
        vision_composition=0.85,
        vision_composition_std=0.1,
        vision_medium=0.65,
        vision_medium_std=0.12,
        vision_proportions=0.8,
        vision_proportions_std=0.09,
        megastyle_similarity_mean=0.67 + seed * 0.01,
        megastyle_similarity_std=0.02,
        compliance_topic_coverage=0.88,
        compliance_marker_coverage=0.92,
        section_ordering_rate=0.97,
        section_balance_rate=0.9,
        subject_specificity_rate=0.86,
        style_canon_fidelity=0.83,
        observation_boilerplate_purity=0.94,
        requested_ref_count=20,
        actual_ref_count=19,
        style_gap_notes=("rim light too saturated", "shading steps skip the ambient occlusion pass"),
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
