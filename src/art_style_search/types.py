"""Shared data structures for the art style search loop.

Scoring helpers (``composite_score``, ``improvement_epsilon``, etc.) and
hypothesis classification live in ``art_style_search.scoring``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@dataclass(frozen=True)
class Caption:
    """A cached caption for a single reference image."""

    image_path: Path
    text: str


@dataclass(frozen=True)
class MetricScores:
    """Evaluation scores for a single generated image against its paired reference."""

    dreamsim_similarity: float  # higher = better, perceptual similarity (DreamSim, human-aligned)
    hps_score: float  # higher = better, human preference for caption-image alignment
    aesthetics_score: float  # higher = better, 1-10 scale
    color_histogram: float = 0.0  # higher = better, HSV histogram similarity [0, 1]
    ssim: float = 0.0  # higher = better, structural similarity index [0, 1]
    vision_style: float = 0.5  # higher = better, ternary: MATCH=1.0, PARTIAL=0.5, MISS=0.0
    vision_subject: float = 0.5  # higher = better, ternary: MATCH=1.0, PARTIAL=0.5, MISS=0.0
    vision_composition: float = 0.5  # higher = better, ternary: MATCH=1.0, PARTIAL=0.5, MISS=0.0
    is_fallback: bool = False  # True for zero-score sentinels substituted on evaluation failure


# Ternary verdict map for Gemini vision scoring
VISION_VERDICT_MAP: dict[str, float] = {"MATCH": 1.0, "PARTIAL": 0.5, "MISS": 0.0}
VISION_VERDICT_DEFAULT = 0.5  # neutral when parsing fails
_VISION_SCORE_LABEL: dict[float, str] = {1.0: "M", 0.5: "P", 0.0: "X"}


def verdict_label(score: float) -> str:
    """Map a ternary vision score to its display letter (M/P/X)."""
    return _VISION_SCORE_LABEL.get(score, "X")


@dataclass(frozen=True)
class VisionDimensionScore:
    """One dimension of the Gemini vision comparison — ternary + qualitative."""

    dimension: str  # "style", "subject", "composition"
    score: float  # 0.0 (MISS), 0.5 (PARTIAL), 1.0 (MATCH)
    assessment: str  # qualitative text explaining the verdict


@dataclass(frozen=True)
class VisionScores:
    """Structured per-image scores from Gemini vision comparison."""

    style: VisionDimensionScore
    subject: VisionDimensionScore
    composition: VisionDimensionScore

    @classmethod
    def default(cls) -> VisionScores:
        """Neutral scores when parsing fails."""
        return cls(
            style=VisionDimensionScore("style", 0.5, ""),
            subject=VisionDimensionScore("subject", 0.5, ""),
            composition=VisionDimensionScore("composition", 0.5, ""),
        )


@dataclass(frozen=True)
class CaptionComplianceStats:
    """Structured caption-compliance rates used by scoring and reporting."""

    section_topic_coverage: float = 1.0
    section_marker_coverage: float = 1.0
    section_ordering_rate: float = 1.0
    section_balance_rate: float = 1.0
    subject_specificity_rate: float = 1.0

    @property
    def overall(self) -> float:
        return compliance_components_mean(
            self.section_topic_coverage,
            self.section_marker_coverage,
            self.section_ordering_rate,
            self.section_balance_rate,
            self.subject_specificity_rate,
        )


_COMPLIANCE_COMPONENT_COUNT = len(fields(CaptionComplianceStats))


def compliance_components_mean(*values: float) -> float:
    """Mean of the caption-compliance components — divisor tracks dataclass field count."""
    if len(values) != _COMPLIANCE_COMPONENT_COUNT:
        msg = f"Expected {_COMPLIANCE_COMPONENT_COUNT} compliance components, got {len(values)}"
        raise ValueError(msg)
    return sum(values) / _COMPLIANCE_COMPONENT_COUNT


@dataclass(frozen=True)
class AggregatedMetrics:
    """Mean + std of per-image MetricScores, plus experiment-level vision scores."""

    # Per-image metric aggregates
    dreamsim_similarity_mean: float
    dreamsim_similarity_std: float
    hps_score_mean: float
    hps_score_std: float
    aesthetics_score_mean: float
    aesthetics_score_std: float
    color_histogram_mean: float = 0.0
    color_histogram_std: float = 0.0
    ssim_mean: float = 0.0
    ssim_std: float = 0.0

    # Style consistency: Jaccard similarity of [Art Style] blocks across captions [0, 1]
    style_consistency: float = 0.0

    # Completion rate: fraction of attempted images that succeeded (1.0 = all)
    completion_rate: float = 1.0

    # Per-image Gemini vision scores (ternary: MATCH=1.0, PARTIAL=0.5, MISS=0.0)
    vision_style: float = 0.5
    vision_style_std: float = 0.0
    vision_subject: float = 0.5
    vision_subject_std: float = 0.0
    vision_composition: float = 0.5
    vision_composition_std: float = 0.0

    # Structured caption-compliance signals [0, 1]
    compliance_topic_coverage: float = 1.0
    compliance_marker_coverage: float = 1.0
    section_ordering_rate: float = 1.0
    section_balance_rate: float = 1.0
    subject_specificity_rate: float = 1.0

    # Requested-vs-actual accounting for scoring/reporting
    requested_ref_count: int = 0
    actual_ref_count: int = 0

    def summary_dict(self) -> dict[str, float]:
        """Flat dict for JSON serialization and reasoning model consumption."""
        return {
            "dreamsim_similarity_mean": self.dreamsim_similarity_mean,
            "dreamsim_similarity_std": self.dreamsim_similarity_std,
            "hps_score_mean": self.hps_score_mean,
            "hps_score_std": self.hps_score_std,
            "aesthetics_score_mean": self.aesthetics_score_mean,
            "aesthetics_score_std": self.aesthetics_score_std,
            "color_histogram_mean": self.color_histogram_mean,
            "color_histogram_std": self.color_histogram_std,
            "ssim_mean": self.ssim_mean,
            "ssim_std": self.ssim_std,
            "style_consistency": self.style_consistency,
            "completion_rate": self.completion_rate,
            "vision_style": self.vision_style,
            "vision_style_std": self.vision_style_std,
            "vision_subject": self.vision_subject,
            "vision_subject_std": self.vision_subject_std,
            "vision_composition": self.vision_composition,
            "vision_composition_std": self.vision_composition_std,
            "compliance_topic_coverage": self.compliance_topic_coverage,
            "compliance_marker_coverage": self.compliance_marker_coverage,
            "section_ordering_rate": self.section_ordering_rate,
            "section_balance_rate": self.section_balance_rate,
            "subject_specificity_rate": self.subject_specificity_rate,
            "requested_ref_count": float(self.requested_ref_count),
            "actual_ref_count": float(self.actual_ref_count),
        }


@dataclass(frozen=True)
class StyleProfile:
    """Structured art style analysis — foundation for all prompt work."""

    color_palette: str
    composition: str
    technique: str
    mood_atmosphere: str
    subject_matter: str
    influences: str
    gemini_raw_analysis: str
    claude_raw_analysis: str


@dataclass
class PromptSection:
    """One section of the prompt template."""

    name: str
    description: str
    value: str


@dataclass
class PromptTemplate:
    """The prompt's structure (sections) — evolves separately from content.

    The template defines WHAT sections exist and their purpose.
    Values fill the sections with specific style descriptions.
    The reasoning model can propose changes to either the template or the values.

    ``caption_sections`` lists the labeled output sections the captioner
    should produce (e.g. ``["Art Style", "Color Palette", …]``).  Their
    order is part of the optimisation surface.

    ``caption_length_target`` is the target word count for produced captions.
    """

    sections: list[PromptSection] = field(default_factory=list)
    negative_prompt: str | None = None
    caption_sections: list[str] = field(default_factory=list)
    caption_length_target: int = 0

    def render(self) -> str:
        """Combine all section values into the final captioning meta-prompt."""
        parts = [s.value for s in self.sections if s.value]
        if self.negative_prompt:
            parts.append(f"Do NOT include: {self.negative_prompt}")
        main_block = " ".join(parts)
        caption_parts: list[str] = []
        if self.caption_sections:
            section_list = ", ".join(f"[{s}]" for s in self.caption_sections)
            caption_parts.append(f"Format your response with these labeled sections in this order: {section_list}.")
        if self.caption_length_target > 0:
            caption_parts.append(f"Target length: approximately {self.caption_length_target} words.")
        if caption_parts:
            return main_block + "\n\n" + " ".join(caption_parts)
        return main_block


class ConvergenceReason(enum.Enum):
    MAX_ITERATIONS = "max_iterations"
    PLATEAU = "plateau"
    REASONING_STOP = "reasoning_stop"


# ---------------------------------------------------------------------------
# Knowledge Base — structured iteration-to-iteration learning
# ---------------------------------------------------------------------------


def get_category_names(template: PromptTemplate) -> list[str]:
    """Merge synonym-map keys, StyleProfile field names, and PromptSection names into
    the canonical category set. Synonym-map keys (lighting, texture, background,
    caption_structure, …) must be included so ``suggest_target_categories`` can rank
    them as unexplored in fresh runs.
    """
    from art_style_search.utils import CATEGORY_SYNONYMS  # lazy import to avoid cycle via scoring.py

    cats = set(CATEGORY_SYNONYMS.keys())
    cats.update({"color_palette", "composition", "technique", "mood_atmosphere", "subject_matter"})
    for section in template.sections:
        cats.add(section.name)
    return sorted(cats)


@dataclass
class Hypothesis:
    """A hypothesis tested during optimization, with lineage tracking."""

    id: str  # "H1", "H2", auto-incremented
    iteration: int
    parent_id: str | None  # "H3" or None for root hypotheses
    statement: str  # from <hypothesis> tag
    experiment: str  # from <experiment> tag
    category: str  # auto-classified from text
    outcome: str  # "confirmed" | "rejected" | "partial"
    metric_delta: dict[str, float]  # {"dreamsim": +0.02, "hps": +0.01, ...}
    kept: bool
    lesson: str  # confirmed/rejected/insight text


@dataclass
class OpenProblem:
    """A ranked open problem — proposed by the reasoning model, validated by code."""

    text: str  # reasoning model's description
    category: str  # auto-classified
    priority: str  # "HIGH" | "MED" | "LOW" — set by code from metrics
    metric_gap: float | None = None  # DreamSim gap vs best category
    since_iteration: int = 0  # when first identified


@dataclass
class CategoryProgress:
    """Accumulated knowledge about one style dimension."""

    category: str
    best_perceptual_delta: float | None = None
    confirmed_insights: list[str] = field(default_factory=list)
    rejected_approaches: list[str] = field(default_factory=list)
    hypothesis_ids: list[str] = field(default_factory=list)


@dataclass
class KnowledgeBase:
    """Structured knowledge replacing flat research_log."""

    hypotheses: list[Hypothesis] = field(default_factory=list)
    categories: dict[str, CategoryProgress] = field(default_factory=dict)
    open_problems: list[OpenProblem] = field(default_factory=list)
    next_id: int = 1

    def add_hypothesis(
        self,
        iteration: int,
        parent_id: str | None,
        statement: str,
        experiment: str,
        category: str,
        kept: bool,
        metric_delta: dict[str, float],
        lesson: str,
        confirmed: str,
        rejected: str,
        *,
        outcome: str | None = None,
        update_progress: bool = True,
    ) -> Hypothesis:
        """Create a hypothesis, append it, and update category progress."""
        hid = f"H{self.next_id}"
        self.next_id += 1

        # Backward-compatible default when callers do not provide an explicit outcome.
        if outcome is None:
            if kept and confirmed and not rejected:
                outcome = "confirmed"
            elif not kept and rejected:
                outcome = "rejected"
            elif kept and rejected:
                outcome = "partial"
            elif not kept:
                outcome = "rejected"
            else:
                # kept=True, confirmed=False, rejected=False
                outcome = "confirmed"

        hyp = Hypothesis(
            id=hid,
            iteration=iteration,
            parent_id=parent_id,
            statement=statement,
            experiment=experiment,
            category=category,
            outcome=outcome,
            metric_delta=metric_delta,
            kept=kept,
            lesson=lesson,
        )
        self.hypotheses.append(hyp)

        # Update category progress
        cat = self.categories.get(category)
        if cat is None:
            cat = CategoryProgress(category=category)
            self.categories[category] = cat
        cat.hypothesis_ids.append(hid)

        if not update_progress:
            return hyp

        max_insights = 5

        perceptual_delta = metric_delta.get("dreamsim", 0.0)
        if outcome == "confirmed" or outcome == "partial":
            if lesson and lesson not in cat.confirmed_insights:
                cat.confirmed_insights.append(lesson)
                # Cap: drop oldest when exceeding limit (newer insights subsume older)
                if len(cat.confirmed_insights) > max_insights:
                    cat.confirmed_insights = cat.confirmed_insights[-max_insights:]
            if cat.best_perceptual_delta is None or perceptual_delta > cat.best_perceptual_delta:
                cat.best_perceptual_delta = perceptual_delta
        if outcome == "rejected":
            short = statement[:120]
            if short not in cat.rejected_approaches:
                cat.rejected_approaches.append(short)
                if len(cat.rejected_approaches) > max_insights:
                    cat.rejected_approaches = cat.rejected_approaches[-max_insights:]

        return hyp


@dataclass
class ReviewResult:
    """Independent review of one iteration's experiment outcomes."""

    experiment_assessments: list[str]  # per-experiment: did it achieve its hypothesis?
    noise_vs_signal: str  # which metric movements are real vs noise?
    strategic_guidance: str  # what should next iteration focus on?
    recommended_categories: list[str]  # which categories to target next


@dataclass
class IterationResult:
    """Complete record of one experiment within an iteration.

    Previously called a "branch iteration" — now represents a single
    hypothesis experiment (no persistent branch identity).
    """

    branch_id: int  # kept for backward compat; now used as experiment_id
    iteration: int
    template: PromptTemplate
    rendered_prompt: str
    image_paths: list[Path]
    per_image_scores: list[MetricScores]
    aggregated: AggregatedMetrics
    claude_analysis: str
    template_changes: str
    kept: bool
    hypothesis: str = ""
    experiment: str = ""
    vision_feedback: str = ""
    roundtrip_feedback: str = ""
    iteration_captions: list[Caption] = field(default_factory=list)
    n_images_attempted: int = 0
    n_images_succeeded: int = 0
    changed_section: str = ""
    target_category: str = ""


@dataclass
class LoopState:
    """Top-level state that gets persisted to state.json.

    Uses a shared KnowledgeBase and per-iteration experiments.
    """

    iteration: int
    current_template: PromptTemplate
    best_template: PromptTemplate
    best_metrics: AggregatedMetrics | None
    knowledge_base: KnowledgeBase
    captions: list[Caption]
    style_profile: StyleProfile
    fixed_references: list[Path] = field(default_factory=list)
    experiment_history: list[IterationResult] = field(default_factory=list)
    last_iteration_results: list[IterationResult] = field(default_factory=list)
    prev_best_captions: list[Caption] = field(default_factory=list)
    plateau_counter: int = 0
    global_best_prompt: str = ""
    global_best_metrics: AggregatedMetrics | None = None
    review_feedback: str = ""
    pairwise_feedback: str = ""
    converged: bool = False
    convergence_reason: ConvergenceReason | None = None
    # Scientific rigor fields (Phase 1-3)
    seed: int = 0
    protocol: str = "classic"  # "classic" or "rigorous"
    feedback_refs: list[Path] = field(default_factory=list)  # shown to reasoning model
    silent_refs: list[Path] = field(default_factory=list)  # evaluated but hidden from optimizer


# ---------------------------------------------------------------------------
# Scientific rigor types (Phases 1-3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunManifest:
    """Write-once provenance record for a run."""

    protocol_version: str  # "classic" or "rigorous_v1"
    seed: int
    cli_args: dict[str, Any]
    model_names: dict[str, str]  # caption_model, generator_model, reasoning_model, comparison_model
    reasoning_provider: str
    git_sha: str | None
    python_version: str
    platform: str
    timestamp_utc: str  # ISO 8601
    reference_image_hashes: dict[str, str]  # filename -> SHA256
    num_fixed_refs: int
    discovered_reference_count: int = 0
    uv_lock_hash: str | None = None
    comparison_provider: str = "gemini"


@dataclass(frozen=True)
class PromotionDecision:
    """One promotion decision logged per iteration."""

    iteration: int
    candidate_score: float
    baseline_score: float
    epsilon: float
    delta: float
    decision: str  # "promoted" | "rejected" | "exploration"
    reason: str
    candidate_branch_id: int
    candidate_hypothesis: str
    replicate_scores: list[float] | None = None
    p_value: float | None = None
    test_statistic: float | None = None


@dataclass(frozen=True)
class PromotionTestResult:
    """Statistical test result for promotion decisions (rigorous mode)."""

    statistic: float
    p_value: float  # one-sided
    effect_size: float  # mean paired difference
    ci_lower: float  # 95% CI lower bound of mean difference
    ci_upper: float
    passed: bool  # p < 0.10 AND effect_size > 0


@dataclass
class ReplicatedEvaluation:
    """Replicated evaluation for confirmatory validation (rigorous mode)."""

    template: PromptTemplate
    branch_id: int
    replicate_scores: list[list[MetricScores]]  # [replicate][image]
    replicate_aggregated: list[AggregatedMetrics]  # per-replicate
    median_per_image: list[MetricScores]  # median across replicates per image
    median_aggregated: AggregatedMetrics  # aggregated from medians
