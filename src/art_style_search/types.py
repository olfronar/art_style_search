"""Shared data structures for the art style search loop."""

from __future__ import annotations

import enum
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


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

    # Per-image Gemini vision scores (ternary: MATCH=1.0, PARTIAL=0.5, MISS=0.0)
    vision_style: float = 0.5
    vision_style_std: float = 0.0
    vision_subject: float = 0.5
    vision_subject_std: float = 0.0
    vision_composition: float = 0.5
    vision_composition_std: float = 0.0

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
            "vision_style": self.vision_style,
            "vision_style_std": self.vision_style_std,
            "vision_subject": self.vision_subject,
            "vision_subject_std": self.vision_subject_std,
            "vision_composition": self.vision_composition,
            "vision_composition_std": self.vision_composition_std,
        }


_HPS_CEILING = 0.35  # default empirical max for HPS v2 scores; used to normalize to [0, 1]

# Improvement must exceed this threshold to be accepted (filters generation noise)
IMPROVEMENT_EPSILON = 0.005
_EPSILON_FLOOR = 0.001  # Minimum epsilon to prevent false-positive improvements at high baselines


def improvement_epsilon(baseline: float) -> float:
    """Threshold that shrinks as baseline score climbs, with a minimum floor.

    At baseline 0.45 → ~0.00275. At 0.51 → ~0.00245. At 0.90 → 0.001 (floor).
    When baseline is -inf (no prior metrics), falls back to IMPROVEMENT_EPSILON.
    """
    clamped = min(max(baseline, 0.0), 1.0)
    return max(IMPROVEMENT_EPSILON * (1.0 - clamped), _EPSILON_FLOOR)


def _normalize_hps(raw: float, ceiling: float = _HPS_CEILING) -> float:
    """Normalize raw HPS v2 score to [0, 1] using the empirical ceiling."""
    return min(raw / ceiling, 1.0)


def composite_score(m: AggregatedMetrics) -> float:
    """Fixed-weight composite score used for absolute quality comparison.

    All metrics normalized to ~[0, 1] before weighting.
    Weights: DreamSim 40%, Color 22%, SSIM 11%, HPS 5%,
    Aesthetics 6%, StyleConsistency 4%, Vision 4%+4%+4%=12%.  Total = 1.00.
    Includes a consistency penalty based on per-image score variance.
    """
    base = (
        0.40 * m.dreamsim_similarity_mean
        + 0.05 * _normalize_hps(m.hps_score_mean)
        + 0.06 * (m.aesthetics_score_mean / 10.0)
        + 0.22 * m.color_histogram_mean
        + 0.11 * m.ssim_mean
        + 0.04 * m.style_consistency
        + 0.04 * m.vision_style
        + 0.04 * m.vision_subject
        + 0.04 * m.vision_composition
    )
    # Penalize inconsistency: high std across images means unreliable reproduction
    variance_penalty = 0.30 * (m.dreamsim_similarity_std + m.color_histogram_std) / 2.0
    return base - variance_penalty


def adaptive_composite_score(
    target: AggregatedMetrics,
    all_results: list[AggregatedMetrics],
) -> float:
    """Score with adaptive weights proportional to cross-experiment variance.

    Metrics that differentiate between experiments get higher weight.
    Falls back to fixed weights for single-experiment evaluation.
    """
    if len(all_results) < 2:
        return composite_score(target)

    # All metrics are higher=better after normalization
    metric_extractors: list[Callable[[AggregatedMetrics], float]] = [
        lambda r: r.dreamsim_similarity_mean,
        lambda r: _normalize_hps(r.hps_score_mean),
        lambda r: r.aesthetics_score_mean / 10.0,
        lambda r: r.color_histogram_mean,
        lambda r: r.ssim_mean,
        lambda r: r.style_consistency,
        lambda r: r.vision_style,
        lambda r: r.vision_subject,
        lambda r: r.vision_composition,
    ]

    weighted_sum = 0.0
    total_weight = 0.0

    for extractor in metric_extractors:
        values = [extractor(r) for r in all_results]
        target_val = extractor(target)

        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

        if std < 1e-8:
            continue

        vmin, vmax = min(values), max(values)
        rng = vmax - vmin
        if rng < 1e-8:
            continue
        normalized = (target_val - vmin) / rng

        weighted_sum += std * normalized
        total_weight += std

    if total_weight < 1e-8:
        return composite_score(target)

    return weighted_sum / total_weight


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
    Claude can propose changes to either the template or the values.

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
    CLAUDE_STOP = "claude_stop"


# ---------------------------------------------------------------------------
# Knowledge Base — structured iteration-to-iteration learning
# ---------------------------------------------------------------------------

# Synonym map for category auto-classification
_CATEGORY_SYNONYMS: dict[str, list[str]] = {
    "color_palette": ["color", "hue", "palette", "saturation", "tone", "gradient", "shade"],
    "composition": ["layout", "framing", "spatial", "arrangement", "perspective", "depth"],
    "technique": ["medium", "brushwork", "brushstroke", "rendering", "stroke", "paint", "watercolor"],
    "mood_atmosphere": ["mood", "atmosphere", "emotion", "feeling", "ambiance", "tone"],
    "lighting": ["light", "shadow", "illumination", "glow", "highlight", "contrast"],
    "texture": ["texture", "surface", "grain", "detail", "pattern"],
    "subject_matter": ["subject", "character", "figure", "object", "scene"],
    "background": ["background", "environment", "setting", "landscape", "sky"],
    "caption_structure": ["section", "label", "order", "ordering", "structure", "format", "length"],
}


def classify_hypothesis(text: str, categories: list[str]) -> str:
    """Auto-classify a hypothesis into a category via keyword matching."""
    text_lower = text.lower()
    best_cat = "general"
    best_score = 0
    for cat in categories:
        score = 0
        # Match words from the category name itself
        for word in cat.replace("_", " ").split():
            if word in text_lower:
                score += 1
        # Match synonyms
        for synonym in _CATEGORY_SYNONYMS.get(cat, []):
            if synonym in text_lower:
                score += 1
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat if best_score > 0 else "general"


def get_category_names(template: PromptTemplate) -> list[str]:
    """Merge StyleProfile field names and PromptSection names into canonical category set."""
    cats = {"color_palette", "composition", "technique", "mood_atmosphere", "subject_matter"}
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
    """A ranked open problem — proposed by Claude, validated by code."""

    text: str  # Claude's description
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
    ) -> Hypothesis:
        """Create a hypothesis, append it, and update category progress."""
        hid = f"H{self.next_id}"
        self.next_id += 1

        # Determine outcome from kept + lesson text signals
        if kept and confirmed and not rejected:
            outcome = "confirmed"
        elif not kept and rejected:
            outcome = "rejected"
        elif kept and rejected:
            outcome = "partial"
        elif not kept:
            outcome = "rejected"
        else:
            outcome = "confirmed" if kept else "rejected"

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

    def render_for_claude(self, max_words: int = 1500) -> str:
        """Produce a structured, budget-enforced prompt section for the reasoning model.

        Priority order (fill until budget exhausted):
        1. Open Problems (~60 words)
        2. Per-Category Status — count + latest insight only (not all insights)
        3. Hypothesis Chain — last 5 full, rest as one-liners
        Rejected hypotheses are already marked REJECTED in the tree — no standalone section.
        """
        if not self.hypotheses:
            return ""

        num_cats = len(self.categories)
        header = f"## Knowledge Base ({len(self.hypotheses)} hypotheses across {num_cats} categories)\n"

        # --- Section 1: Open Problems (always included) ---
        problems_text = ""
        if self.open_problems:
            lines = ["### Open Problems"]
            for i, prob in enumerate(self.open_problems, 1):
                gap_str = f" \u2014 gap: {prob.metric_gap:.3f}" if prob.metric_gap is not None else ""
                lines.append(f"{i}. [{prob.priority}] {prob.text}{gap_str}")
            problems_text = "\n".join(lines)

        # --- Section 2: Per-Category Status (compact: count + latest insight) ---
        cat_lines = ["### Per-Category Status"]
        for cat_name in sorted(self.categories):
            cat = self.categories[cat_name]
            delta_str = f", best Δ: {cat.best_perceptual_delta:+.3f}" if cat.best_perceptual_delta is not None else ""
            n_conf = len(cat.confirmed_insights)
            n_rej = len(cat.rejected_approaches)
            cat_lines.append(
                f"**{cat_name}** [{len(cat.hypothesis_ids)} hyp, {n_conf} confirmed, {n_rej} rejected{delta_str}]"
            )
            if cat.confirmed_insights:
                cat_lines.append(f"  Latest: {cat.confirmed_insights[-1][:120]}")
            if cat.rejected_approaches:
                cat_lines.append(f"  Last rejected: {cat.rejected_approaches[-1][:150]}")
        cat_text = "\n".join(cat_lines)

        # --- Section 3: Hypothesis Chain (last 5 full, rest collapsed) ---
        recent_ids = {h.id for h in self.hypotheses[-5:]}

        def _render_hyp(h: Hypothesis, indent: int, full: bool) -> str:
            prefix = "  " * indent + ("\u2514\u2500 " if indent > 0 else "")
            builds = f", builds on {h.parent_id}" if h.parent_id else ""
            if full:
                stmt = h.statement[:80] + ("..." if len(h.statement) > 80 else "")
                return f'{prefix}{h.id} (iter {h.iteration}, {h.category}{builds}) \u2192 {h.outcome.upper()}: "{stmt}"'
            return f"{prefix}{h.id} (iter {h.iteration}, {h.category}) \u2192 {h.outcome.upper()}"

        roots: list[Hypothesis] = []
        children_map: dict[str, list[Hypothesis]] = {}
        for h in self.hypotheses:
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
        known_ids = {h.id for h in self.hypotheses}
        for h in self.hypotheses:
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
            for h in self.hypotheses[-5:]:
                short_lines.append(_render_hyp(h, 0, full=True))
            result_parts.append("\n" + "\n".join(short_lines))

        return "\n".join(result_parts)

    def suggest_target_categories(self, num_targets: int, categories: list[str]) -> list[str]:
        """Rank categories by improvement potential for diverse experiment targeting."""
        scored: list[tuple[str, float]] = []
        for cat in categories:
            progress = self.categories.get(cat)
            if not progress:
                scored.append((cat, 1.0))  # unexplored = high priority
                continue
            n_confirmed = len(progress.confirmed_insights)
            n_rejected = len(progress.rejected_approaches)
            n_total = len(progress.hypothesis_ids)

            if n_rejected >= 3 and n_confirmed == 0:
                score = 0.1  # diminishing returns
            elif n_confirmed > 0 and progress.best_perceptual_delta and progress.best_perceptual_delta > 0:
                score = 0.7  # partial success, room to build
            else:
                score = 0.5 / max(n_total, 1)
            scored.append((cat, score))

        scored.sort(key=lambda x: -x[1])
        return [cat for cat, _ in scored[:num_targets]]


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
