"""Independent review (CycleResearcher-inspired): skeptical reviewer over experiment results."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from art_style_search.prompt._format import _format_metrics, format_knowledge_base
from art_style_search.prompt.json_contracts import response_schema, schema_hint, validate_review_payload
from art_style_search.scoring import compliance_mean, metric_deltas
from art_style_search.types import (
    AggregatedMetrics,
    IterationResult,
    KnowledgeBase,
    ReviewResult,
)
from art_style_search.utils import ReasoningClient

if TYPE_CHECKING:
    from art_style_search.contracts import ExperimentProposal


_REVIEW_EXAMPLE = (
    "## Example of a good assessment\n"
    "[EXP 1] MIXED — DreamSim improved +0.03 (above noise floor of ±0.01), confirming the subject-anchor "
    "change helped perceptual similarity. But color_histogram declined -0.05, suggesting the revised "
    "color section lost palette specificity. HPS and Aesthetics were flat (within noise). "
    "The hypothesis was specific and falsifiable — partial confirmation.\n\n"
    "## Example of a bad assessment (too generous — avoid this)\n"
    "[EXP 2] SIGNAL — Metrics improved! — This is not specific enough. Which metrics? By how much? "
    "Were they above noise floor? Was the hypothesis supported or just coincidental?"
)

_REVIEW_SYSTEM = (
    "You are a critical scientific reviewer evaluating prompt optimization experiments.\n"
    "Your role is INDEPENDENT from the proposer — be skeptical and evidence-based.\n\n"
    "## PHASE 1: ASSESS each experiment\n"
    "For each experiment, determine whether the metric changes are real signal or noise:\n"
    "- Did the metric changes actually support the hypothesis?\n"
    "- Was the hypothesis specific enough to be falsifiable?\n"
    "- Did the changed section actually cause the observed effects, or could it be coincidence?\n"
    "- Are there confounding factors (e.g., caption length changed alongside content)?\n"
    "Noise floors are provided below when enough experiments exist. "
    "Changes below the provided noise floor should usually be treated as noise.\n"
    "Classify each as SIGNAL (consistent multi-metric improvement), NOISE (random fluctuation), "
    "or MIXED (some real, some spurious).\n\n"
    "## PHASE 2: SYNTHESIZE across experiments\n"
    "- Which metrics moved consistently across experiments? Those are real signals.\n"
    "- Which moved in only one experiment? Likely noise.\n"
    "- Are there hypotheses that keep getting proposed but never confirmed? Flag them.\n\n"
    "## PHASE 3: STRATEGIC GUIDANCE\n"
    "- Which categories have the most room for improvement based on metric gaps?\n"
    "- What is the single most impactful thing the next iteration should try?\n"
    "- Recommend specific target categories for the next batch.\n\n"
    "## Calibration standards\n"
    "- If fewer than half the experiments show consistent multi-metric improvement, classify most as NOISE or MIXED.\n"
    "- A single metric improvement < 0.01 is within noise range even without formal noise floors.\n"
    "- If no experiment clearly beats baseline, say so explicitly — do not grade on a curve.\n"
    "- 'SIGNAL' requires improvement in at least 2 independent metrics, not just one.\n"
    "- If vision scores improve but DreamSim/SSIM decline, that is MIXED at best (perceptual metrics take priority).\n\n"
    "Respond with:\n"
    "{\n"
    '  "experiment_assessments": ["[EXP_ID] SIGNAL|NOISE|MIXED - explanation"],\n'
    '  "noise_vs_signal": "...",\n'
    '  "strategic_guidance": "...",\n'
    '  "recommended_categories": ["color_palette", "composition"]\n'
    "}\n"
    "Return JSON only. No markdown fences, no commentary."
)


# Short labels for the delta/noise-floor summaries shown in the reviewer prompt.
_REVIEW_DELTA_LABELS: dict[str, str] = {
    "dreamsim_similarity_mean": "DS",
    "color_histogram_mean": "Color",
    "hps_score_mean": "HPS",
    "ssim_mean": "SSIM",
    "aesthetics_score_mean": "Aes",
    "megastyle_similarity_mean": "MegaStyle",
    "vision_style": "vision_style",
    "vision_subject": "vision_subject",
    "vision_composition": "vision_composition",
    "vision_medium": "vision_medium",
    "vision_proportions": "vision_proportions",
    "completion_rate": "completion_rate",
    "compliance": "compliance",
}


def _delta_summary(metrics: AggregatedMetrics, baseline: AggregatedMetrics) -> str:
    deltas = metric_deltas(metrics, baseline)
    parts = " ".join(f"{label}={deltas[attr]:+.4f}" for attr, label in _REVIEW_DELTA_LABELS.items())
    return f"Deltas vs baseline: {parts}\n"


def _noise_floor_summary(experiments: list[IterationResult]) -> str:
    if len(experiments) < 2:
        return ""

    def _std(values: list[float]) -> float:
        return statistics.pstdev(values) if len(values) >= 2 else 0.0

    def _pearson(xs: list[float], ys: list[float]) -> float | None:
        """Sample Pearson correlation, None on degenerate input (constant series / <3 points)."""
        if len(xs) != len(ys) or len(xs) < 3:
            return None
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
        den_x = sum((x - mean_x) ** 2 for x in xs)
        den_y = sum((y - mean_y) ** 2 for y in ys)
        if den_x <= 0 or den_y <= 0:
            return None
        return num / ((den_x**0.5) * (den_y**0.5))

    metrics = [exp.aggregated for exp in experiments]
    value_fns: dict[str, list[float]] = {
        "dreamsim_similarity_mean": [m.dreamsim_similarity_mean for m in metrics],
        "color_histogram_mean": [m.color_histogram_mean for m in metrics],
        "hps_score_mean": [m.hps_score_mean for m in metrics],
        "ssim_mean": [m.ssim_mean for m in metrics],
        "aesthetics_score_mean": [m.aesthetics_score_mean for m in metrics],
        "megastyle_similarity_mean": [m.megastyle_similarity_mean for m in metrics],
        "vision_style": [m.vision_style for m in metrics],
        "vision_subject": [m.vision_subject for m in metrics],
        "vision_composition": [m.vision_composition for m in metrics],
        "vision_medium": [m.vision_medium for m in metrics],
        "vision_proportions": [m.vision_proportions for m in metrics],
        "completion_rate": [m.completion_rate for m in metrics],
        "compliance": [compliance_mean(m) for m in metrics],
    }
    parts = " ".join(f"{label}=±{_std(value_fns[attr]):.4f}" for attr, label in _REVIEW_DELTA_LABELS.items())

    # Vision-verdict vs continuous-metric calibration. Surfaces the observed correlation
    # between each vision-judge axis and its closest continuous-metric counterpart so the
    # reviewer can discount noisy judge signal ("vision_style r=0.1 with DreamSim → treat
    # as independent axis, don't over-interpret"). Low r under noisy judging prevents the
    # reasoner from chasing spurious verdict swings.
    correlation_pairs: tuple[tuple[str, str, str], ...] = (
        ("vision_style", "dreamsim_similarity_mean", "r(vision_style,DS)"),
        ("vision_subject", "dreamsim_similarity_mean", "r(vision_subject,DS)"),
        ("vision_composition", "ssim_mean", "r(vision_composition,SSIM)"),
        # MegaStyle is intended as a content-disentangled style axis — if it correlates
        # strongly with DreamSim (content) in a given run, MegaStyle is likely tracking
        # content rather than style and its movement should be discounted.
        ("megastyle_similarity_mean", "dreamsim_similarity_mean", "r(MegaStyle,DS)"),
    )
    corr_bits: list[str] = []
    for axis_a, axis_b, label in correlation_pairs:
        r = _pearson(value_fns[axis_a], value_fns[axis_b])
        if r is not None:
            corr_bits.append(f"{label}={r:+.3f}")
    calibration_line = f"\nJudge calibration (this iteration): {' '.join(corr_bits)}" if corr_bits else ""
    return f"## Noise floors for this run\n{parts}{calibration_line}\n"


async def review_iteration(
    experiments: list[IterationResult],
    proposals: list[ExperimentProposal],
    baseline_metrics: AggregatedMetrics | None,
    knowledge_base: KnowledgeBase,
    *,
    client: ReasoningClient,
    model: str,
) -> ReviewResult:
    """Independent review of iteration results — skeptical assessment of improvements."""

    user_parts: list[str] = []
    user_parts.append(f"{_REVIEW_EXAMPLE}\n\n")
    user_parts.append("## Experiments to Review\n")
    noise_floor_block = _noise_floor_summary(experiments)
    if noise_floor_block:
        user_parts.append(noise_floor_block + "\n")

    for exp, prop in zip(experiments, proposals, strict=False):
        m = exp.aggregated
        user_parts.append(
            f"### Experiment {exp.branch_id}\n"
            f"Hypothesis: {prop.hypothesis}\n"
            f"Changed section: {prop.changed_section}\n"
            f"Kept: {exp.kept}\n"
            f"Metrics: {_format_metrics(m)}\n"
        )
        if baseline_metrics:
            user_parts.append(_delta_summary(m, baseline_metrics))

    if baseline_metrics:
        user_parts.append(f"\n## Baseline Metrics\n{_format_metrics(baseline_metrics)}\n")

    kb_text = format_knowledge_base(knowledge_base, max_words=1000)
    if kb_text:
        user_parts.append(f"\n{kb_text}\n")

    user = "\n".join(user_parts)
    return await client.call_json(
        model=model,
        system=_REVIEW_SYSTEM,
        user=user,
        validator=validate_review_payload,
        response_name="review",
        schema_hint=schema_hint("review"),
        response_schema=response_schema("review"),
        max_tokens=6000,
        temperature=0.2,
        stage="review",
    )
