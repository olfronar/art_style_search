"""Independent review (CycleResearcher-inspired): skeptical reviewer over experiment results."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from art_style_search.prompt._format import _format_metrics, format_knowledge_base
from art_style_search.prompt.json_contracts import response_schema, schema_hint, validate_review_payload
from art_style_search.types import (
    AggregatedMetrics,
    IterationResult,
    KnowledgeBase,
    ReviewResult,
    compliance_components_mean,
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


def _compliance_mean(metrics: AggregatedMetrics) -> float:
    return compliance_components_mean(
        metrics.compliance_topic_coverage,
        metrics.compliance_marker_coverage,
        metrics.section_ordering_rate,
        metrics.section_balance_rate,
        metrics.subject_specificity_rate,
    )


def _delta_summary(metrics: AggregatedMetrics, baseline: AggregatedMetrics) -> str:
    return (
        "Deltas vs baseline: "
        f"DS={metrics.dreamsim_similarity_mean - baseline.dreamsim_similarity_mean:+.4f} "
        f"Color={metrics.color_histogram_mean - baseline.color_histogram_mean:+.4f} "
        f"HPS={metrics.hps_score_mean - baseline.hps_score_mean:+.4f} "
        f"SSIM={metrics.ssim_mean - baseline.ssim_mean:+.4f} "
        f"Aes={metrics.aesthetics_score_mean - baseline.aesthetics_score_mean:+.4f} "
        f"vision_style={metrics.vision_style - baseline.vision_style:+.4f} "
        f"vision_subject={metrics.vision_subject - baseline.vision_subject:+.4f} "
        f"vision_composition={metrics.vision_composition - baseline.vision_composition:+.4f} "
        f"style_consistency={metrics.style_consistency - baseline.style_consistency:+.4f} "
        f"completion_rate={metrics.completion_rate - baseline.completion_rate:+.4f} "
        f"compliance={_compliance_mean(metrics) - _compliance_mean(baseline):+.4f}\n"
    )


def _noise_floor_summary(experiments: list[IterationResult]) -> str:
    if len(experiments) < 2:
        return ""

    def _std(values: list[float]) -> float:
        return statistics.pstdev(values) if len(values) >= 2 else 0.0

    metrics = [exp.aggregated for exp in experiments]
    return (
        "## Noise floors for this run\n"
        f"DS=±{_std([m.dreamsim_similarity_mean for m in metrics]):.4f} "
        f"Color=±{_std([m.color_histogram_mean for m in metrics]):.4f} "
        f"HPS=±{_std([m.hps_score_mean for m in metrics]):.4f} "
        f"SSIM=±{_std([m.ssim_mean for m in metrics]):.4f} "
        f"Aes=±{_std([m.aesthetics_score_mean for m in metrics]):.4f} "
        f"vision_style=±{_std([m.vision_style for m in metrics]):.4f} "
        f"vision_subject=±{_std([m.vision_subject for m in metrics]):.4f} "
        f"vision_composition=±{_std([m.vision_composition for m in metrics]):.4f} "
        f"style_consistency=±{_std([m.style_consistency for m in metrics]):.4f} "
        f"completion_rate=±{_std([m.completion_rate for m in metrics]):.4f} "
        f"compliance=±{_std([_compliance_mean(m) for m in metrics]):.4f}\n"
    )


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
    )
