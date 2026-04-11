"""Independent review (CycleResearcher-inspired): skeptical reviewer over experiment results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from art_style_search.prompt._format import _format_metrics, format_knowledge_base
from art_style_search.prompt.json_contracts import schema_hint, validate_review_payload
from art_style_search.types import AggregatedMetrics, IterationResult, KnowledgeBase, ReviewResult
from art_style_search.utils import ReasoningClient

if TYPE_CHECKING:
    from art_style_search.contracts import ExperimentProposal


_REVIEW_SYSTEM = (
    "You are a critical scientific reviewer evaluating prompt optimization experiments.\n"
    "Your role is INDEPENDENT from the proposer — be skeptical and evidence-based.\n\n"
    "For each experiment, assess:\n"
    "1. Did the metric changes actually support the hypothesis? (Many metric movements are noise)\n"
    "2. Was the hypothesis specific enough to be falsifiable?\n"
    "3. Did the changed section actually cause the observed effects, or could it be coincidence?\n"
    "4. Are there confounding factors (e.g., caption length changed alongside content)?\n\n"
    "Then provide strategic guidance:\n"
    "- Which categories have the most room for improvement based on metric gaps?\n"
    "- Are there hypotheses that keep getting proposed but never confirmed? Flag them.\n"
    "- What is the single most impactful thing the next iteration should try?\n\n"
    "Be brutally honest. A confirmed improvement of +0.005 on one metric while others stayed flat "
    "is likely noise, not a real improvement. Look for consistent patterns across multiple metrics.\n\n"
    "Respond with:\n"
    '{\n'
    '  "experiment_assessments": ["[EXP_ID] SIGNAL|NOISE|MIXED - explanation"],\n'
    '  "noise_vs_signal": "...",\n'
    '  "strategic_guidance": "...",\n'
    '  "recommended_categories": ["color_palette", "composition"]\n'
    "}\n"
    "Return JSON only. No markdown fences, no commentary."
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
    user_parts.append("## Experiments to Review\n")

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
            delta_ds = m.dreamsim_similarity_mean - baseline_metrics.dreamsim_similarity_mean
            delta_color = m.color_histogram_mean - baseline_metrics.color_histogram_mean
            delta_hps = m.hps_score_mean - baseline_metrics.hps_score_mean
            user_parts.append(f"Deltas vs baseline: DS={delta_ds:+.4f} Color={delta_color:+.4f} HPS={delta_hps:+.4f}\n")

    if baseline_metrics:
        user_parts.append(f"\n## Baseline Metrics\n{_format_metrics(baseline_metrics)}\n")

    kb_text = format_knowledge_base(knowledge_base, max_words=500)
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
        max_tokens=6000,
    )
