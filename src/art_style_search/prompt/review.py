"""Independent review (CycleResearcher-inspired): skeptical reviewer over experiment results."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from art_style_search.prompt._format import _format_metrics
from art_style_search.types import AggregatedMetrics, IterationResult, KnowledgeBase, ReviewResult
from art_style_search.utils import ReasoningClient

if TYPE_CHECKING:
    from art_style_search.experiment import ExperimentProposal


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
    "<assessments>\nOne paragraph per experiment: [EXP_ID] SIGNAL|NOISE|MIXED — explanation\n</assessments>\n"
    "<noise_vs_signal>Overall analysis of which metric movements are real</noise_vs_signal>\n"
    "<strategic_guidance>What next iteration should focus on</strategic_guidance>\n"
    "<recommended_categories>comma-separated list of categories to target</recommended_categories>"
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
            f"Changed section: {prop.experiment_desc}\n"
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

    kb_text = knowledge_base.render_for_claude(max_words=500)
    if kb_text:
        user_parts.append(f"\n{kb_text}\n")

    user = "\n".join(user_parts)
    text = await client.call(model=model, system=_REVIEW_SYSTEM, user=user, max_tokens=6000)

    # Parse response
    assessments_match = re.search(r"<assessments>(.*?)</assessments>", text, re.DOTALL)
    noise_match = re.search(r"<noise_vs_signal>(.*?)</noise_vs_signal>", text, re.DOTALL)
    guidance_match = re.search(r"<strategic_guidance>(.*?)</strategic_guidance>", text, re.DOTALL)
    cats_match = re.search(r"<recommended_categories>(.*?)</recommended_categories>", text, re.DOTALL)

    assessments_raw = assessments_match.group(1).strip() if assessments_match else ""
    experiment_assessments = [line.strip() for line in assessments_raw.split("\n") if line.strip()]

    return ReviewResult(
        experiment_assessments=experiment_assessments,
        noise_vs_signal=noise_match.group(1).strip() if noise_match else "",
        strategic_guidance=guidance_match.group(1).strip() if guidance_match else "",
        recommended_categories=[
            c.strip() for c in (cats_match.group(1).strip().split(",") if cats_match else []) if c.strip()
        ],
    )
