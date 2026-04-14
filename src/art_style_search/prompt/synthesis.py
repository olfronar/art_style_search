"""Synthesis: merge best sections from top experiments into one combined template."""

from __future__ import annotations

import logging

from art_style_search.prompt._format import _format_metrics, _format_style_profile, _format_template
from art_style_search.prompt.json_contracts import schema_hint, validate_synthesis_payload
from art_style_search.types import AggregatedMetrics, IterationResult, PromptTemplate, StyleProfile
from art_style_search.utils import ReasoningClient

logger = logging.getLogger(__name__)


def _metric_strength_annotations(exp: IterationResult, baseline: AggregatedMetrics | None) -> str:
    """Annotate which metrics this experiment improved vs baseline."""
    if baseline is None:
        return "No baseline available for comparison."
    m = exp.aggregated
    strengths: list[str] = []
    weaknesses: list[str] = []

    deltas = [
        ("DreamSim", m.dreamsim_similarity_mean - baseline.dreamsim_similarity_mean),
        ("Color histogram", m.color_histogram_mean - baseline.color_histogram_mean),
        ("SSIM", m.ssim_mean - baseline.ssim_mean),
        ("HPS v2", m.hps_score_mean - baseline.hps_score_mean),
        ("Aesthetics", m.aesthetics_score_mean - baseline.aesthetics_score_mean),
        ("vision_style", m.vision_style - baseline.vision_style),
        ("vision_subject", m.vision_subject - baseline.vision_subject),
        ("vision_composition", m.vision_composition - baseline.vision_composition),
        ("style_consistency", m.style_consistency - baseline.style_consistency),
    ]

    for name, delta in deltas:
        if delta > 0.01:
            strengths.append(f"{name} +{delta:.3f}")
        elif delta < -0.01:
            weaknesses.append(f"{name} {delta:.3f}")

    parts: list[str] = []
    if strengths:
        parts.append("Strengths: " + ", ".join(strengths))
    if weaknesses:
        parts.append("Weaknesses: " + ", ".join(weaknesses))
    return "; ".join(parts) if parts else "All metrics within ±0.01 of baseline."


async def synthesize_templates(
    experiments: list[IterationResult],
    style_profile: StyleProfile,
    *,
    client: ReasoningClient,
    model: str,
    baseline_metrics: AggregatedMetrics | None = None,
) -> tuple[PromptTemplate, str]:
    """Merge the best aspects of multiple experiments into one template.

    Takes the top-performing experiments and asks the reasoning model to
    combine their strongest sections into a single template. Returns
    ``(merged_template, hypothesis)``.

    When *baseline_metrics* is provided, each experiment is annotated with
    per-section quality signals (which metrics it improved vs baseline) so the
    reasoning model can make informed section-selection decisions.
    """

    system = (
        "You are an expert art director merging several meta-prompt templates that were tested in parallel. "
        "Each produced different strengths — some improved color accuracy, others improved composition or character fidelity.\n\n"
        "Your task: MERGE the best aspects of each template into one combined template.\n\n"
        "## NON-NEGOTIABLE RULES\n"
        "- The first section MUST be 'style_foundation'. The second MUST be 'subject_anchor'.\n"
        "- caption_sections MUST start with ['Art Style', 'Subject', ...].\n"
        "- Do NOT average or water down sections — pick the strongest phrasing verbatim.\n"
        "  Why: averaging removes the specificity that made a section effective.\n\n"
        "## Decision guidance\n"
        "Each experiment is annotated with its metric STRENGTHS vs baseline. Use these annotations to guide section selection:\n"
        "- If an experiment improved DreamSim and SSIM, its structural sections (composition, subject_anchor) are likely strong.\n"
        "- If it improved color_histogram, its color_palette section is likely strong.\n"
        "- If it improved vision_style, its technique/style sections are likely strong.\n"
        "- If it improved vision_subject, its subject_anchor section is likely strong.\n"
        "- If two experiments improved DIFFERENT aspects, combine their strongest sections — this is the ideal outcome.\n"
        "- When metric annotations are ambiguous, prefer the experiment with higher overall DreamSim.\n"
        "For caption_sections ordering: prefer the experiment with higher style_consistency.\n"
        "For caption_length_target: prefer the experiment with higher completion_rate.\n"
        "Preserve embedded style rules in section values. Keep the template 8-15 sections, 1200-1800 words rendered.\n\n"
        "## EXECUTION CHECKLIST — verify before outputting\n"
        "- [ ] First section is 'style_foundation', second is 'subject_anchor'\n"
        "- [ ] caption_sections starts with ['Art Style', 'Subject']\n"
        "- [ ] No section was averaged or merged — each comes from a single experiment\n"
        "- [ ] Template has 8-15 sections\n"
        "- [ ] Total rendered word count is 1200-1800\n"
        "- [ ] All embedded style rules are preserved\n\n"
        "Response format:\n"
        "{\n"
        '  "rationale": "...",\n'
        '  "template": {\n'
        '    "sections": [{"name": "style_foundation", "description": "...", "value": "..."}, {"name": "subject_anchor", "description": "...", "value": "..."}, {"name": "color_palette", "description": "...", "value": "..."}, {"name": "composition", "description": "...", "value": "..."}],\n'
        '    "negative_prompt": "...",\n'
        '    "caption_sections": ["Art Style", "Subject", "Color Palette"],\n'
        '    "caption_length_target": 500\n'
        "  }\n"
        "}\n"
        "Return JSON only. No markdown fences, no commentary."
    )

    user_parts: list[str] = [
        "## Style Profile\n",
        _format_style_profile(style_profile, compact=True),
        "\n\n## Experiments to Merge\n\n",
    ]

    for exp in experiments:
        kept_label = "BEST" if exp.kept else "IMPROVED"
        user_parts.append(f"### Experiment {exp.branch_id} [{kept_label}]\n")
        user_parts.append(f"Hypothesis: {exp.hypothesis}\n")
        user_parts.append(f"Metrics:\n{_format_metrics(exp.aggregated)}\n")
        annotations = _metric_strength_annotations(exp, baseline_metrics)
        user_parts.append(f"Metric analysis: {annotations}\n")
        user_parts.append(f"Template:\n{_format_template(exp.template)}\n\n")

    user_parts.append(
        "Merge these into one template that combines the strongest aspects of each. "
        "Focus on sections where specific experiments showed clear metric advantages."
    )

    user = "".join(user_parts)

    logger.info("Requesting template synthesis (%s)", model)

    merged, rationale = await client.call_json(
        model=model,
        system=system,
        user=user,
        validator=validate_synthesis_payload,
        response_name="synthesis",
        schema_hint=schema_hint("synthesis"),
        max_tokens=12000,
    )

    if not merged.sections:
        logger.warning("Synthesis produced no sections — falling back to best experiment's template")
        merged = experiments[0].template

    hypothesis = (
        f"Synthesis: combining best sections from experiments {', '.join(str(e.branch_id) for e in experiments)}"
    )
    if rationale:
        hypothesis += f" — {rationale[:200]}"

    return merged, hypothesis
