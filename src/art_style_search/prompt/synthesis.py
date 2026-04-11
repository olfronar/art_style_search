"""Synthesis: merge best sections from top experiments into one combined template."""

from __future__ import annotations

import logging

from art_style_search.prompt._format import _format_metrics, _format_style_profile, _format_template
from art_style_search.prompt.json_contracts import schema_hint, validate_synthesis_payload
from art_style_search.types import IterationResult, PromptTemplate, StyleProfile
from art_style_search.utils import ReasoningClient

logger = logging.getLogger(__name__)


async def synthesize_templates(
    experiments: list[IterationResult],
    style_profile: StyleProfile,
    *,
    client: ReasoningClient,
    model: str,
) -> tuple[PromptTemplate, str]:
    """Merge the best aspects of multiple experiments into one template.

    Takes the top-performing experiments and asks the reasoning model to
    combine their strongest sections into a single template. Returns
    ``(merged_template, hypothesis)``.
    """

    system = (
        "You are an expert art director. You are given several meta-prompt templates that were "
        "tested in parallel. Each produced different strengths — some improved color accuracy, "
        "others improved composition or character fidelity.\n\n"
        "Your task: MERGE the best aspects of each template into one combined template.\n\n"
        "Rules:\n"
        "- For each section, pick the best version from the experiments based on per-image scores.\n"
        "- If two experiments improved different aspects, combine their strengths.\n"
        "- Do NOT average or water down — pick the strongest phrasing for each section.\n"
        "- Keep the template 8-15 sections, 1200-1800 words rendered.\n"
        "- Preserve embedded style rules in section values.\n"
        "- MANDATORY: The first section must be 'style_foundation' with fixed style rules. "
        "The first entry in caption_sections must be 'Art Style'. Never remove these.\n"
        "- Merge caption output structure: pick the best caption_sections ordering and caption_length "
        "from the experiments, or combine them.\n\n"
        "Response format:\n"
        '{\n'
        '  "rationale": "...",\n'
        '  "template": {\n'
        '    "sections": [{"name": "...", "description": "...", "value": "..."}],\n'
        '    "negative_prompt": "...",\n'
        '    "caption_sections": ["Art Style", "Color Palette"],\n'
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
