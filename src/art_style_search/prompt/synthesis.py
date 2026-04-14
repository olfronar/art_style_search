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
        "## TIER 1: NON-NEGOTIABLE RULES (CRITICAL)\n"
        "- The first section MUST be 'style_foundation'. The second MUST be 'subject_anchor'.\n"
        "- caption_sections MUST start with ['Art Style', 'Subject', ...].\n"
        "- Do NOT average or water down sections — pick the strongest phrasing verbatim.\n"
        "  Why: averaging removes the specificity that made a section effective.\n\n"
        "## TIER 2: DECISION CRITERIA FOR MERGING\n"
        "When choosing which experiment's version of a section to keep, use this priority:\n"
        "1. The experiment with HIGHER DreamSim similarity → prefer its version.\n"
        "2. If DreamSim ties (within 0.02), break by vision_subject score.\n"
        "3. If still tied, break by color_histogram score.\n"
        "For caption_sections ordering: prefer the experiment with higher style_consistency.\n"
        "For caption_length_target: prefer the experiment with higher completion_rate.\n\n"
        "If two experiments improved DIFFERENT aspects (e.g., one improved colors, another improved "
        "composition), combine their strongest sections — this is the ideal outcome.\n"
        "Preserve embedded style rules in section values. "
        "Keep the template 8-15 sections, 1200-1800 words rendered.\n\n"
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
