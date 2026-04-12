"""Zero-step: propose N diverse initial meta-prompt templates."""

from __future__ import annotations

import logging

from art_style_search.prompt._format import _format_style_profile
from art_style_search.prompt.json_contracts import schema_hint, validate_initial_templates_payload
from art_style_search.types import PromptTemplate, StyleProfile
from art_style_search.utils import ReasoningClient

logger = logging.getLogger(__name__)


async def propose_initial_templates(
    style_profile: StyleProfile,
    num_branches: int,
    *,
    client: ReasoningClient,
    model: str,
) -> list[PromptTemplate]:
    """Generate diverse initial prompt templates for population branches."""

    system = (
        "You are an expert art director and prompt engineer. Your task is to create "
        "diverse META-PROMPTS — instructions that tell an AI vision model (Gemini Pro) "
        "HOW to describe/caption reference images.\n\n"
        "## Goal\n"
        "The captions produced must serve a DUAL purpose:\n"
        "1. Contain enough detail to RECREATE the original image (we measure this with metrics).\n"
        "2. Embed REUSABLE art-style guidance — labeled sections with style rules, color palette, "
        "technique descriptions, etc. — that can later be applied to generate NEW art in the same style "
        "with different subjects.\n\n"
        "## How the system works\n"
        "meta-prompt + reference image → Gemini Pro caption → Gemini Flash generation → compare with original.\n"
        "The meta-prompt is the ONLY thing being optimized. It must instruct the captioner "
        "to describe every detail needed for faithful recreation AND embed the art-style guidance "
        "from the Style Profile into every caption as reusable style rules.\n\n"
        "## Meta-prompt requirements\n"
        "- 8-15 sections, each instructing the captioner WHAT to describe and HOW precisely.\n"
        "- Must cover: technique/medium, colors, composition, characters/figures, "
        "background/environment, textures/details, lighting, mood/atmosphere.\n"
        "- Sections should EMBED the core style rules from the Style Profile as literal text — "
        "the captioner should weave these rules into every caption as a shared style foundation, "
        "then add per-image observations on top.\n"
        "- Each section should be 4-8 sentences of instruction.\n"
        "- Total rendered meta-prompt should be 1200-1800 words.\n"
        "- Include a negative section: what the captioner should tell the generator to AVOID.\n"
        "- The meta-prompt must produce captions specific enough that someone who has never "
        "seen the image could recreate it from the caption alone.\n\n"
        "## MANDATORY: Anchor sections\n"
        "Every template MUST include a section named 'style_foundation' as the FIRST section. "
        "This section instructs the captioner to open every caption with an [Art Style] block "
        "containing FIXED, REUSABLE style rules copied verbatim from the Style Profile. "
        "The [Art Style] block must be nearly IDENTICAL across all captions — it is the shared "
        "style DNA that enables generating new art in the same style with different subjects. "
        "Keep this section first and reusable, but you MAY refine its wording and specificity.\n"
        "Every template MUST also include a section named 'subject_anchor' as the SECOND section. "
        "It instructs the captioner to produce a [Subject] block immediately after [Art Style], covering "
        "identity/species, distinguishing features, clothing/equipment, pose/action, expression, "
        "and props/context with concrete detail. Keep this section second, but you MAY refine its wording.\n\n"
        "## Caption output structure\n"
        "The meta-prompt must instruct the captioner to produce captions with LABELED SECTIONS. "
        "The FIRST section must always be [Art Style] (the shared style block). "
        "The SECOND section must always be [Subject]. "
        "You decide the remaining sections and their order — that is part of experimentation.\n"
        "- Specify the caption output sections as a <caption_sections> tag (comma-separated list). "
        "The first two entries MUST be 'Art Style' and 'Subject'.\n"
        "- Specify the target caption length as a <caption_length> tag (word count).\n"
        "- The [Art Style] section should be IDENTICAL across captions (shared style rules). "
        "The [Subject] section should be image-specific and rich enough for faithful subject reconstruction. "
        "Target roughly 80-140 words for [Subject]. All remaining sections contain per-image specific observations.\n"
        "- A style_consistency metric measures how similar the [Art Style] blocks are across "
        "captions — higher consistency is rewarded in the composite score.\n\n"
        "## Example of a good meta-prompt section\n"
        '<section name="colors_and_palette" description="instruct captioner on color description '
        'with embedded style rules">'
        "This art style uses a warm earth-tone palette dominated by burnt sienna, raw umber, "
        "and cadmium yellow, with cool accents of cerulean blue. Saturation is moderate — colors "
        "feel muted and aged rather than vibrant. "
        "When describing the image, note the EXACT colors visible using specific color names "
        "(not just 'brown' or 'yellow'). Describe the overall color temperature, saturation levels, "
        "and how colors relate to each other. Note any gradients, color transitions, or areas where "
        "the palette deviates from the core warm-earth foundation. "
        "In your [Color Palette] section, first state the core style palette rules, then describe "
        "how this specific image applies or varies from them."
        "</section>\n\n"
        "## Diversity across meta-prompts\n"
        "- Vary the set of caption output section names and their ordering.\n"
        "- Vary caption length targets (e.g. 400, 600, 800 words).\n"
        "- Vary emphasis: some focus on technique precision, others on spatial accuracy, "
        "others on mood fidelity.\n"
        "- Vary the balance between shared style guidance vs per-image detail.\n"
        "- Vary instruction style: some give the captioner strict checklists, "
        "others give artistic direction, others ask for technical analysis.\n"
        "- All must be comprehensive — diversity is in approach, not coverage.\n\n"
        f"Produce exactly {num_branches} meta-prompts in one JSON object.\n\n"
        "Return EXACTLY one JSON object with this shape:\n"
        '{\n'
        '  "templates": [\n'
        "    {\n"
        '      "sections": [{"name": "style_foundation", "description": "...", "value": "..."}, {"name": "subject_anchor", "description": "...", "value": "..."}],\n'
        '      "negative_prompt": "...",\n'
        '      "caption_sections": ["Art Style", "Subject", "Color Palette"],\n'
        '      "caption_length_target": 500\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Return JSON only. No markdown fences, no commentary."
    )

    user = (
        "Based on the following style profile of the reference images, create the initial "
        "meta-prompts. Remember: these are INSTRUCTIONS for a captioner, not direct image prompts.\n\n"
        f"{_format_style_profile(style_profile)}"
    )

    logger.info("Requesting %d initial templates (%s)", num_branches, model)

    templates = await client.call_json(
        model=model,
        system=system,
        user=user,
        validator=lambda data: validate_initial_templates_payload(data, num_branches=num_branches),
        response_name="initial_templates",
        schema_hint=schema_hint("initial_templates"),
        max_tokens=16000,
    )

    for i, t in enumerate(templates):
        if not t.sections:
            logger.warning("Branch %d initial template has no sections — raw response may need review", i)

    return templates
