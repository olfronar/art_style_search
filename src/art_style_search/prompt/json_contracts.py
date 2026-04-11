"""Validated JSON contracts for reasoning-model exchanges."""

from __future__ import annotations

import json
from typing import Any

from art_style_search.prompt._parse import Lessons, RefinementResult
from art_style_search.types import PromptSection, PromptTemplate, ReviewResult, StyleProfile


def _require_dict(data: object, *, label: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        msg = f"{label} must be a JSON object"
        raise ValueError(msg)
    return data


def _require_list(data: object, *, label: str) -> list[Any]:
    if not isinstance(data, list):
        msg = f"{label} must be a JSON array"
        raise ValueError(msg)
    return data


def _as_str(data: object, *, label: str, default: str = "") -> str:
    if data is None:
        return default
    if not isinstance(data, str):
        msg = f"{label} must be a string"
        raise ValueError(msg)
    return data.strip()


def _as_int(data: object, *, label: str, default: int = 0) -> int:
    if data is None:
        return default
    if isinstance(data, bool):
        msg = f"{label} must be an integer, not a boolean"
        raise ValueError(msg)
    if isinstance(data, int):
        return data
    if isinstance(data, str) and data.strip().isdigit():
        return int(data.strip())
    msg = f"{label} must be an integer"
    raise ValueError(msg)


def _as_str_list(data: object, *, label: str) -> list[str]:
    items = _require_list(data, label=label)
    result: list[str] = []
    for i, item in enumerate(items):
        value = _as_str(item, label=f"{label}[{i}]")
        if value:
            result.append(value)
    return result


def template_to_payload(template: PromptTemplate) -> dict[str, Any]:
    return {
        "sections": [
            {
                "name": section.name,
                "description": section.description,
                "value": section.value,
            }
            for section in template.sections
        ],
        "negative_prompt": template.negative_prompt or "",
        "caption_sections": list(template.caption_sections),
        "caption_length_target": template.caption_length_target,
    }


def style_profile_to_payload(profile: StyleProfile) -> dict[str, Any]:
    return {
        "color_palette": profile.color_palette,
        "composition": profile.composition,
        "technique": profile.technique,
        "mood_atmosphere": profile.mood_atmosphere,
        "subject_matter": profile.subject_matter,
        "influences": profile.influences,
    }


def payload_to_template(data: object, *, label: str = "template") -> PromptTemplate:
    obj = _require_dict(data, label=label)
    sections_raw = _require_list(obj.get("sections") or [], label=f"{label}.sections")
    sections: list[PromptSection] = []
    for i, item in enumerate(sections_raw):
        sec = _require_dict(item, label=f"{label}.sections[{i}]")
        sections.append(
            PromptSection(
                name=_as_str(sec.get("name"), label=f"{label}.sections[{i}].name"),
                description=_as_str(sec.get("description"), label=f"{label}.sections[{i}].description"),
                value=_as_str(sec.get("value"), label=f"{label}.sections[{i}].value"),
            )
        )

    negative_prompt = _as_str(
        obj.get("negative_prompt", obj.get("negative")),
        label=f"{label}.negative_prompt",
        default="",
    )
    caption_sections = _as_str_list(obj.get("caption_sections") or [], label=f"{label}.caption_sections")
    caption_length_target = _as_int(
        obj.get("caption_length_target", obj.get("caption_length")),
        label=f"{label}.caption_length_target",
        default=0,
    )

    return PromptTemplate(
        sections=sections,
        negative_prompt=negative_prompt or None,
        caption_sections=caption_sections,
        caption_length_target=caption_length_target,
    )


def validate_initial_templates_payload(data: object, *, num_branches: int) -> list[PromptTemplate]:
    obj = _require_dict(data, label="initial_templates_response")
    templates = [
        payload_to_template(item, label=f"templates[{i}]")
        for i, item in enumerate(_require_list(obj.get("templates") or [], label="templates"))
    ]
    if not templates:
        raise ValueError("templates must contain at least one template")
    while len(templates) < num_branches:
        templates.append(templates[-1])
    return templates[:num_branches]


def validate_experiment_batch_payload(data: object, *, num_experiments: int) -> tuple[list[RefinementResult], bool]:
    obj = _require_dict(data, label="experiment_batch_response")
    experiments = _require_list(obj.get("experiments") or [], label="experiments")
    results: list[RefinementResult] = []
    for i, item in enumerate(experiments):
        exp = _require_dict(item, label=f"experiments[{i}]")
        lessons_obj = _require_dict(exp.get("lessons") or {}, label=f"experiments[{i}].lessons")
        template = payload_to_template(exp.get("template") or {}, label=f"experiments[{i}].template")
        results.append(
            RefinementResult(
                template=template,
                analysis=_as_str(exp.get("analysis"), label=f"experiments[{i}].analysis"),
                template_changes=_as_str(
                    exp.get("template_changes"),
                    label=f"experiments[{i}].template_changes",
                    default="none",
                ),
                should_stop=False,
                hypothesis=_as_str(exp.get("hypothesis"), label=f"experiments[{i}].hypothesis"),
                experiment=_as_str(exp.get("experiment"), label=f"experiments[{i}].experiment"),
                lessons=Lessons(
                    confirmed=_as_str(lessons_obj.get("confirmed"), label=f"experiments[{i}].lessons.confirmed"),
                    rejected=_as_str(lessons_obj.get("rejected"), label=f"experiments[{i}].lessons.rejected"),
                    new_insight=_as_str(
                        lessons_obj.get("new_insight"),
                        label=f"experiments[{i}].lessons.new_insight",
                    ),
                ),
                builds_on=_as_str(exp.get("builds_on"), label=f"experiments[{i}].builds_on", default="") or None,
                open_problems=_as_str_list(exp.get("open_problems") or [], label=f"experiments[{i}].open_problems"),
                changed_section=_as_str(
                    exp.get("changed_section"),
                    label=f"experiments[{i}].changed_section",
                ),
                target_category=_as_str(
                    exp.get("target_category"),
                    label=f"experiments[{i}].target_category",
                ),
            )
        )

    if len(results) > num_experiments:
        results = results[:num_experiments]

    converged = bool(obj.get("converged", False))
    return results, converged


def validate_synthesis_payload(data: object) -> tuple[PromptTemplate, str]:
    obj = _require_dict(data, label="synthesis_response")
    template = payload_to_template(obj.get("template") or {}, label="template")
    rationale = _as_str(obj.get("rationale"), label="rationale")
    return template, rationale


def validate_review_payload(data: object) -> ReviewResult:
    obj = _require_dict(data, label="review_response")
    return ReviewResult(
        experiment_assessments=_as_str_list(obj.get("experiment_assessments") or [], label="experiment_assessments"),
        noise_vs_signal=_as_str(obj.get("noise_vs_signal"), label="noise_vs_signal"),
        strategic_guidance=_as_str(obj.get("strategic_guidance"), label="strategic_guidance"),
        recommended_categories=_as_str_list(obj.get("recommended_categories") or [], label="recommended_categories"),
    )


def validate_style_compilation_payload(
    data: object,
    *,
    gemini_raw: str,
    reasoning_raw: str,
) -> tuple[StyleProfile, PromptTemplate]:
    obj = _require_dict(data, label="style_compilation_response")
    profile_obj = _require_dict(obj.get("style_profile") or {}, label="style_profile")
    template = payload_to_template(obj.get("initial_template") or {}, label="initial_template")
    profile = StyleProfile(
        color_palette=_as_str(profile_obj.get("color_palette"), label="style_profile.color_palette"),
        composition=_as_str(profile_obj.get("composition"), label="style_profile.composition"),
        technique=_as_str(profile_obj.get("technique"), label="style_profile.technique"),
        mood_atmosphere=_as_str(profile_obj.get("mood_atmosphere"), label="style_profile.mood_atmosphere"),
        subject_matter=_as_str(profile_obj.get("subject_matter"), label="style_profile.subject_matter"),
        influences=_as_str(profile_obj.get("influences"), label="style_profile.influences"),
        gemini_raw_analysis=gemini_raw,
        claude_raw_analysis=reasoning_raw,
    )
    return profile, template


_SCHEMA_HINTS = {
    "initial_templates": {
        "templates": [
            {
                "sections": [{"name": "style_foundation", "description": "core style rules", "value": "..."}],
                "negative_prompt": "...",
                "caption_sections": ["Art Style", "Color Palette"],
                "caption_length_target": 500,
            }
        ]
    },
    "experiment_batch": {
        "experiments": [
            {
                "analysis": "...",
                "lessons": {"confirmed": "...", "rejected": "...", "new_insight": "..."},
                "hypothesis": "...",
                "builds_on": "H3",
                "experiment": "...",
                "changed_section": "color_palette",
                "target_category": "color_palette",
                "open_problems": ["..."],
                "template_changes": "...",
                "template": {
                    "sections": [{"name": "style_foundation", "description": "core style rules", "value": "..."}],
                    "negative_prompt": "...",
                    "caption_sections": ["Art Style"],
                    "caption_length_target": 500,
                },
            }
        ],
        "converged": False,
    },
    "synthesis": {
        "rationale": "...",
        "template": {
            "sections": [{"name": "style_foundation", "description": "core style rules", "value": "..."}],
            "negative_prompt": "...",
            "caption_sections": ["Art Style"],
            "caption_length_target": 500,
        },
    },
    "review": {
        "experiment_assessments": ["[EXP 0] SIGNAL - ..."],
        "noise_vs_signal": "...",
        "strategic_guidance": "...",
        "recommended_categories": ["color_palette", "composition"],
    },
    "style_compilation": {
        "style_profile": {
            "color_palette": "...",
            "composition": "...",
            "technique": "...",
            "mood_atmosphere": "...",
            "subject_matter": "...",
            "influences": "...",
        },
        "initial_template": {
            "sections": [{"name": "style_foundation", "description": "core style rules", "value": "..."}],
            "negative_prompt": "...",
            "caption_sections": ["Art Style"],
            "caption_length_target": 500,
        },
    },
}


def schema_hint(name: str) -> str:
    return json.dumps(_SCHEMA_HINTS[name], indent=2)
