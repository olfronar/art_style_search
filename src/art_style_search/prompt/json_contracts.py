"""Validated JSON contracts for reasoning-model exchanges."""

from __future__ import annotations

import json
from typing import Any

from art_style_search.contracts import ExperimentSketch, InitialTemplateSketch, Lessons, RefinementResult
from art_style_search.prompt._parse import validate_template
from art_style_search.types import PromptSection, PromptTemplate, ReviewResult, StyleProfile
from art_style_search.utils import CATEGORY_SYNONYMS


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


def _as_prose_str(data: object, *, label: str, default: str = "") -> str:
    if data is None:
        return default
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        return json.dumps(data, sort_keys=True)
    if isinstance(data, list):
        parts = [_as_str(item, label=f"{label}[{i}]") for i, item in enumerate(data)]
        return "\n".join(part for part in parts if part)
    msg = f"{label} must be a string"
    raise ValueError(msg)


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
    if data is None:
        return []
    if isinstance(data, str):
        value = data.strip()
        return [value] if value else []
    items = _require_list(data, label=label)
    result: list[str] = []
    for i, item in enumerate(items):
        value = _as_str(item, label=f"{label}[{i}]")
        if value:
            result.append(value)
    return result


def _normalize_builds_on(data: object, *, label: str, default: str = "") -> str:
    if data is None:
        return default
    if isinstance(data, str):
        value = data.strip()
        return "" if value.lower() == "none" else value
    if isinstance(data, list):
        parts = [_as_str(item, label=f"{label}[{i}]") for i, item in enumerate(data)]
        normalized = [part for part in parts if part and part.lower() != "none"]
        return ", ".join(normalized)
    msg = f"{label} must be a string"
    raise ValueError(msg)


def _lessons_from_payload(data: object, *, label: str) -> Lessons:
    if data is None:
        return Lessons()
    if isinstance(data, str):
        value = data.strip()
        return Lessons() if not value or value.lower() == "none" else Lessons(new_insight=value)
    if isinstance(data, list):
        combined = _as_prose_str(data, label=label)
        return Lessons() if not combined else Lessons(new_insight=combined)

    lessons_obj = _require_dict(data, label=label)
    return Lessons(
        confirmed=_as_prose_str(lessons_obj.get("confirmed"), label=f"{label}.confirmed"),
        rejected=_as_prose_str(lessons_obj.get("rejected"), label=f"{label}.rejected"),
        new_insight=_as_prose_str(lessons_obj.get("new_insight"), label=f"{label}.new_insight"),
    )


def _normalize_changed_sections(exp: dict[str, Any], *, label: str) -> tuple[str, list[str]]:
    changed_section = _as_str(exp.get("changed_section"), label=f"{label}.changed_section")
    changed_sections = _as_str_list(exp.get("changed_sections") or [], label=f"{label}.changed_sections")

    # changed_sections is the authoritative multi-section field. If repair output
    # leaves the legacy singular field inconsistent, normalize it instead of
    # dropping the whole batch.
    if changed_sections and changed_section and changed_sections[0] != changed_section:
        changed_section = changed_sections[0]
    if not changed_sections and changed_section:
        changed_sections = [changed_section]
    if not changed_section and changed_sections:
        changed_section = changed_sections[0]
    return changed_section, changed_sections


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


def _validate_template_or_raise(template: PromptTemplate, *, label: str) -> PromptTemplate:
    errors = validate_template(template)
    if errors:
        msg = "; ".join(errors)
        raise ValueError(msg)
    return template


def _initial_sketch_from_payload(data: object, *, label: str) -> InitialTemplateSketch:
    obj = _require_dict(data, label=label)
    caption_sections = _as_str_list(obj.get("caption_sections") or [], label=f"{label}.caption_sections")
    return InitialTemplateSketch(
        approach_summary=_as_str(obj.get("approach_summary"), label=f"{label}.approach_summary"),
        emphasis=_as_str(obj.get("emphasis"), label=f"{label}.emphasis"),
        instruction_style=_as_str(obj.get("instruction_style"), label=f"{label}.instruction_style"),
        caption_length_target=_as_int(
            obj.get("caption_length_target"),
            label=f"{label}.caption_length_target",
            default=0,
        ),
        caption_sections=caption_sections,
        distinguishing_feature=_as_str(obj.get("distinguishing_feature"), label=f"{label}.distinguishing_feature"),
    )


def validate_initial_brainstorm_payload(data: object, *, num_sketches: int) -> list[InitialTemplateSketch]:
    obj = _require_dict(data, label="initial_brainstorm_response")
    sketches_raw = _require_list(obj.get("sketches") or [], label="sketches")
    sketches = [
        _initial_sketch_from_payload(item, label=f"sketches[{i}]") for i, item in enumerate(sketches_raw[:num_sketches])
    ]
    if not sketches:
        raise ValueError("sketches must contain at least one entry")
    return sketches


def validate_initial_expansion_payload(data: object) -> PromptTemplate:
    return _validate_template_or_raise(payload_to_template(data, label="initial_expansion_response"), label="template")


def _refinement_result_from_payload(data: object, *, label: str) -> RefinementResult:
    exp = _require_dict(data, label=label)
    template = payload_to_template(exp.get("template") or {}, label=f"{label}.template")
    changed_section, changed_sections = _normalize_changed_sections(exp, label=label)
    return RefinementResult(
        template=template,
        analysis=_as_prose_str(exp.get("analysis"), label=f"{label}.analysis"),
        template_changes=_as_prose_str(
            exp.get("template_changes"),
            label=f"{label}.template_changes",
            default="none",
        ),
        should_stop=False,
        hypothesis=_as_prose_str(exp.get("hypothesis"), label=f"{label}.hypothesis"),
        experiment=_as_prose_str(exp.get("experiment"), label=f"{label}.experiment"),
        lessons=_lessons_from_payload(exp.get("lessons"), label=f"{label}.lessons"),
        builds_on=_normalize_builds_on(exp.get("builds_on"), label=f"{label}.builds_on", default="") or None,
        open_problems=_as_str_list(exp.get("open_problems") or [], label=f"{label}.open_problems"),
        changed_section=changed_section,
        changed_sections=changed_sections,
        target_category=_as_str(
            exp.get("target_category"),
            label=f"{label}.target_category",
        ),
        direction_id=_as_str(exp.get("direction_id"), label=f"{label}.direction_id"),
        direction_summary=_as_prose_str(exp.get("direction_summary"), label=f"{label}.direction_summary"),
        failure_mechanism=_as_prose_str(
            exp.get("failure_mechanism"),
            label=f"{label}.failure_mechanism",
        ),
        intervention_type=_as_str(
            exp.get("intervention_type"),
            label=f"{label}.intervention_type",
        ),
        risk_level=_as_str(exp.get("risk_level"), label=f"{label}.risk_level", default="targeted") or "targeted",
        expected_primary_metric=_as_str(
            exp.get("expected_primary_metric"),
            label=f"{label}.expected_primary_metric",
        ),
        expected_tradeoff=_as_prose_str(
            exp.get("expected_tradeoff"),
            label=f"{label}.expected_tradeoff",
        ),
    )


def validate_brainstorm_payload(data: object, *, num_sketches: int) -> tuple[list[ExperimentSketch], bool]:
    obj = _require_dict(data, label="brainstorm_response")
    sketches_raw = _require_list(obj.get("sketches") or [], label="sketches")
    sketches: list[ExperimentSketch] = []
    for i, item in enumerate(sketches_raw[:num_sketches]):
        sketch = _require_dict(item, label=f"sketches[{i}]")
        sketches.append(
            ExperimentSketch(
                hypothesis=_as_str(sketch.get("hypothesis"), label=f"sketches[{i}].hypothesis"),
                target_category=_as_str(sketch.get("target_category"), label=f"sketches[{i}].target_category"),
                failure_mechanism=_as_str(
                    sketch.get("failure_mechanism"),
                    label=f"sketches[{i}].failure_mechanism",
                ),
                intervention_type=_as_str(
                    sketch.get("intervention_type"),
                    label=f"sketches[{i}].intervention_type",
                ),
                direction_id=_as_str(sketch.get("direction_id"), label=f"sketches[{i}].direction_id"),
                direction_summary=_as_str(
                    sketch.get("direction_summary"),
                    label=f"sketches[{i}].direction_summary",
                ),
                risk_level=_as_str(sketch.get("risk_level"), label=f"sketches[{i}].risk_level", default="targeted")
                or "targeted",
                expected_primary_metric=_as_str(
                    sketch.get("expected_primary_metric"),
                    label=f"sketches[{i}].expected_primary_metric",
                ),
                builds_on=_normalize_builds_on(sketch.get("builds_on"), label=f"sketches[{i}].builds_on", default=""),
            )
        )
    return sketches, bool(obj.get("converged", False))


def validate_ranking_payload(data: object, *, num_sketches: int) -> list[int]:
    if isinstance(data, list):
        ranked_raw = _require_list(data, label="ranking_response")
    else:
        obj = _require_dict(data, label="ranking_response")
        ranked_raw = _require_list(obj.get("ranked_indices") or [], label="ranked_indices")
    ranked: list[int] = []
    seen: set[int] = set()
    for i, item in enumerate(ranked_raw):
        idx = _as_int(item, label=f"ranked_indices[{i}]")
        if idx < 0 or idx >= num_sketches or idx in seen:
            continue
        seen.add(idx)
        ranked.append(idx)
    ranked.extend(idx for idx in range(num_sketches) if idx not in seen)
    return ranked


def validate_expansion_payload(data: object) -> RefinementResult:
    result = _refinement_result_from_payload(data, label="expansion_response")
    _validate_template_or_raise(result.template, label="expansion_response.template")
    return result


def validate_synthesis_payload(data: object) -> tuple[PromptTemplate, str]:
    obj = _require_dict(data, label="synthesis_response")
    template = _validate_template_or_raise(
        payload_to_template(obj.get("template") or {}, label="template"), label="template"
    )
    rationale = _as_str(obj.get("rationale"), label="rationale")
    return template, rationale


def validate_review_payload(data: object) -> ReviewResult:
    obj = _require_dict(data, label="review_response")
    valid_categories = set(CATEGORY_SYNONYMS)
    return ReviewResult(
        experiment_assessments=_as_str_list(obj.get("experiment_assessments") or [], label="experiment_assessments"),
        noise_vs_signal=_as_str(obj.get("noise_vs_signal"), label="noise_vs_signal"),
        strategic_guidance=_as_str(obj.get("strategic_guidance"), label="strategic_guidance"),
        recommended_categories=[
            category
            for category in _as_str_list(obj.get("recommended_categories") or [], label="recommended_categories")
            if category in valid_categories
        ],
    )


def validate_style_compilation_payload(
    data: object,
    *,
    gemini_raw: str,
    reasoning_raw: str,
) -> tuple[StyleProfile, PromptTemplate]:
    obj = _require_dict(data, label="style_compilation_response")
    profile_obj = _require_dict(obj.get("style_profile") or {}, label="style_profile")
    template = _validate_template_or_raise(
        payload_to_template(obj.get("initial_template") or {}, label="initial_template"),
        label="initial_template",
    )
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


def _schema_word_block(token: str, *, words: int = 260) -> str:
    return " ".join([token] * words)


_STYLE_FOUNDATION_DRAWING_PREFIX = (
    "How to Draw: silhouette primitives, construction order (head -> torso -> limbs -> props), "
    "line policy, shading layers (base -> AO -> midtones -> rim -> specular), edge softness, "
    "texture grain, signature quirk. "
)
_SUBJECT_ANCHOR_PROPORTIONS_PREFIX = (
    "Proportions: N.N heads tall, archetype (chibi / stylized-youth / heroic / realistic-adult / "
    "elongated), eye-size bucket, limb-proportion bucket. "
)


def _schema_template_payload() -> dict[str, Any]:
    return {
        "sections": [
            {
                "name": "style_foundation",
                "description": "core style rules",
                "value": _STYLE_FOUNDATION_DRAWING_PREFIX + _schema_word_block("style_dna", words=240),
            },
            {
                "name": "subject_anchor",
                "description": "subject fidelity instructions",
                "value": _SUBJECT_ANCHOR_PROPORTIONS_PREFIX + _schema_word_block("subject_lock", words=240),
            },
            {
                "name": "color_palette",
                "description": "palette guidance",
                "value": _schema_word_block("palette_role"),
            },
            {
                "name": "accent_colors",
                "description": "accent placement guidance",
                "value": _schema_word_block("accent_focus"),
            },
            {
                "name": "lighting",
                "description": "light direction guidance",
                "value": _schema_word_block("lighting_balance"),
            },
            {
                "name": "material_finish",
                "description": "surface treatment guidance",
                "value": _schema_word_block("material_finish"),
            },
            {
                "name": "environment",
                "description": "setting and staging guidance",
                "value": _schema_word_block("environment_stage"),
            },
            {
                "name": "negative_constraints",
                "description": "avoidance guidance",
                "value": _schema_word_block("negative_guard"),
            },
        ],
        "negative_prompt": "photorealism, gritty realism, muddy neutrals, harsh black shadows, noisy microtexture",
        "caption_sections": [
            "Art Style",
            "Subject",
            "Color Palette",
            "Accent Colors",
            "Lighting",
            "Material Finish",
            "Environment",
            "Negative Cues",
        ],
        "caption_length_target": 4000,
    }


def _section_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "description", "value"],
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "value": {"type": "string"},
        },
    }


def _template_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["sections", "negative_prompt", "caption_sections", "caption_length_target"],
        "properties": {
            "sections": {"type": "array", "items": _section_schema()},
            "negative_prompt": {"type": "string"},
            "caption_sections": {"type": "array", "items": {"type": "string"}},
            "caption_length_target": {"type": "integer"},
        },
    }


_SCHEMA_HINTS = {
    "initial_brainstorm": {
        "sketches": [
            {
                "approach_summary": "subject-first strict checklist",
                "emphasis": "technique",
                "instruction_style": "checklist",
                "caption_length_target": 4000,
                "caption_sections": ["Art Style", "Subject", "Color Palette", "Composition"],
                "distinguishing_feature": "Terse imperative checklists; compact sections; subject facets enumerated.",
            }
        ]
    },
    "initial_expansion": _schema_template_payload(),
    "brainstorm": {
        "sketches": [
            {
                "hypothesis": "...",
                "target_category": "subject_anchor",
                "failure_mechanism": "Identity cues are buried behind style language.",
                "intervention_type": "information_priority",
                "direction_id": "D1",
                "direction_summary": "Subject identity lock",
                "risk_level": "targeted",
                "expected_primary_metric": "vision_subject",
                "builds_on": "H3",
            }
        ],
        "converged": False,
    },
    "ranking": {
        "ranked_indices": [2, 7, 0, 5, 1],
    },
    "expansion": {
        "analysis": "...",
        "lessons": {"confirmed": "...", "rejected": "...", "new_insight": "..."},
        "hypothesis": "...",
        "builds_on": "H3",
        "experiment": "...",
        "changed_section": "color_palette",
        "changed_sections": ["color_palette", "lighting"],
        "target_category": "color_palette",
        "direction_id": "D1",
        "direction_summary": "Palette localization",
        "failure_mechanism": "Large frame regions and tiny accents are described with the same priority.",
        "intervention_type": "information_priority",
        "risk_level": "bold",
        "expected_primary_metric": "color_histogram",
        "expected_tradeoff": "Broader edit across two related sections; may reduce attribution precision.",
        "open_problems": ["..."],
        "template_changes": "...",
        "template": _schema_template_payload(),
    },
    "synthesis": {
        "rationale": "...",
        "template": _schema_template_payload(),
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
        "initial_template": _schema_template_payload(),
    },
}


_RESPONSE_SCHEMAS = {
    "initial_brainstorm": {
        "type": "object",
        "additionalProperties": False,
        "required": ["sketches"],
        "properties": {
            "sketches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "approach_summary",
                        "emphasis",
                        "instruction_style",
                        "caption_length_target",
                        "caption_sections",
                        "distinguishing_feature",
                    ],
                    "properties": {
                        "approach_summary": {"type": "string"},
                        "emphasis": {"type": "string"},
                        "instruction_style": {"type": "string"},
                        "caption_length_target": {"type": "integer"},
                        "caption_sections": {"type": "array", "items": {"type": "string"}},
                        "distinguishing_feature": {"type": "string"},
                    },
                },
            }
        },
    },
    "initial_expansion": _template_schema(),
    "brainstorm": {
        "type": "object",
        "additionalProperties": False,
        "required": ["sketches", "converged"],
        "properties": {
            "sketches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "hypothesis",
                        "target_category",
                        "failure_mechanism",
                        "intervention_type",
                        "direction_id",
                        "direction_summary",
                        "risk_level",
                        "expected_primary_metric",
                        "builds_on",
                    ],
                    "properties": {
                        "hypothesis": {"type": "string"},
                        "target_category": {"type": "string"},
                        "failure_mechanism": {"type": "string"},
                        "intervention_type": {"type": "string"},
                        "direction_id": {"type": "string"},
                        "direction_summary": {"type": "string"},
                        "risk_level": {"type": "string"},
                        "expected_primary_metric": {"type": "string"},
                        "builds_on": {"type": "string"},
                    },
                },
            },
            "converged": {"type": "boolean"},
        },
    },
    "ranking": {
        "type": "object",
        "additionalProperties": False,
        "required": ["ranked_indices"],
        "properties": {
            "ranked_indices": {"type": "array", "items": {"type": "integer"}},
        },
    },
    "expansion": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "analysis",
            "lessons",
            "hypothesis",
            "builds_on",
            "experiment",
            "changed_section",
            "changed_sections",
            "target_category",
            "direction_id",
            "direction_summary",
            "failure_mechanism",
            "intervention_type",
            "risk_level",
            "expected_primary_metric",
            "expected_tradeoff",
            "open_problems",
            "template_changes",
            "template",
        ],
        "properties": {
            "analysis": {"type": "string"},
            "lessons": {
                "type": "object",
                "additionalProperties": False,
                "required": ["confirmed", "rejected", "new_insight"],
                "properties": {
                    "confirmed": {"type": "string"},
                    "rejected": {"type": "string"},
                    "new_insight": {"type": "string"},
                },
            },
            "hypothesis": {"type": "string"},
            "builds_on": {"type": "string"},
            "experiment": {"type": "string"},
            "changed_section": {"type": "string"},
            "changed_sections": {"type": "array", "items": {"type": "string"}},
            "target_category": {"type": "string"},
            "direction_id": {"type": "string"},
            "direction_summary": {"type": "string"},
            "failure_mechanism": {"type": "string"},
            "intervention_type": {"type": "string"},
            "risk_level": {"type": "string"},
            "expected_primary_metric": {"type": "string"},
            "expected_tradeoff": {"type": "string"},
            "open_problems": {"type": "array", "items": {"type": "string"}},
            "template_changes": {"type": "string"},
            "template": _template_schema(),
        },
    },
    "synthesis": {
        "type": "object",
        "additionalProperties": False,
        "required": ["rationale", "template"],
        "properties": {
            "rationale": {"type": "string"},
            "template": _template_schema(),
        },
    },
    "review": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "experiment_assessments",
            "noise_vs_signal",
            "strategic_guidance",
            "recommended_categories",
        ],
        "properties": {
            "experiment_assessments": {"type": "array", "items": {"type": "string"}},
            "noise_vs_signal": {"type": "string"},
            "strategic_guidance": {"type": "string"},
            "recommended_categories": {"type": "array", "items": {"type": "string"}},
        },
    },
    "style_compilation": {
        "type": "object",
        "additionalProperties": False,
        "required": ["style_profile", "initial_template"],
        "properties": {
            "style_profile": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "color_palette",
                    "composition",
                    "technique",
                    "mood_atmosphere",
                    "subject_matter",
                    "influences",
                ],
                "properties": {
                    "color_palette": {"type": "string"},
                    "composition": {"type": "string"},
                    "technique": {"type": "string"},
                    "mood_atmosphere": {"type": "string"},
                    "subject_matter": {"type": "string"},
                    "influences": {"type": "string"},
                },
            },
            "initial_template": _template_schema(),
        },
    },
}


def schema_hint(name: str) -> str:
    return json.dumps(_SCHEMA_HINTS[name], indent=2)


def response_schema(name: str) -> dict[str, Any]:
    return json.loads(json.dumps(_RESPONSE_SCHEMAS[name]))
