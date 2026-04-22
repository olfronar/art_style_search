"""Unit tests for parsing and formatting helpers in art_style_search.prompt."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from art_style_search import contracts
from art_style_search.prompt import (
    Lessons,
    RefinementResult,
    _format_metrics,
    _format_template,
    propose_experiments,
    propose_initial_templates,
    rank_experiment_sketches,
    rank_initial_sketches,
    synthesize_templates,
    validate_template,
)
from art_style_search.prompt.json_contracts import (
    schema_hint,
    template_to_payload,
    validate_brainstorm_payload,
    validate_expansion_payload,
    validate_initial_brainstorm_payload,
    validate_initial_expansion_payload,
    validate_ranking_payload,
    validate_review_payload,
    validate_style_compilation_payload,
    validate_synthesis_payload,
)
from art_style_search.types import (
    AggregatedMetrics,
    Caption,
    KnowledgeBase,
    MetricScores,
    PromptSection,
    PromptTemplate,
)
from tests.conftest import make_style_profile


class TestPromptContracts:
    def test_prompt_reexports_contract_types_from_neutral_contracts_module(self) -> None:
        assert Lessons is contracts.Lessons
        assert RefinementResult is contracts.RefinementResult


# ---------------------------------------------------------------------------
# _format_template
# ---------------------------------------------------------------------------


class TestFormatTemplate:
    def test_output_contains_sections_and_negative(self) -> None:
        """The XML context string surfaces every section plus the negative prompt."""
        tpl = PromptTemplate(
            sections=[
                PromptSection(name="style", description="overall style", value="impressionist painting"),
                PromptSection(name="color", description="color palette", value="warm earth tones"),
            ],
            negative_prompt="photorealistic, 3d render",
        )
        xml = _format_template(tpl)
        assert xml.startswith("<template>")
        assert xml.endswith("</template>")
        for section in tpl.sections:
            assert f'name="{section.name}"' in xml
            assert f'description="{section.description}"' in xml
            assert section.value in xml
        assert "<negative>photorealistic, 3d render</negative>" in xml

    def test_no_negative_prompt(self) -> None:
        tpl = PromptTemplate(
            sections=[PromptSection(name="style", description="style", value="watercolor")],
            negative_prompt=None,
        )
        xml = _format_template(tpl)
        assert "<negative>" not in xml


# ---------------------------------------------------------------------------
# _format_metrics
# ---------------------------------------------------------------------------


class TestFormatMetrics:
    def test_all_metrics_present(self) -> None:
        metrics = AggregatedMetrics(
            dreamsim_similarity_mean=0.7234,
            dreamsim_similarity_std=0.0123,
            hps_score_mean=0.2789,
            hps_score_std=0.0045,
            aesthetics_score_mean=6.1234,
            aesthetics_score_std=0.5678,
        )
        output = _format_metrics(metrics)
        # Every key from summary_dict should appear
        for key in metrics.summary_dict():
            assert key in output
        # Values should be formatted to 4 decimal places
        assert "0.7234" in output
        assert "0.0123" in output

    def test_format_uses_dash_prefix(self) -> None:
        metrics = AggregatedMetrics(
            dreamsim_similarity_mean=0.5,
            dreamsim_similarity_std=0.1,
            hps_score_mean=0.2,
            hps_score_std=0.01,
            aesthetics_score_mean=5.0,
            aesthetics_score_std=0.5,
        )
        output = _format_metrics(metrics)
        lines = output.strip().split("\n")
        assert len(lines) == len(metrics.summary_dict())
        for line in lines:
            assert line.startswith("- ")


# ---------------------------------------------------------------------------
# validate_template
# ---------------------------------------------------------------------------


def _make_valid_template() -> PromptTemplate:
    """Build a minimal valid template for validation tests."""
    sections = [
        PromptSection(
            name="style_foundation",
            description="Style rules",
            value=(
                "How to Draw: silhouette primitives, construction order, line policy, shading layers, signature quirk. "
            )
            + "Foundation rules. " * 125,
        ),
        PromptSection(
            name="subject_anchor",
            description="Subject rules",
            value=(
                "Proportions: 3.2 heads tall, chibi archetype, stubby limbs. "
                "Distinguishing Features: species, hair/fur, markings, apparel, props. "
            )
            + "Subject rules. " * 125,
        ),
        PromptSection(name="color_palette", description="Colors", value="Color rules. " * 130),
        PromptSection(name="composition", description="Layout", value="Comp rules. " * 130),
        PromptSection(name="technique", description="Technique", value="Tech rules. " * 130),
        PromptSection(name="lighting", description="Lighting", value="Lighting rules. " * 130),
        PromptSection(name="environment", description="Environment", value="Environment rules. " * 130),
        PromptSection(name="textures", description="Textures", value="Texture rules. " * 130),
    ]
    return PromptTemplate(
        sections=sections,
        caption_sections=["Art Style", "Subject", "Color Palette", "Composition", "Technique"],
        caption_length_target=500,
    )


class TestValidateTemplate:
    def test_valid_template_passes(self) -> None:
        assert validate_template(_make_valid_template()) == []

    def test_missing_style_foundation_first(self) -> None:
        t = _make_valid_template()
        t.sections[0], t.sections[1] = t.sections[1], t.sections[0]
        errors = validate_template(t)
        assert any("style_foundation" in error for error in errors)

    def test_missing_subject_anchor_second(self) -> None:
        t = _make_valid_template()
        t.sections[1], t.sections[2] = t.sections[2], t.sections[1]
        errors = validate_template(t)
        assert len(errors) == 1
        assert "subject_anchor" in errors[0]

    def test_missing_art_style_first_caption(self) -> None:
        t = _make_valid_template()
        t.caption_sections = ["Color Palette", "Art Style", "Subject"]
        errors = validate_template(t)
        assert any("Art Style" in error for error in errors)

    def test_missing_subject_second_caption(self) -> None:
        t = _make_valid_template()
        t.caption_sections = ["Art Style", "Color Palette", "Subject", "Composition"]
        errors = validate_template(t)
        assert len(errors) == 1
        assert "Subject" in errors[0]

    def test_too_few_sections(self) -> None:
        t = _make_valid_template()
        t.sections = t.sections[:4]
        errors = validate_template(t)
        assert any("Section count" in e for e in errors)

    def test_caption_length_out_of_bounds(self) -> None:
        t = _make_valid_template()
        t.caption_length_target = 499
        errors = validate_template(t)
        assert any("Caption length" in e for e in errors)

    def test_large_caption_length_target_is_allowed(self) -> None:
        t = _make_valid_template()
        t.caption_length_target = 6000
        errors = validate_template(t)
        assert not any("Caption length" in e for e in errors)

    def test_rendered_prompt_word_count_out_of_bounds(self) -> None:
        t = _make_valid_template()
        for section in t.sections:
            section.value = "tiny"
        errors = validate_template(t)
        assert any("Rendered prompt word count" in e for e in errors)

    def test_changed_section_not_in_template(self) -> None:
        t = _make_valid_template()
        errors = validate_template(t, changed_section="nonexistent_section")
        assert len(errors) == 1
        assert "nonexistent_section" in errors[0]

    def test_changed_section_valid(self) -> None:
        t = _make_valid_template()
        assert validate_template(t, changed_section="color_palette") == []

    def test_changed_section_allows_caption_structure_fields(self) -> None:
        t = _make_valid_template()
        assert (
            validate_template(
                t,
                changed_section="caption_sections",
                changed_sections=["caption_sections"],
            )
            == []
        )

    def test_changed_sections_allow_removed_incumbent_section_when_reference_template_provided(self) -> None:
        current = _make_valid_template()
        current.sections[2] = PromptSection(name="face_hands_pose", description="anatomy", value="Pose rules. " * 80)

        proposed = _make_valid_template()
        proposed.sections[2] = PromptSection(
            name="scene_type_and_asset_class",
            description="scene taxonomy",
            value="Scene taxonomy rules. " * 80,
        )

        assert (
            validate_template(
                proposed,
                changed_section="face_hands_pose",
                changed_sections=["face_hands_pose", "caption_sections"],
                risk_level="bold",
                reference_template=current,
            )
            == []
        )

    def test_targeted_change_rejects_multiple_changed_sections(self) -> None:
        t = _make_valid_template()
        errors = validate_template(
            t,
            changed_section="color_palette",
            changed_sections=["color_palette", "composition"],
            risk_level="targeted",
        )
        assert any("targeted" in error for error in errors)

    def test_bold_change_allows_up_to_three_changed_sections(self) -> None:
        t = _make_valid_template()
        assert (
            validate_template(
                t,
                changed_section="color_palette",
                changed_sections=["color_palette", "composition", "technique"],
                risk_level="bold",
            )
            == []
        )

    def test_bold_change_rejects_more_than_three_changed_sections(self) -> None:
        t = _make_valid_template()
        errors = validate_template(
            t,
            changed_section="color_palette",
            changed_sections=["color_palette", "composition", "technique", "subject_anchor"],
            risk_level="bold",
        )
        assert any("up to 3" in error for error in errors)

    def test_zero_caption_length_allowed(self) -> None:
        t = _make_valid_template()
        t.caption_length_target = 0
        assert validate_template(t) == []

    def test_rejects_canon_shaped_as_methodology(self) -> None:
        t = _make_valid_template()
        t.sections[0].value = (
            "How to Draw: describe the medium in plain vocabulary.\n"
            "\n"
            "SLOT 1 — How to Draw (medium + construction + line policy):\n"
            "- [ ] Declare the medium in plain observable vocabulary.\n"
            "- [ ] Construction order: silhouette → forms → details.\n"
            "\n"
            "Write the [Art Style] block as a dense, compact ruleset.\n"
            "Target 400-800 words across the 5 slots.\n"
            "Begin the block with the Medium reminder line.\n"
        ) + "Filler rules. " * 120
        errors = validate_template(t)
        assert any("methodology" in e for e in errors), errors

    def test_rejects_canon_addressed_to_captioner(self) -> None:
        """Captioner-addressing preambles ('Begin the caption with...', 'This block is...')
        are the dominant drift shape observed post-9277be7. The tightened validator must
        reject them even without the older SLOT/MANDATORY markers."""
        t = _make_valid_template()
        t.sections[0].value = (
            "How to Draw: the style renders as lit 3D volumes with beveled edges. "
            "Begin the caption with an [Art Style] block 400-800 words long. "
            "This block is the reusable style DNA the captioner must copy verbatim. "
            "Each slot is a dense paragraph of rules that applies to every image.\n"
        ) + "Filler rules. " * 120
        errors = validate_template(t)
        assert any("methodology" in e for e in errors), errors

    def test_rejects_canon_referencing_the_captioner_directly(self) -> None:
        t = _make_valid_template()
        t.sections[0].value = (
            "How to Draw: style renders as soft-beveled volumes with chromatic shadows. "
            "The captioner must paraphrase these assertions into the [Art Style] block verbatim."
        ) + "Filler rules. " * 120
        errors = validate_template(t)
        assert any("methodology" in e for e in errors), errors

    def test_rejects_canon_with_reusable_dna_phrase(self) -> None:
        t = _make_valid_template()
        t.sections[0].value = (
            "How to Draw: style renders as soft volumes.\nThis is the REUSABLE style DNA repeated across captions.\n"
        ) + "Filler rules. " * 120
        errors = validate_template(t)
        assert any("methodology" in e for e in errors), errors

    def test_accepts_canon_of_concrete_style_assertions(self) -> None:
        t = _make_valid_template()
        t.sections[0].value = (
            "How to Draw: lineless 2D digital illustration mimicking stylized PBR 3D rendering. "
            "Construction: silhouette primitives (spheres, capsules) merged into beveled volumes. "
            "Line policy: absolute zero linework; separation via value, hue, and occlusion.\n"
            "Shading & Light: saturated flat albedo, tight AO crevices, broad feathered midtones, "
            "soft specular bloom, crisp cool rim light opposite a warm key.\n"
            "Color Principle: high-key candy-spectrum palette; complementary/triadic anchoring; "
            "shadows hue-shift cooler at equal-or-higher chroma, never desaturated.\n"
            "Surface & Texture: zero grain of any kind; uniform matte-fondant / polished-clay "
            "material vocabulary; every edge beveled and rounded.\n"
            "Style Invariants: MUST bevel every edge; MUST hue-shift shadows cooler; NEVER outline "
            "or cel-band any form; NEVER pure black or white outside pupils and pinpoint speculars.\n"
        ) + "Further concrete style assertions. " * 60
        assert validate_template(t) == []


class TestJsonContracts:
    def test_brainstorm_payload_reads_sketches_and_converged_flag(self) -> None:
        payload = {
            "sketches": [
                {
                    "hypothesis": "Promote subject identity to a harder requirement.",
                    "target_category": "subject_anchor",
                    "failure_mechanism": "Identity details are buried behind style language.",
                    "intervention_type": "information_priority",
                    "direction_id": "D1",
                    "direction_summary": "Subject identity lock",
                    "risk_level": "targeted",
                    "expected_primary_metric": "vision_subject",
                    "builds_on": "H4",
                }
            ],
            "converged": True,
        }

        sketches, converged = validate_brainstorm_payload(payload, num_sketches=4)

        assert converged is True
        assert len(sketches) == 1
        assert isinstance(sketches[0], contracts.ExperimentSketch)
        assert sketches[0].direction_id == "D1"
        assert sketches[0].builds_on == "H4"

    def test_ranking_payload_keeps_unique_in_range_indices_then_appends_unranked(self) -> None:
        ranked = validate_ranking_payload({"ranked_indices": [2, 2, 99, 0]}, num_sketches=4)

        assert ranked == [2, 0, 1, 3]

    def test_ranking_payload_accepts_top_level_array(self) -> None:
        ranked = validate_ranking_payload([1, 3, 1], num_sketches=4)

        assert ranked == [1, 3, 0, 2]

    def test_brainstorm_payload_normalizes_nullable_and_list_builds_on(self) -> None:
        payload = {
            "sketches": [
                {
                    "hypothesis": "First",
                    "target_category": "subject_anchor",
                    "failure_mechanism": "Identity details are buried.",
                    "intervention_type": "information_priority",
                    "direction_id": "D1",
                    "direction_summary": "Subject identity lock",
                    "risk_level": "targeted",
                    "expected_primary_metric": "vision_subject",
                    "builds_on": None,
                },
                {
                    "hypothesis": "Second",
                    "target_category": "composition",
                    "failure_mechanism": "Layout cues are too diffuse.",
                    "intervention_type": "section_schema",
                    "direction_id": "D2",
                    "direction_summary": "Composition lock",
                    "risk_level": "bold",
                    "expected_primary_metric": "vision_composition",
                    "builds_on": "none",
                },
                {
                    "hypothesis": "Third",
                    "target_category": "lighting",
                    "failure_mechanism": "Lighting cues drift.",
                    "intervention_type": "negative_constraints",
                    "direction_id": "D3",
                    "direction_summary": "Lighting lock",
                    "risk_level": "targeted",
                    "expected_primary_metric": "vision_style",
                    "builds_on": ["H3", "H5"],
                },
            ]
        }

        sketches, converged = validate_brainstorm_payload(payload, num_sketches=3)

        assert converged is False
        assert [sketch.builds_on for sketch in sketches] == ["", "", "H3, H5"]

    def test_expansion_payload_normalizes_nullable_lessons_and_list_analysis(self) -> None:
        valid_template = _make_valid_template()
        payload = {
            "analysis": ["Line 1", "Line 2"],
            "lessons": None,
            "hypothesis": "Strengthen subject identity anchors.",
            "builds_on": None,
            "experiment": "Rewrite the subject section to prioritize identity anchors.",
            "changed_section": "subject_anchor",
            "changed_sections": ["subject_anchor"],
            "target_category": "subject_anchor",
            "direction_id": "D1",
            "direction_summary": "Subject identity lock",
            "failure_mechanism": "Identity traits are underspecified.",
            "intervention_type": "information_priority",
            "risk_level": "targeted",
            "expected_primary_metric": "vision_subject",
            "expected_tradeoff": "May over-constrain stylization language.",
            "open_problems": ["Identity drift on crowded scenes"],
            "template_changes": "Reinforce the subject block with required identity facets.",
            "template": {
                "sections": [
                    {"name": section.name, "description": section.description, "value": section.value}
                    for section in valid_template.sections
                ],
                "negative_prompt": "avoid blur",
                "caption_sections": list(valid_template.caption_sections),
                "caption_length_target": 500,
            },
        }

        result = validate_expansion_payload(payload)

        assert result.analysis == "Line 1\nLine 2"
        assert result.lessons == Lessons()

    def test_expansion_payload_normalizes_string_or_list_lessons_into_new_insight(self) -> None:
        valid_template = _make_valid_template()
        payload = {
            "analysis": "Need a stronger subject block.",
            "lessons": ["Identity details drift.", "Props disappear."],
            "hypothesis": "Strengthen subject identity anchors.",
            "builds_on": "none",
            "experiment": "Rewrite the subject section to prioritize identity anchors.",
            "changed_section": "subject_anchor",
            "changed_sections": ["subject_anchor"],
            "target_category": "subject_anchor",
            "direction_id": "D1",
            "direction_summary": "Subject identity lock",
            "failure_mechanism": "Identity traits are underspecified.",
            "intervention_type": "information_priority",
            "risk_level": "targeted",
            "expected_primary_metric": "vision_subject",
            "expected_tradeoff": "May over-constrain stylization language.",
            "open_problems": ["Identity drift on crowded scenes"],
            "template_changes": "Reinforce the subject block with required identity facets.",
            "template": {
                "sections": [
                    {"name": section.name, "description": section.description, "value": section.value}
                    for section in valid_template.sections
                ],
                "negative_prompt": "avoid blur",
                "caption_sections": list(valid_template.caption_sections),
                "caption_length_target": 500,
            },
        }

        result = validate_expansion_payload(payload)

        assert result.builds_on is None
        assert result.lessons.new_insight == "Identity details drift.\nProps disappear."

    def test_expansion_payload_reads_single_refinement_result(self) -> None:
        valid_template = _make_valid_template()
        payload = {
            "analysis": "Need a stronger subject block.",
            "lessons": {"confirmed": "", "rejected": "", "new_insight": "Identity details drift."},
            "hypothesis": "Strengthen subject identity anchors.",
            "builds_on": "H2",
            "experiment": "Rewrite the subject section to prioritize identity anchors.",
            "changed_section": "subject_anchor",
            "changed_sections": ["subject_anchor"],
            "target_category": "subject_anchor",
            "direction_id": "D1",
            "direction_summary": "Subject identity lock",
            "failure_mechanism": "Identity traits are underspecified.",
            "intervention_type": "information_priority",
            "risk_level": "targeted",
            "expected_primary_metric": "vision_subject",
            "expected_tradeoff": "May over-constrain stylization language.",
            "open_problems": ["Identity drift on crowded scenes"],
            "template_changes": "Reinforce the subject block with required identity facets.",
            "template": {
                "sections": [
                    {"name": section.name, "description": section.description, "value": section.value}
                    for section in valid_template.sections
                ],
                "negative_prompt": "avoid blur",
                "caption_sections": list(valid_template.caption_sections),
                "caption_length_target": 500,
            },
        }

        result = validate_expansion_payload(payload)

        assert result.hypothesis == "Strengthen subject identity anchors."
        assert result.changed_sections == ["subject_anchor"]
        assert result.expected_primary_metric == "vision_subject"

    def test_expansion_payload_normalizes_live_openai_shape(self) -> None:
        valid_template = _make_valid_template()
        payload = {
            "analysis": "Current metrics show a strong style-versus-subject imbalance.",
            "lessons": {"confirmed": "", "rejected": "", "new_insight": "Ordering pressure dominates."},
            "hypothesis": "Compress the Art Style anchor.",
            "builds_on": "H1, H2",
            "experiment": "Compress style_foundation only.",
            "changed_section": "style_foundation",
            "changed_sections": ["style_foundation"],
            "target_category": "caption_structure",
            "direction_id": "D1",
            "direction_summary": "Subject signal recovery by reducing style-token dominance",
            "failure_mechanism": "Early style boilerplate monopolizes attention.",
            "intervention_type": "information_priority",
            "risk_level": "targeted",
            "expected_primary_metric": "vision_subject",
            "expected_tradeoff": "May slightly reduce style redundancy.",
            "open_problems": "Background-only references may still underperform.",
            "template_changes": {
                "style_foundation": {
                    "change": "Replace the long block with a compact shared DNA preamble.",
                    "why": "Reduce early token competition.",
                }
            },
            "template": template_to_payload(valid_template),
        }

        result = validate_expansion_payload(payload)

        assert result.open_problems == ["Background-only references may still underperform."]
        assert result.template.sections[0].name == "style_foundation"
        assert result.template.caption_sections[:2] == ["Art Style", "Subject"]
        assert "compact shared DNA preamble" in result.template_changes

    def test_initial_brainstorm_payload_parses_sketches(self) -> None:
        payload = {
            "sketches": [
                {
                    "approach_summary": "subject-first strict checklist",
                    "emphasis": "technique",
                    "instruction_style": "checklist",
                    "caption_length_target": 500,
                    "caption_sections": ["Art Style", "Subject", "Color Palette"],
                    "distinguishing_feature": "Numbered subject facets force identity specificity.",
                },
                {
                    "approach_summary": "mood-led artistic direction",
                    "emphasis": "mood",
                    "instruction_style": "artistic_direction",
                    "caption_length_target": 700,
                    "caption_sections": ["Art Style", "Subject", "Mood", "Composition"],
                    "distinguishing_feature": "Atmospheric language first; lets composition follow mood.",
                },
            ]
        }
        sketches = validate_initial_brainstorm_payload(payload, num_sketches=2)
        assert len(sketches) == 2
        assert sketches[0].approach_summary == "subject-first strict checklist"
        assert sketches[1].caption_length_target == 700
        assert sketches[0].caption_sections[:2] == ["Art Style", "Subject"]

    def test_synthesis_payload_accepts_negative_prompt_key(self) -> None:
        template, rationale = validate_synthesis_payload(
            {
                "rationale": "Take the color section from experiment 1",
                "template": {
                    "sections": [
                        {"name": section.name, "description": section.description, "value": section.value}
                        for section in _make_valid_template().sections
                    ],
                    "negative_prompt": "avoid blur",
                    "caption_sections": list(_make_valid_template().caption_sections),
                    "caption_length_target": 500,
                },
            }
        )
        assert rationale.startswith("Take the color")
        assert template.negative_prompt == "avoid blur"

    def test_review_payload_keeps_recommended_categories(self) -> None:
        review = validate_review_payload(
            {
                "experiment_assessments": ["[EXP 0] SIGNAL - palette improved"],
                "noise_vs_signal": "DreamSim and color moved together.",
                "strategic_guidance": "Keep pushing palette precision.",
                "recommended_categories": ["color_palette", "composition"],
            }
        )
        assert review.experiment_assessments == ["[EXP 0] SIGNAL - palette improved"]
        assert review.recommended_categories == ["color_palette", "composition"]

    def test_review_payload_filters_unknown_recommended_categories(self) -> None:
        review = validate_review_payload(
            {
                "experiment_assessments": ["[EXP 0] SIGNAL - palette improved"],
                "noise_vs_signal": "DreamSim and color moved together.",
                "strategic_guidance": "Keep pushing palette precision.",
                "recommended_categories": ["color_palette", "made_up_category"],
            }
        )

        assert review.recommended_categories == ["color_palette"]

    def test_style_compilation_payload_rejects_invalid_caption_sections(self) -> None:
        payload = {
            "style_profile": {
                "color_palette": "Muted earth tones.",
                "composition": "Low horizon.",
                "technique": "Wet-on-wet watercolor.",
                "mood_atmosphere": "Quiet and contemplative.",
                "subject_matter": "Rural landscapes.",
                "influences": "Turner and Wyeth.",
            },
            "initial_template": {
                "sections": [
                    {"name": "style_foundation", "description": "rules", "value": "Shared rules"},
                    {"name": "color_palette", "description": "colors", "value": "Palette guidance"},
                    {"name": "composition", "description": "layout", "value": "Composition guidance"},
                    {"name": "technique", "description": "medium", "value": "Technique guidance"},
                ],
                "negative_prompt": "avoid blur",
                "caption_sections": ["Art Style Overview", "Color Palette"],
                "caption_length_target": 500,
            },
        }

        with pytest.raises(ValueError, match="First caption section must be 'Art Style'"):
            validate_style_compilation_payload(
                payload,
                gemini_raw="visual analysis",
                reasoning_raw="reasoning analysis",
            )

    def test_initial_expansion_payload_validates_template_shape(self) -> None:
        valid_template = _make_valid_template()
        payload = {
            "sections": [
                {"name": section.name, "description": section.description, "value": section.value}
                for section in valid_template.sections
            ],
            "negative_prompt": "avoid blur",
            "caption_sections": list(valid_template.caption_sections),
            "caption_length_target": valid_template.caption_length_target,
        }

        template = validate_initial_expansion_payload(payload)

        assert template.sections[0].name == "style_foundation"
        assert template.sections[1].name == "subject_anchor"

    def test_initial_expansion_payload_rejects_under_length_template(self) -> None:
        # Two-section payload renders to far under 1200 words.
        payload = {
            "sections": [
                {"name": "style_foundation", "description": "rules", "value": "Shared rules"},
                {"name": "subject_anchor", "description": "subject", "value": "Subject guidance"},
            ],
            "negative_prompt": "avoid blur",
            "caption_sections": ["Art Style", "Subject"],
            "caption_length_target": 500,
        }

        with pytest.raises(ValueError, match="outside bounds"):
            validate_initial_expansion_payload(payload)

    def test_expansion_payload_rejects_invalid_template_shape(self) -> None:
        payload = {
            "analysis": "Need a stronger subject block.",
            "lessons": {"confirmed": "", "rejected": "", "new_insight": "Identity details drift."},
            "hypothesis": "Strengthen subject identity anchors.",
            "builds_on": "H2",
            "experiment": "Rewrite the subject section to prioritize identity anchors.",
            "changed_section": "subject_anchor",
            "changed_sections": ["subject_anchor"],
            "target_category": "subject_anchor",
            "direction_id": "D1",
            "direction_summary": "Subject identity lock",
            "failure_mechanism": "Identity traits are underspecified.",
            "intervention_type": "information_priority",
            "risk_level": "targeted",
            "expected_primary_metric": "vision_subject",
            "expected_tradeoff": "May over-constrain stylization language.",
            "open_problems": ["Identity drift on crowded scenes"],
            "template_changes": "Reinforce the subject block with required identity facets.",
            "template": {
                "sections": [
                    {"name": "style_foundation", "description": "rules", "value": "Shared rules"},
                    {"name": "subject_anchor", "description": "subject rules", "value": "Subject guidance"},
                    {"name": "composition", "description": "layout", "value": "Layout guidance"},
                    {"name": "technique", "description": "medium", "value": "Technique guidance"},
                ],
                "negative_prompt": "avoid blur",
                "caption_sections": ["Art Style", "Subject", "Composition"],
                "caption_length_target": 500,
            },
        }

        with pytest.raises(ValueError, match="Section count"):
            validate_expansion_payload(payload)

    def test_synthesis_payload_rejects_invalid_template_shape(self) -> None:
        with pytest.raises(ValueError, match="Second section must be 'subject_anchor'"):
            validate_synthesis_payload(
                {
                    "rationale": "Take the color section from experiment 1",
                    "template": {
                        "sections": [
                            {"name": "style_foundation", "description": "rules", "value": "Shared rules"},
                            {"name": "color_palette", "description": "colors", "value": "Palette rules"},
                            {"name": "composition", "description": "layout", "value": "Layout rules"},
                            {"name": "technique", "description": "medium", "value": "Technique rules"},
                            {"name": "lighting", "description": "light", "value": "Lighting rules"},
                            {"name": "environment", "description": "env", "value": "Environment rules"},
                            {"name": "texture", "description": "texture", "value": "Texture rules"},
                            {"name": "mood", "description": "mood", "value": "Mood rules"},
                        ],
                        "negative_prompt": "avoid blur",
                        "caption_sections": ["Art Style", "Color Palette"],
                        "caption_length_target": 500,
                    },
                }
            )

    def test_schema_hints_use_subject_in_synthesis_and_show_nontrivial_templates(self) -> None:
        initial_brainstorm_hint = schema_hint("initial_brainstorm")
        initial_expansion_hint = schema_hint("initial_expansion")
        synthesis_hint = schema_hint("synthesis")
        expansion_hint = schema_hint("expansion")

        assert '"Subject"' in synthesis_hint
        assert '"Art Style"' in initial_brainstorm_hint
        assert '"Subject"' in initial_brainstorm_hint
        assert initial_expansion_hint.count('"name"') >= 4
        assert expansion_hint.count('"name"') >= 4

    def test_schema_hint_examples_satisfy_template_validators(self) -> None:
        initial_template = validate_initial_expansion_payload(json.loads(schema_hint("initial_expansion")))
        expansion_result = validate_expansion_payload(json.loads(schema_hint("expansion")))
        synthesis_template, _ = validate_synthesis_payload(json.loads(schema_hint("synthesis")))

        assert len(initial_template.sections) >= 8
        assert len(expansion_result.template.sections) >= 8
        assert len(synthesis_template.sections) >= 8


class TestProposeExperiments:
    @pytest.mark.asyncio
    async def test_uses_two_repair_retries_and_concrete_changed_section_instructions(self) -> None:
        captured: dict[str, Any] = {}

        class FakeClient:
            async def call_json(self, **kwargs):
                captured.update(kwargs)
                return ([], False)

        await propose_experiments(
            make_style_profile(),
            _make_valid_template(),
            KnowledgeBase(),
            None,
            None,
            client=FakeClient(),  # type: ignore[arg-type]
            model="fake-model",
            num_experiments=9,
        )

        assert captured["repair_retries"] == 2
        system = captured["system"]  # type: ignore[assignment]
        assert "changed_sections must use concrete template section names" in system
        assert "caption_sections" in system
        assert "caption_length_target" in system


class TestPromptSurfaceExamples:
    @pytest.mark.asyncio
    async def test_initial_prompt_exposes_expansion_schema(self) -> None:
        expand_calls: list[dict[str, object]] = []

        sketch = contracts.InitialTemplateSketch(
            approach_summary="subject-first strict checklist",
            emphasis="technique",
            instruction_style="checklist",
            caption_length_target=500,
            caption_sections=["Art Style", "Subject", "Color Palette"],
            distinguishing_feature="Numbered subject facets force identity specificity.",
        )

        class FakeClient:
            async def call_json(self, **kwargs):
                response_name = kwargs.get("response_name", "")
                if response_name == "initial_brainstorm":
                    return [sketch, sketch]
                if response_name == "initial_ranking":
                    return [0, 1]
                if isinstance(response_name, str) and response_name.startswith("initial_expansion"):
                    expand_calls.append(kwargs)
                    return _make_valid_template()
                raise AssertionError(f"unexpected response_name: {response_name}")

        templates = await propose_initial_templates(
            make_style_profile(),
            1,
            client=FakeClient(),  # type: ignore[arg-type]
            model="fake-model",
        )

        assert len(templates) == 1
        assert len(expand_calls) == 1
        expand_schema = expand_calls[0]["schema_hint"]
        assert isinstance(expand_schema, str)
        assert expand_schema.count('"name":') >= 4
        assert '"caption_sections": [\n    "Art Style",\n    "Subject"' in expand_schema or (
            '"caption_sections": ["Art Style", "Subject"' in expand_schema
        )
        assert expand_calls[0]["max_tokens"] == 24000

    @pytest.mark.asyncio
    async def test_synthesis_prompt_example_keeps_subject_anchor_in_caption_sections(self) -> None:
        captured: dict[str, Any] = {}

        class FakeClient:
            async def call_json(self, **kwargs):
                captured.update(kwargs)
                return _make_valid_template(), "because"

        exp = type(
            "Result",
            (),
            {
                "branch_id": 0,
                "kept": True,
                "hypothesis": "test",
                "aggregated": AggregatedMetrics(
                    dreamsim_similarity_mean=0.7,
                    dreamsim_similarity_std=0.01,
                    hps_score_mean=0.25,
                    hps_score_std=0.01,
                    aesthetics_score_mean=6.0,
                    aesthetics_score_std=0.2,
                ),
                "template": _make_valid_template(),
            },
        )()

        await synthesize_templates(
            [exp],  # type: ignore[list-item]
            make_style_profile(),
            client=FakeClient(),  # type: ignore[arg-type]
            model="fake-model",
        )

        system = captured["system"]  # type: ignore[assignment]
        assert '"caption_sections": ["Art Style", "Subject"' in system
        assert "2000-8000 words" in system

    @pytest.mark.asyncio
    async def test_experiment_expansion_uses_large_template_budget(self) -> None:
        from art_style_search.prompt.experiments import expand_experiment_sketches

        captured: dict[str, Any] = {}
        sketch = contracts.ExperimentSketch(
            hypothesis="Subject identity is too diffuse.",
            target_category="subject_anchor",
            failure_mechanism="Identity details are buried under style language.",
            intervention_type="information_priority",
            direction_id="D1",
            direction_summary="Subject identity lock",
            risk_level="targeted",
            expected_primary_metric="vision_subject",
            builds_on="H3",
        )

        class FakeClient:
            async def call_json(self, **kwargs):
                captured.update(kwargs)
                return RefinementResult(
                    template=_make_valid_template(),
                    analysis="Need a stronger subject block.",
                    template_changes="Strengthen subject_anchor.",
                    should_stop=False,
                    hypothesis=sketch.hypothesis,
                    experiment="Rewrite the subject block to front-load identity facets.",
                    lessons=Lessons(),
                    builds_on=sketch.builds_on,
                    open_problems=[],
                    changed_section="subject_anchor",
                    changed_sections=["subject_anchor"],
                    target_category=sketch.target_category,
                    direction_id=sketch.direction_id,
                    direction_summary=sketch.direction_summary,
                    failure_mechanism=sketch.failure_mechanism,
                    intervention_type=sketch.intervention_type,
                    risk_level=sketch.risk_level,
                    expected_primary_metric=sketch.expected_primary_metric,
                )

        results = await expand_experiment_sketches(
            make_style_profile(),
            _make_valid_template(),
            KnowledgeBase(),
            None,
            None,
            client=FakeClient(),  # type: ignore[arg-type]
            model="fake-model",
            sketches=[sketch],
            is_first_iteration=True,
        )

        assert len(results) == 1
        assert captured["max_tokens"] == 24000
        assert "2000-8000 words" in captured["system"]  # type: ignore[operator]

    def test_shared_proposal_user_prioritizes_key_caption_sections_and_larger_feedback_budget(self) -> None:
        from art_style_search.prompt.experiments import _build_shared_proposal_user

        best = type(
            "Result",
            (),
            {
                "kept": True,
                "branch_id": 0,
                "aggregated": AggregatedMetrics(
                    dreamsim_similarity_mean=0.7,
                    dreamsim_similarity_std=0.01,
                    hps_score_mean=0.25,
                    hps_score_std=0.01,
                    aesthetics_score_mean=6.0,
                    aesthetics_score_std=0.2,
                ),
                "hypothesis": "Best hypothesis",
                "experiment": "Best experiment",
                "per_image_scores": [],
                "iteration_captions": [],
                "vision_feedback": "",
            },
        )()
        worst_caption = (
            "[Color Palette] "
            + ("palette " * 160)
            + "[Subject] SUBJECT_MARKER fox with amber eyes, satchel, lantern, lifted paw, wary glance. "
            + ("subject_detail " * 80)
            + "[Art Style] STYLE_MARKER dense impasto brushwork, muted sienna, dry-brush edges. "
            + ("style_detail " * 80)
            + "[Composition] COMPOSITION_MARKER low horizon, centered subject, marsh reeds framing both sides. "
            + ("composition_detail " * 40)
        )
        worst = type(
            "Result",
            (),
            {
                "kept": False,
                "branch_id": 1,
                "aggregated": AggregatedMetrics(
                    dreamsim_similarity_mean=0.3,
                    dreamsim_similarity_std=0.02,
                    hps_score_mean=0.12,
                    hps_score_std=0.01,
                    aesthetics_score_mean=4.0,
                    aesthetics_score_std=0.3,
                ),
                "hypothesis": "Worst hypothesis",
                "experiment": "Worst experiment",
                "per_image_scores": [MetricScores(dreamsim_similarity=0.1, hps_score=0.1, aesthetics_score=4.0)],
                "iteration_captions": [Caption(image_path=Path("worst.png"), text=worst_caption)],
                "vision_feedback": " ".join(f"vf{i}" for i in range(450)),
            },
        )()

        user = _build_shared_proposal_user(
            make_style_profile(),
            _make_valid_template(),
            KnowledgeBase(),
            best.aggregated,  # type: ignore[attr-defined]
            [best, worst],  # type: ignore[list-item]
            vision_feedback="",
            roundtrip_feedback="",
            caption_diffs="",
        )

        assert "SUBJECT_MARKER" in user
        assert "STYLE_MARKER" in user
        assert "vf350" in user
        assert "vf430" not in user


class TestRankInitialSketches:
    @pytest.mark.asyncio
    async def test_uses_ten_thousand_token_budget_for_ranking(self) -> None:
        sketches = [
            contracts.InitialTemplateSketch(
                approach_summary="First sketch",
                emphasis="technique",
                instruction_style="checklist",
                caption_length_target=500,
                caption_sections=["Art Style", "Subject", "Technique"],
                distinguishing_feature="Forces subject fidelity first.",
            ),
            contracts.InitialTemplateSketch(
                approach_summary="Second sketch",
                emphasis="palette",
                instruction_style="hybrid",
                caption_length_target=700,
                caption_sections=["Art Style", "Subject", "Color Palette"],
                distinguishing_feature="Pushes explicit palette transfer.",
            ),
        ]
        captured: dict[str, Any] = {}

        class FakeClient:
            async def call_json(self, **kwargs):
                captured.update(kwargs)
                return [1, 0]

        ranked = await rank_initial_sketches(
            sketches,
            client=FakeClient(),  # type: ignore[arg-type]
            model="fake-model",
        )

        assert ranked == [sketches[1], sketches[0]]
        assert captured["max_tokens"] == 10000


class TestRankExperimentSketches:
    @pytest.mark.asyncio
    async def test_uses_ten_thousand_token_budget_for_ranking(self) -> None:
        sketches = [
            contracts.ExperimentSketch(
                hypothesis="First sketch",
                target_category="subject_anchor",
                failure_mechanism="identity drift",
                intervention_type="information_priority",
                direction_id="D1",
                direction_summary="Direction D1",
                risk_level="targeted",
                expected_primary_metric="vision_subject",
            ),
            contracts.ExperimentSketch(
                hypothesis="Second sketch",
                target_category="composition",
                failure_mechanism="layout drift",
                intervention_type="section_schema",
                direction_id="D2",
                direction_summary="Direction D2",
                risk_level="bold",
                expected_primary_metric="vision_composition",
            ),
        ]
        captured: dict[str, Any] = {}

        class FakeClient:
            async def call_json(self, **kwargs):
                captured.update(kwargs)
                return [1, 0]

        ranked = await rank_experiment_sketches(
            sketches,
            KnowledgeBase(),
            None,
            client=FakeClient(),  # type: ignore[arg-type]
            model="fake-model",
        )

        assert ranked == [sketches[1], sketches[0]]
        assert captured["max_tokens"] == 10000

    @pytest.mark.asyncio
    async def test_falls_back_to_original_order_when_ranking_json_fails(self, caplog) -> None:
        sketches = [
            contracts.ExperimentSketch(
                hypothesis="First sketch",
                target_category="subject_anchor",
                failure_mechanism="identity drift",
                intervention_type="information_priority",
                direction_id="D1",
                direction_summary="Direction D1",
                risk_level="targeted",
                expected_primary_metric="vision_subject",
            ),
            contracts.ExperimentSketch(
                hypothesis="Second sketch",
                target_category="composition",
                failure_mechanism="layout drift",
                intervention_type="section_schema",
                direction_id="D2",
                direction_summary="Direction D2",
                risk_level="bold",
                expected_primary_metric="vision_composition",
            ),
        ]

        class FakeClient:
            async def call_json(self, **kwargs):
                raise RuntimeError("ranking validation failed after repair")

        with caplog.at_level("INFO"):
            ranked = await rank_experiment_sketches(
                sketches,
                KnowledgeBase(),
                None,
                client=FakeClient(),  # type: ignore[arg-type]
                model="fake-model",
            )

        assert ranked == sketches
        assert "Ranking failed; falling back to brainstorm order" in caplog.text
        assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


class TestHowToDrawSlotAntiConstructionGuard:
    """The "How to Draw:" slot must describe observable surface behavior, not drawing procedure.

    Every seeder prompt that describes slot 1 of the 5-facet skeleton must:
    (a) NOT list "construction order" / "construction" / "primitives" / "fabrication" as
        expected content of the slot — those prime the reasoner toward artist-procedure prose
        like "built from inflated spheres fused into composite forms" which is useless to the
        generator;
    (b) carry an explicit negative-guard sentence telling the reasoner to describe the finished
        surface and NOT construction/primitives/fabrication.

    The validator error message in prompt/_parse.py must also drop "construction order" and
    "silhouette primitives" from its enumerated hints.
    """

    _GUARD_PHRASE = "finished surface"
    _FORBIDDEN_SLOT_CONTENT_TOKENS = (
        "construction order",
        "construction + line policy",
        "construction, line policy",
    )

    def _slot_description_window(self, text: str) -> str:
        """Return text between 'How to Draw' and the next facet boundary.

        The guard rail applies only to the slot-1 instruction, not to the full seeder prompt
        (which might legitimately discuss 'prompt construction' or similar unrelated words).
        """
        idx = text.find("How to Draw")
        if idx < 0:
            return ""
        tail = text[idx:]
        for boundary in ("Shading & Light", "Style Invariants", "\n\n"):
            cut = tail.find(boundary)
            if cut > 0:
                tail = tail[:cut]
                break
        return tail

    def _assert_clean_and_guarded(self, text: str, *, label: str) -> None:
        window = self._slot_description_window(text)
        assert window, f"{label}: no 'How to Draw' slot description found"
        for token in self._FORBIDDEN_SLOT_CONTENT_TOKENS:
            assert token not in window.lower(), (
                f"{label}: slot description still lists '{token}' as expected content. Window: {window[:200]}"
            )
        assert self._GUARD_PHRASE in window.lower(), (
            f"{label}: slot description lacks the '{self._GUARD_PHRASE}' negative-guard phrase. Window: {window[:200]}"
        )

    def test_analyze_compilation_prompt_strips_construction_and_adds_guard(self) -> None:
        from art_style_search.analyze import _COMPILATION_PROMPT

        self._assert_clean_and_guarded(_COMPILATION_PROMPT, label="_COMPILATION_PROMPT")

    def test_initial_base_requirements_strips_construction_and_adds_guard(self) -> None:
        from art_style_search.prompt.initial import _BASE_REQUIREMENTS

        self._assert_clean_and_guarded(_BASE_REQUIREMENTS, label="initial._BASE_REQUIREMENTS")

    def test_initial_expand_system_strips_construction_and_adds_guard(self) -> None:
        from art_style_search.prompt.initial import _expand_system

        text = _expand_system()
        self._assert_clean_and_guarded(text, label="initial._expand_system()")

    def test_json_contracts_skeleton_prefix_strips_construction_and_adds_guard(self) -> None:
        from art_style_search.prompt.json_contracts import _STYLE_FOUNDATION_DRAWING_PREFIX

        self._assert_clean_and_guarded(_STYLE_FOUNDATION_DRAWING_PREFIX, label="_STYLE_FOUNDATION_DRAWING_PREFIX")

    def test_zero_step_caption_prompt_strips_construction_and_adds_guard(self) -> None:
        from art_style_search.caption import CAPTION_PROMPT

        self._assert_clean_and_guarded(CAPTION_PROMPT, label="CAPTION_PROMPT")

    def test_validator_error_message_no_longer_mentions_construction_or_primitives(self) -> None:
        from art_style_search.prompt._parse import _check_anchor_sub_blocks

        template = PromptTemplate(
            sections=[
                PromptSection(name="style_foundation", description="missing marker", value="no marker here"),
                PromptSection(
                    name="subject_anchor",
                    description="ok",
                    value="Proportions: 3.2 heads tall, chibi archetype. Identity: person. Distinguishing Features: stuff.",
                ),
            ],
            negative_prompt="n",
            caption_sections=["Art Style", "Subject"],
            caption_length_target=3000,
        )
        errors = _check_anchor_sub_blocks(template)
        combined = "\n".join(errors).lower()
        assert "construction" not in combined, f"validator error still mentions 'construction': {combined}"
        assert "silhouette primitives" not in combined, (
            f"validator error still mentions 'silhouette primitives': {combined}"
        )
