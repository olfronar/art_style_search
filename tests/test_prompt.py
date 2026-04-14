"""Unit tests for parsing and formatting helpers in art_style_search.prompt."""

from __future__ import annotations

import pytest

from art_style_search import contracts
from art_style_search.prompt import (
    Lessons,
    RefinementResult,
    _format_metrics,
    _format_template,
    _parse_analysis,
    _parse_builds_on,
    _parse_changed_section,
    _parse_converged,
    _parse_initial_templates,
    _parse_open_problems,
    _parse_template,
    _parse_template_changes,
    propose_experiments,
    rank_experiment_sketches,
    validate_template,
)
from art_style_search.prompt.json_contracts import (
    validate_brainstorm_payload,
    validate_expansion_payload,
    validate_experiment_batch_payload,
    validate_initial_templates_payload,
    validate_ranking_payload,
    validate_review_payload,
    validate_style_compilation_payload,
    validate_synthesis_payload,
)
from art_style_search.types import AggregatedMetrics, KnowledgeBase, PromptSection, PromptTemplate
from tests.conftest import make_style_profile

# ---------------------------------------------------------------------------
# _parse_template
# ---------------------------------------------------------------------------


class TestParseTemplate:
    def test_valid_sections(self) -> None:
        xml = (
            "<template>\n"
            '  <section name="style" description="overall style">impressionist painting</section>\n'
            '  <section name="color" description="color palette">warm earth tones</section>\n'
            "</template>"
        )
        tpl = _parse_template(xml)
        assert len(tpl.sections) == 2
        assert tpl.sections[0].name == "style"
        assert tpl.sections[0].description == "overall style"
        assert tpl.sections[0].value == "impressionist painting"
        assert tpl.sections[1].name == "color"
        assert tpl.sections[1].description == "color palette"
        assert tpl.sections[1].value == "warm earth tones"

    def test_negative_prompt_present(self) -> None:
        xml = (
            '<section name="style" description="overall style">watercolor</section>\n'
            "<negative>photorealistic, 3d render</negative>"
        )
        tpl = _parse_template(xml)
        assert tpl.negative_prompt == "photorealistic, 3d render"

    def test_negative_prompt_absent(self) -> None:
        xml = '<section name="style" description="overall style">watercolor</section>'
        tpl = _parse_template(xml)
        assert tpl.negative_prompt is None

    def test_negative_prompt_empty(self) -> None:
        xml = '<section name="style" description="overall style">watercolor</section>\n<negative>  </negative>'
        tpl = _parse_template(xml)
        # Empty whitespace-only negative should become None
        assert tpl.negative_prompt is None

    def test_no_sections(self) -> None:
        tpl = _parse_template("just some random text with no XML")
        assert tpl.sections == []
        assert tpl.negative_prompt is None

    def test_malformed_section_missing_closing_tag(self) -> None:
        xml = '<section name="style" description="overall style">watercolor'
        tpl = _parse_template(xml)
        assert tpl.sections == []

    def test_multiline_section_value(self) -> None:
        xml = (
            '<section name="mood" description="mood and atmosphere">\n'
            "  Dreamy, ethereal quality\n"
            "  with soft diffused lighting\n"
            "  and gentle gradients\n"
            "</section>"
        )
        tpl = _parse_template(xml)
        assert len(tpl.sections) == 1
        assert "Dreamy, ethereal quality" in tpl.sections[0].value
        assert "gentle gradients" in tpl.sections[0].value

    def test_whitespace_stripped_from_names_and_values(self) -> None:
        xml = '<section name="  style  " description="  desc  ">  value  </section>'
        tpl = _parse_template(xml)
        assert tpl.sections[0].name == "style"
        assert tpl.sections[0].description == "desc"
        assert tpl.sections[0].value == "value"


# ---------------------------------------------------------------------------
# _parse_analysis
# ---------------------------------------------------------------------------


class TestParseAnalysis:
    def test_analysis_present(self) -> None:
        text = "Some preamble\n<analysis>The DINO score improved by 0.05.</analysis>\nMore text"
        assert _parse_analysis(text) == "The DINO score improved by 0.05."

    def test_analysis_absent(self) -> None:
        assert _parse_analysis("No analysis tags here") == ""

    def test_analysis_multiline(self) -> None:
        text = "<analysis>\nLine 1\nLine 2\nLine 3\n</analysis>"
        result = _parse_analysis(text)
        assert "Line 1" in result
        assert "Line 3" in result

    def test_analysis_empty(self) -> None:
        assert _parse_analysis("<analysis>   </analysis>") == ""


class TestPromptContracts:
    def test_prompt_reexports_contract_types_from_neutral_contracts_module(self) -> None:
        assert Lessons is contracts.Lessons
        assert RefinementResult is contracts.RefinementResult


# ---------------------------------------------------------------------------
# _parse_template_changes
# ---------------------------------------------------------------------------


class TestParseTemplateChanges:
    def test_changes_present(self) -> None:
        text = "<template_changes>Added a new technique section</template_changes>"
        assert _parse_template_changes(text) == "Added a new technique section"

    def test_changes_absent(self) -> None:
        assert _parse_template_changes("No changes tags") == ""

    def test_changes_none_value(self) -> None:
        text = "<template_changes>none</template_changes>"
        assert _parse_template_changes(text) == "none"

    def test_changes_multiline(self) -> None:
        text = "<template_changes>\nRemoved color section.\nAdded texture section.\n</template_changes>"
        result = _parse_template_changes(text)
        assert "Removed color section." in result
        assert "Added texture section." in result


# ---------------------------------------------------------------------------
# _parse_converged
# ---------------------------------------------------------------------------


class TestParseConverged:
    def test_converged_present(self) -> None:
        assert _parse_converged("Some analysis text\n[CONVERGED]") is True

    def test_converged_absent(self) -> None:
        assert _parse_converged("Some analysis text, not converged yet") is False

    def test_converged_in_middle(self) -> None:
        assert _parse_converged("Blah [CONVERGED] blah") is True

    def test_converged_substring_not_matched(self) -> None:
        # "CONVERGED" without brackets should not match
        assert _parse_converged("The model has CONVERGED on a solution") is False


# ---------------------------------------------------------------------------
# _parse_initial_templates
# ---------------------------------------------------------------------------


class TestParseInitialTemplates:
    def test_multiple_branches(self) -> None:
        text = (
            "<branch>\n"
            '<section name="style" description="style">bold lines</section>\n'
            "</branch>\n"
            "<branch>\n"
            '<section name="mood" description="mood">calm atmosphere</section>\n'
            "<negative>harsh lighting</negative>\n"
            "</branch>"
        )
        templates = _parse_initial_templates(text, num_branches=2)
        assert len(templates) == 2
        assert templates[0].sections[0].value == "bold lines"
        assert templates[1].sections[0].value == "calm atmosphere"
        assert templates[1].negative_prompt == "harsh lighting"

    def test_single_template_fallback(self) -> None:
        # No <branch> tags: should fall back to parsing the whole text as one template
        text = '<section name="style" description="style">watercolor wash</section>'
        templates = _parse_initial_templates(text, num_branches=1)
        assert len(templates) == 1
        assert templates[0].sections[0].value == "watercolor wash"

    def test_padding_to_num_branches(self) -> None:
        # Only one branch provided but 3 requested: should pad by duplicating last
        text = '<branch>\n<section name="style" description="style">oil painting</section>\n</branch>'
        templates = _parse_initial_templates(text, num_branches=3)
        assert len(templates) == 3
        # The padded templates should be the same object as the last parsed one
        assert templates[1] is templates[0]
        assert templates[2] is templates[0]

    def test_more_branches_than_requested(self) -> None:
        text = (
            "<branch>\n"
            '<section name="a" description="a">val_a</section>\n'
            "</branch>\n"
            "<branch>\n"
            '<section name="b" description="b">val_b</section>\n'
            "</branch>\n"
            "<branch>\n"
            '<section name="c" description="c">val_c</section>\n'
            "</branch>"
        )
        templates = _parse_initial_templates(text, num_branches=2)
        assert len(templates) == 2
        assert templates[0].sections[0].value == "val_a"
        assert templates[1].sections[0].value == "val_b"

    def test_fallback_pads_when_num_branches_gt_1(self) -> None:
        # No <branch> tags, but num_branches=3: fallback gives 1, then pad to 3
        text = '<section name="style" description="style">pencil sketch</section>'
        templates = _parse_initial_templates(text, num_branches=3)
        assert len(templates) == 3
        for t in templates:
            assert t.sections[0].value == "pencil sketch"


# ---------------------------------------------------------------------------
# _format_template
# ---------------------------------------------------------------------------


class TestFormatTemplate:
    def test_round_trip(self) -> None:
        """A formatted template should re-parse into an equivalent template."""
        original = PromptTemplate(
            sections=[
                PromptSection(name="style", description="overall style", value="impressionist painting"),
                PromptSection(name="color", description="color palette", value="warm earth tones"),
            ],
            negative_prompt="photorealistic, 3d render",
        )
        xml = _format_template(original)
        parsed = _parse_template(xml)

        assert len(parsed.sections) == len(original.sections)
        for orig_sec, parsed_sec in zip(original.sections, parsed.sections, strict=True):
            assert orig_sec.name == parsed_sec.name
            assert orig_sec.description == parsed_sec.description
            assert orig_sec.value == parsed_sec.value
        assert parsed.negative_prompt == original.negative_prompt

    def test_no_negative_prompt(self) -> None:
        tpl = PromptTemplate(
            sections=[PromptSection(name="style", description="style", value="watercolor")],
            negative_prompt=None,
        )
        xml = _format_template(tpl)
        assert "<negative>" not in xml
        parsed = _parse_template(xml)
        assert parsed.negative_prompt is None

    def test_output_contains_template_tags(self) -> None:
        tpl = PromptTemplate(
            sections=[PromptSection(name="x", description="y", value="z")],
        )
        xml = _format_template(tpl)
        assert xml.startswith("<template>")
        assert xml.endswith("</template>")

    def test_empty_sections(self) -> None:
        tpl = PromptTemplate(sections=[], negative_prompt="avoid noise")
        xml = _format_template(tpl)
        parsed = _parse_template(xml)
        assert parsed.sections == []
        assert parsed.negative_prompt == "avoid noise"


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
# _parse_builds_on
# ---------------------------------------------------------------------------


class TestParseBuildsOn:
    def test_single_id(self) -> None:
        text = "<builds_on>H3</builds_on>"
        assert _parse_builds_on(text) == "H3"

    def test_multiple_ids(self) -> None:
        text = "<builds_on>H3, H5</builds_on>"
        assert _parse_builds_on(text) == "H3, H5"

    def test_none_value(self) -> None:
        text = "<builds_on>none</builds_on>"
        assert _parse_builds_on(text) is None

    def test_none_case_insensitive(self) -> None:
        text = "<builds_on>None</builds_on>"
        assert _parse_builds_on(text) is None

    def test_absent(self) -> None:
        text = "Some text without builds_on tag"
        assert _parse_builds_on(text) is None

    def test_with_whitespace(self) -> None:
        text = "<builds_on>  H7  </builds_on>"
        assert _parse_builds_on(text) == "H7"


# ---------------------------------------------------------------------------
# _parse_open_problems
# ---------------------------------------------------------------------------


class TestParseOpenProblems:
    def test_numbered_list(self) -> None:
        text = (
            "<open_problems>\n"
            "1. Color matching on dark palettes\n"
            "2. Fine texture detail\n"
            "3. Mood consistency\n"
            "</open_problems>"
        )
        result = _parse_open_problems(text)
        assert len(result) == 3
        assert result[0] == "Color matching on dark palettes"
        assert result[1] == "Fine texture detail"
        assert result[2] == "Mood consistency"

    def test_absent(self) -> None:
        assert _parse_open_problems("No tag here") == []

    def test_empty(self) -> None:
        assert _parse_open_problems("<open_problems>  </open_problems>") == []

    def test_single_item(self) -> None:
        text = "<open_problems>\n1. Only one problem\n</open_problems>"
        result = _parse_open_problems(text)
        assert len(result) == 1
        assert result[0] == "Only one problem"


# ---------------------------------------------------------------------------
# _parse_changed_section
# ---------------------------------------------------------------------------


class TestParseChangedSection:
    def test_present(self) -> None:
        assert _parse_changed_section("<changed_section>colors_and_palette</changed_section>") == "colors_and_palette"

    def test_absent(self) -> None:
        assert _parse_changed_section("No tag here") == ""

    def test_whitespace_trimmed(self) -> None:
        assert _parse_changed_section("<changed_section>  mood_atmosphere  </changed_section>") == "mood_atmosphere"


# ---------------------------------------------------------------------------
# validate_template
# ---------------------------------------------------------------------------


def _make_valid_template() -> PromptTemplate:
    """Build a minimal valid template for validation tests."""
    sections = [
        PromptSection(name="style_foundation", description="Style rules", value="Foundation rules."),
        PromptSection(name="subject_anchor", description="Subject rules", value="Subject rules."),
        PromptSection(name="color_palette", description="Colors", value="Color rules."),
        PromptSection(name="composition", description="Layout", value="Comp rules."),
        PromptSection(name="technique", description="Technique", value="Tech rules."),
    ]
    return PromptTemplate(
        sections=sections,
        caption_sections=["Art Style", "Subject", "Color Palette", "Composition"],
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
        t.sections = t.sections[:2]
        errors = validate_template(t)
        assert any("Section count" in e for e in errors)

    def test_caption_length_out_of_bounds(self) -> None:
        t = _make_valid_template()
        t.caption_length_target = 50
        errors = validate_template(t)
        assert any("Caption length" in e for e in errors)

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
        current.sections[2] = PromptSection(name="face_hands_pose", description="anatomy", value="Pose rules.")

        proposed = _make_valid_template()
        proposed.sections[2] = PromptSection(
            name="scene_type_and_asset_class",
            description="scene taxonomy",
            value="Scene taxonomy rules.",
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


# ---------------------------------------------------------------------------
# JSON contract validation
# ---------------------------------------------------------------------------


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

    def test_expansion_payload_reads_single_refinement_result(self) -> None:
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

        result = validate_expansion_payload(payload)

        assert result.hypothesis == "Strengthen subject identity anchors."
        assert result.changed_sections == ["subject_anchor"]
        assert result.expected_primary_metric == "vision_subject"

    def test_initial_templates_payload_pads_to_requested_count(self) -> None:
        payload = {
            "templates": [
                {
                    "sections": [{"name": "style_foundation", "description": "rules", "value": "Shared rules"}],
                    "negative_prompt": "avoid blur",
                    "caption_sections": ["Art Style"],
                    "caption_length_target": 500,
                }
            ]
        }
        templates = validate_initial_templates_payload(payload, num_branches=3)
        assert len(templates) == 3
        assert templates[0].sections[0].name == "style_foundation"
        assert templates[1] is templates[0]
        assert templates[2] is templates[0]

    def test_experiment_batch_payload_reads_converged_flag(self) -> None:
        payload = {
            "experiments": [
                {
                    "analysis": "Need better palette control",
                    "lessons": {"confirmed": "", "rejected": "", "new_insight": "Color drifts on dark images"},
                    "hypothesis": "Strengthen palette anchoring",
                    "builds_on": "H3",
                    "experiment": "Tighten the color section",
                    "changed_sections": ["color_palette"],
                    "target_category": "color_palette",
                    "direction_id": "D1",
                    "direction_summary": "Palette localization",
                    "failure_mechanism": "Large background fields and tiny accents are described at the same priority.",
                    "intervention_type": "information_priority",
                    "risk_level": "targeted",
                    "expected_primary_metric": "color_histogram",
                    "expected_tradeoff": "May reduce caption naturalness if the section becomes too rigid.",
                    "open_problems": ["Dark palettes drift"],
                    "template_changes": "none",
                    "template": {
                        "sections": [
                            {"name": "style_foundation", "description": "rules", "value": "Shared rules"},
                            {"name": "subject_anchor", "description": "subject rules", "value": "Subject guidance"},
                            {"name": "color_palette", "description": "colors", "value": "Detailed palette guidance"},
                            {"name": "composition", "description": "layout", "value": "Layout guidance"},
                        ],
                        "negative_prompt": "avoid blur",
                        "caption_sections": ["Art Style", "Subject", "Color Palette"],
                        "caption_length_target": 500,
                    },
                }
            ],
            "converged": True,
        }
        results, converged = validate_experiment_batch_payload(payload, num_experiments=2)
        assert converged is True
        assert len(results) == 1
        assert results[0].builds_on == "H3"
        assert results[0].target_category == "color_palette"
        assert results[0].direction_id == "D1"
        assert results[0].direction_summary == "Palette localization"
        assert results[0].failure_mechanism.startswith("Large background fields")
        assert results[0].intervention_type == "information_priority"
        assert results[0].risk_level == "targeted"
        assert results[0].expected_primary_metric == "color_histogram"
        assert results[0].expected_tradeoff.startswith("May reduce caption naturalness")
        assert results[0].changed_sections == ["color_palette"]
        assert results[0].changed_section == "color_palette"

    def test_experiment_batch_payload_backfills_changed_sections_from_changed_section(self) -> None:
        payload = {
            "experiments": [
                {
                    "analysis": "Need a bolder schema change",
                    "lessons": {"confirmed": "", "rejected": "", "new_insight": ""},
                    "hypothesis": "Try a two-section schema shift",
                    "builds_on": None,
                    "experiment": "Rewrite subject and scene geometry together",
                    "changed_section": "subject_anchor",
                    "target_category": "subject_anchor",
                    "risk_level": "bold",
                    "template_changes": "none",
                    "template": {
                        "sections": [
                            {"name": "style_foundation", "description": "rules", "value": "Shared rules"},
                            {"name": "subject_anchor", "description": "subject rules", "value": "Subject guidance"},
                            {"name": "scene_geometry", "description": "layout", "value": "Layout guidance"},
                            {"name": "texture_vocabulary", "description": "texture", "value": "Texture guidance"},
                        ],
                        "caption_sections": ["Art Style", "Subject", "Scene Geometry"],
                        "caption_length_target": 500,
                    },
                }
            ]
        }

        results, converged = validate_experiment_batch_payload(payload, num_experiments=1)
        assert converged is False
        assert results[0].changed_sections == ["subject_anchor"]
        assert results[0].changed_section == "subject_anchor"

    def test_experiment_batch_payload_prefers_changed_sections_when_legacy_field_mismatches(self) -> None:
        payload = {
            "experiments": [
                {
                    "analysis": "Repair output mixed the legacy and list fields",
                    "lessons": {"confirmed": "", "rejected": "", "new_insight": ""},
                    "hypothesis": "Keep the multi-section metadata authoritative",
                    "builds_on": None,
                    "experiment": "Tune composition and environment together",
                    "changed_section": "subject_anchor",
                    "changed_sections": ["composition", "environment_staging"],
                    "target_category": "composition",
                    "risk_level": "bold",
                    "template_changes": "none",
                    "template": {
                        "sections": [
                            {"name": "style_foundation", "description": "rules", "value": "Shared rules"},
                            {"name": "subject_anchor", "description": "subject rules", "value": "Subject guidance"},
                            {"name": "composition", "description": "layout", "value": "Layout guidance"},
                            {
                                "name": "environment_staging",
                                "description": "scene staging",
                                "value": "Environment guidance",
                            },
                        ],
                        "caption_sections": ["Art Style", "Subject", "Composition"],
                        "caption_length_target": 500,
                    },
                }
            ]
        }

        results, converged = validate_experiment_batch_payload(payload, num_experiments=1)

        assert converged is False
        assert results[0].changed_sections == ["composition", "environment_staging"]
        assert results[0].changed_section == "composition"

    def test_synthesis_payload_accepts_negative_prompt_key(self) -> None:
        template, rationale = validate_synthesis_payload(
            {
                "rationale": "Take the color section from experiment 1",
                "template": {
                    "sections": [
                        {"name": "style_foundation", "description": "rules", "value": "Shared rules"},
                        {"name": "color_palette", "description": "colors", "value": "Palette rules"},
                        {"name": "composition", "description": "layout", "value": "Layout rules"},
                        {"name": "technique", "description": "medium", "value": "Technique rules"},
                    ],
                    "negative_prompt": "avoid blur",
                    "caption_sections": ["Art Style", "Color Palette"],
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


class TestProposeExperiments:
    @pytest.mark.asyncio
    async def test_uses_two_repair_retries_and_concrete_changed_section_instructions(self) -> None:
        captured: dict[str, object] = {}

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


class TestRankExperimentSketches:
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

        with caplog.at_level("WARNING"):
            ranked = await rank_experiment_sketches(
                sketches,
                KnowledgeBase(),
                None,
                client=FakeClient(),  # type: ignore[arg-type]
                model="fake-model",
            )

        assert ranked == sketches
        assert "Ranking failed; falling back to brainstorm order" in caplog.text
