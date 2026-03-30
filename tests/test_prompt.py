"""Unit tests for parsing and formatting helpers in art_style_search.prompt."""

from __future__ import annotations

from art_style_search.prompt import (
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
)
from art_style_search.types import AggregatedMetrics, PromptSection, PromptTemplate

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
            dino_similarity_mean=0.7234,
            dino_similarity_std=0.0123,
            lpips_distance_mean=0.4567,
            lpips_distance_std=0.0234,
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
            dino_similarity_mean=0.5,
            dino_similarity_std=0.1,
            lpips_distance_mean=0.3,
            lpips_distance_std=0.05,
            hps_score_mean=0.2,
            hps_score_std=0.01,
            aesthetics_score_mean=5.0,
            aesthetics_score_std=0.5,
        )
        output = _format_metrics(metrics)
        lines = output.strip().split("\n")
        assert len(lines) == 21
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
