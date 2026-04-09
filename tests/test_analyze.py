"""Unit tests for parsing helpers in art_style_search.analyze."""

from __future__ import annotations

from art_style_search.analyze import (
    _extract_tag,
    _parse_compilation,
    _parse_sections,
)
from art_style_search.types import PromptSection

# ---------------------------------------------------------------------------
# _extract_tag
# ---------------------------------------------------------------------------


class TestExtractTag:
    def test_simple_tag(self) -> None:
        assert _extract_tag("<color>red</color>", "color") == "red"

    def test_tag_absent(self) -> None:
        assert _extract_tag("no tags here", "color") == ""

    def test_multiline_content(self) -> None:
        xml = "<description>\n  Line one.\n  Line two.\n</description>"
        result = _extract_tag(xml, "description")
        assert "Line one." in result
        assert "Line two." in result

    def test_whitespace_stripped(self) -> None:
        assert _extract_tag("<tag>  spaced  </tag>", "tag") == "spaced"

    def test_nested_text_with_other_tags(self) -> None:
        xml = "<outer><inner>hello</inner></outer>"
        # Should extract the content of <outer> including the inner tags
        result = _extract_tag(xml, "outer")
        assert "<inner>hello</inner>" in result

    def test_first_match_wins(self) -> None:
        xml = "<tag>first</tag> <tag>second</tag>"
        assert _extract_tag(xml, "tag") == "first"

    def test_empty_tag(self) -> None:
        assert _extract_tag("<tag></tag>", "tag") == ""

    def test_tag_with_surrounding_text(self) -> None:
        xml = "before <result>found it</result> after"
        assert _extract_tag(xml, "result") == "found it"


# ---------------------------------------------------------------------------
# _parse_sections
# ---------------------------------------------------------------------------


class TestParseSections:
    def test_multiple_sections(self) -> None:
        xml = (
            '<section name="style" description="overall style">impressionist</section>\n'
            '<section name="color" description="palette">warm tones</section>\n'
            '<section name="mood" description="atmosphere">serene</section>'
        )
        sections = _parse_sections(xml)
        assert len(sections) == 3
        assert sections[0] == PromptSection(name="style", description="overall style", value="impressionist")
        assert sections[1] == PromptSection(name="color", description="palette", value="warm tones")
        assert sections[2] == PromptSection(name="mood", description="atmosphere", value="serene")

    def test_no_sections(self) -> None:
        assert _parse_sections("plain text, no sections") == []

    def test_whitespace_in_values_stripped(self) -> None:
        xml = '<section name="  name  " description="  desc  ">  value  </section>'
        sections = _parse_sections(xml)
        assert len(sections) == 1
        assert sections[0].name == "name"
        assert sections[0].description == "desc"
        assert sections[0].value == "value"

    def test_multiline_value(self) -> None:
        xml = (
            '<section name="technique" description="rendering technique">\n'
            "  Thick impasto brushstrokes\n"
            "  with visible palette knife marks\n"
            "</section>"
        )
        sections = _parse_sections(xml)
        assert len(sections) == 1
        assert "Thick impasto brushstrokes" in sections[0].value
        assert "palette knife marks" in sections[0].value

    def test_single_section(self) -> None:
        xml = '<section name="only" description="the only one">solo</section>'
        sections = _parse_sections(xml)
        assert len(sections) == 1
        assert sections[0].name == "only"


# ---------------------------------------------------------------------------
# _parse_compilation
# ---------------------------------------------------------------------------


class TestParseCompilation:
    FULL_RESPONSE = (
        "<style_profile>\n"
        "  <color_palette>Muted earth tones with occasional cobalt blue accents</color_palette>\n"
        "  <composition>Centered subjects with generous negative space</composition>\n"
        "  <technique>Loose watercolor washes layered with fine ink linework</technique>\n"
        "  <mood_atmosphere>Contemplative, quiet, suggesting solitude</mood_atmosphere>\n"
        "  <subject_matter>Rural landscapes and aging architecture</subject_matter>\n"
        "  <influences>Japanese sumi-e, Winslow Homer, Edward Hopper</influences>\n"
        "</style_profile>\n"
        "<initial_template>\n"
        '  <section name="medium" description="artistic medium and technique">'
        "Watercolor and ink illustration"
        "</section>\n"
        '  <section name="palette" description="color palette">'
        "Muted earth tones with cobalt accents"
        "</section>\n"
        '  <section name="subject" description="subject matter">'
        "Rural landscape with old barn"
        "</section>\n"
        "  <negative>photorealistic, digital art, vibrant neon colors</negative>\n"
        "</initial_template>"
    )

    def test_style_profile_fields(self) -> None:
        profile, _ = _parse_compilation(self.FULL_RESPONSE, gemini_raw="gemini text", reasoning_raw="claude text")
        assert profile.color_palette == "Muted earth tones with occasional cobalt blue accents"
        assert profile.composition == "Centered subjects with generous negative space"
        assert profile.technique == "Loose watercolor washes layered with fine ink linework"
        assert profile.mood_atmosphere == "Contemplative, quiet, suggesting solitude"
        assert profile.subject_matter == "Rural landscapes and aging architecture"
        assert profile.influences == "Japanese sumi-e, Winslow Homer, Edward Hopper"

    def test_raw_analyses_stored(self) -> None:
        profile, _ = _parse_compilation(self.FULL_RESPONSE, gemini_raw="gemini output", reasoning_raw="claude output")
        assert profile.gemini_raw_analysis == "gemini output"
        assert profile.claude_raw_analysis == "claude output"

    def test_template_sections(self) -> None:
        _, template = _parse_compilation(self.FULL_RESPONSE, gemini_raw="", reasoning_raw="")
        assert len(template.sections) == 3
        assert template.sections[0].name == "medium"
        assert template.sections[0].description == "artistic medium and technique"
        assert template.sections[0].value == "Watercolor and ink illustration"
        assert template.sections[1].name == "palette"
        assert template.sections[2].name == "subject"

    def test_template_negative_prompt(self) -> None:
        _, template = _parse_compilation(self.FULL_RESPONSE, gemini_raw="", reasoning_raw="")
        assert template.negative_prompt == "photorealistic, digital art, vibrant neon colors"

    def test_missing_negative_gives_none(self) -> None:
        text = (
            "<style_profile>\n"
            "  <color_palette>warm</color_palette>\n"
            "  <composition>centered</composition>\n"
            "  <technique>oil</technique>\n"
            "  <mood_atmosphere>calm</mood_atmosphere>\n"
            "  <subject_matter>nature</subject_matter>\n"
            "  <influences>monet</influences>\n"
            "</style_profile>\n"
            "<initial_template>\n"
            '  <section name="style" description="style">oil painting</section>\n'
            "</initial_template>"
        )
        _, template = _parse_compilation(text, gemini_raw="", reasoning_raw="")
        assert template.negative_prompt is None

    def test_missing_style_profile_fields_give_empty_strings(self) -> None:
        text = (
            "<style_profile>\n"
            "  <color_palette>warm tones</color_palette>\n"
            "</style_profile>\n"
            "<initial_template>\n"
            '  <section name="s" description="d">v</section>\n'
            "</initial_template>"
        )
        profile, _ = _parse_compilation(text, gemini_raw="g", reasoning_raw="c")
        assert profile.color_palette == "warm tones"
        # Missing fields should be empty strings
        assert profile.composition == ""
        assert profile.technique == ""
        assert profile.mood_atmosphere == ""
        assert profile.subject_matter == ""
        assert profile.influences == ""

    def test_empty_initial_template_block(self) -> None:
        text = (
            "<style_profile>\n"
            "  <color_palette>blue</color_palette>\n"
            "  <composition>centered</composition>\n"
            "  <technique>digital</technique>\n"
            "  <mood_atmosphere>bright</mood_atmosphere>\n"
            "  <subject_matter>abstract</subject_matter>\n"
            "  <influences>kandinsky</influences>\n"
            "</style_profile>\n"
            "<initial_template></initial_template>"
        )
        _, template = _parse_compilation(text, gemini_raw="", reasoning_raw="")
        assert template.sections == []
        assert template.negative_prompt is None
