"""Unit tests for parsing helpers used by art_style_search.analyze."""

from __future__ import annotations

from art_style_search.utils import extract_xml_tag

# ---------------------------------------------------------------------------
# extract_xml_tag
# ---------------------------------------------------------------------------


class TestExtractTag:
    def test_simple_tag(self) -> None:
        assert extract_xml_tag("<color>red</color>", "color") == "red"

    def test_tag_absent(self) -> None:
        assert extract_xml_tag("no tags here", "color") == ""

    def test_multiline_content(self) -> None:
        xml = "<description>\n  Line one.\n  Line two.\n</description>"
        result = extract_xml_tag(xml, "description")
        assert "Line one." in result
        assert "Line two." in result

    def test_whitespace_stripped(self) -> None:
        assert extract_xml_tag("<tag>  spaced  </tag>", "tag") == "spaced"

    def test_nested_text_with_other_tags(self) -> None:
        xml = "<outer><inner>hello</inner></outer>"
        # Should extract the content of <outer> including the inner tags
        result = extract_xml_tag(xml, "outer")
        assert "<inner>hello</inner>" in result

    def test_first_match_wins(self) -> None:
        xml = "<tag>first</tag> <tag>second</tag>"
        assert extract_xml_tag(xml, "tag") == "first"

    def test_empty_tag(self) -> None:
        assert extract_xml_tag("<tag></tag>", "tag") == ""

    def test_tag_with_surrounding_text(self) -> None:
        xml = "before <result>found it</result> after"
        assert extract_xml_tag(xml, "result") == "found it"
