"""Unit tests for art_style_search.analyze — parsing helpers and cache round-trip."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from art_style_search.analyze import (
    _ANALYSIS_SYSTEM,
    _COMPILATION_PROMPT,
    _GEMINI_ANALYSIS_PROMPT,
    _REASONING_ANALYSIS_PROMPT,
    _gemini_analyze,
    _load_cache,
    _save_cache,
)
from art_style_search.types import PromptSection, PromptTemplate
from art_style_search.utils import extract_xml_tag
from tests.conftest import make_style_profile

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


# ---------------------------------------------------------------------------
# _save_cache / _load_cache round-trip
# ---------------------------------------------------------------------------


class TestStyleCache:
    @staticmethod
    def _make_valid_template() -> PromptTemplate:
        return PromptTemplate(
            sections=[
                PromptSection(name="style_foundation", description="rules", value="Shared rules. " * 80),
                PromptSection(name="subject_anchor", description="subject rules", value="Subject guidance. " * 80),
                PromptSection(name="color_palette", description="colors", value="Palette guidance. " * 80),
                PromptSection(name="composition", description="layout", value="Composition guidance. " * 80),
                PromptSection(name="technique", description="medium", value="Technique guidance. " * 80),
                PromptSection(name="lighting", description="light", value="Lighting guidance. " * 80),
                PromptSection(name="environment", description="environment", value="Environment guidance. " * 80),
                PromptSection(name="textures", description="textures", value="Texture guidance. " * 80),
            ],
            negative_prompt="avoid blur",
            caption_sections=["Art Style", "Subject", "Color Palette", "Composition", "Technique"],
            caption_length_target=500,
        )

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        profile = make_style_profile()
        template = self._make_valid_template()
        cache_file = tmp_path / "style_cache.json"

        _save_cache(profile, template, cache_file)
        assert cache_file.exists()

        result = _load_cache(cache_file)
        assert result is not None
        loaded_profile, loaded_template = result
        assert loaded_profile.color_palette == profile.color_palette
        assert loaded_profile.technique == profile.technique
        assert len(loaded_template.sections) == len(template.sections)
        for orig, loaded in zip(template.sections, loaded_template.sections, strict=True):
            assert orig.name == loaded.name
            assert orig.value == loaded.value

    def test_load_missing_file(self, tmp_path: Path) -> None:
        result = _load_cache(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "corrupt.json"
        cache_file.write_text("not valid json {{{", encoding="utf-8")
        result = _load_cache(cache_file)
        assert result is None

    def test_load_missing_keys(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "incomplete.json"
        cache_file.write_text(json.dumps({"style_profile": {}}), encoding="utf-8")
        result = _load_cache(cache_file)
        assert result is None

    def test_load_invalid_template_returns_none(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "invalid_template.json"
        cache_file.write_text(
            json.dumps(
                {
                    "style_profile": {
                        "color_palette": "Muted earth tones.",
                        "composition": "Low horizon.",
                        "technique": "Wet-on-wet watercolor.",
                        "mood_atmosphere": "Quiet and contemplative.",
                        "subject_matter": "Rural landscapes.",
                        "influences": "Turner and Wyeth.",
                        "gemini_raw_analysis": "visual analysis",
                        "claude_raw_analysis": "reasoning analysis",
                    },
                    "prompt_template": {
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
            ),
            encoding="utf-8",
        )

        result = _load_cache(cache_file)
        assert result is None

    def test_cache_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "cache.json"
        _save_cache(make_style_profile(), self._make_valid_template(), nested)
        assert nested.exists()


class TestAnalyzePrompts:
    def test_reasoning_prompt_drops_generic_format_disclaimer(self) -> None:
        assert "generic section format" not in _REASONING_ANALYSIS_PROMPT

    def test_compilation_prompt_uses_json_key_language_not_xml_tags(self) -> None:
        assert "<caption_sections>" not in _COMPILATION_PROMPT
        assert "<caption_length>" not in _COMPILATION_PROMPT
        assert '"caption_sections"' in _COMPILATION_PROMPT
        assert '"caption_length_target"' in _COMPILATION_PROMPT

    @pytest.mark.asyncio
    async def test_gemini_analyze_uses_system_instruction(self, tmp_path: Path) -> None:
        image_path = tmp_path / "ref.png"
        Image.new("RGB", (8, 8), color="blue").save(image_path)
        captured: dict[str, object] = {}

        async def fake_generate_content(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="analysis")

        client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content)))

        result = await _gemini_analyze([image_path], client=client, model="fake-model")  # type: ignore[arg-type]

        assert result == "analysis"
        assert captured["config"].system_instruction == _ANALYSIS_SYSTEM
        assert _GEMINI_ANALYSIS_PROMPT in captured["contents"]
