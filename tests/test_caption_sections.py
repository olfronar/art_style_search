"""Tests for labeled caption parsing and subject-first generation prompts."""

from __future__ import annotations

from art_style_search.caption_sections import build_generation_prompt, parse_labeled_sections


class TestParseLabeledSections:
    def test_preserves_section_order_and_content(self) -> None:
        caption = (
            "[Art Style] Shared watercolor rules.\n"
            "[Subject] A fox with a patched satchel trots through reeds.\n"
            "[Color Palette] Ochre, moss green, slate blue.\n"
            "[Composition] Low horizon with empty upper sky."
        )

        sections = parse_labeled_sections(caption)

        assert list(sections) == ["Art Style", "Subject", "Color Palette", "Composition"]
        assert sections["Subject"] == "A fox with a patched satchel trots through reeds."


class TestBuildGenerationPrompt:
    def test_prioritizes_subject_then_style_then_remaining_sections(self) -> None:
        caption = (
            "[Art Style] Shared watercolor rules with soft edges and paper grain.\n"
            "[Subject] A red fox with white socks, a patched satchel, alert ears, and a mid-stride pose.\n"
            "[Color Palette] Ochre, moss green, slate blue.\n"
            "[Composition] Low horizon with empty upper sky."
        )

        prompt = build_generation_prompt(caption)

        assert prompt.startswith("[Subject]\nA red fox")
        assert "\n\nRender in this style:\n[Art Style]\nShared watercolor rules" in prompt
        assert prompt.endswith("[Composition]\nLow horizon with empty upper sky.")

    def test_falls_back_to_raw_caption_when_subject_block_missing(self) -> None:
        caption = "[Art Style] Shared watercolor rules.\n[Color Palette] Ochre and slate blue."
        assert build_generation_prompt(caption) == caption

    def test_falls_back_to_raw_caption_when_art_style_block_missing(self) -> None:
        caption = "[Subject] A red fox with a satchel.\n[Color Palette] Ochre and slate blue."
        assert build_generation_prompt(caption) == caption
