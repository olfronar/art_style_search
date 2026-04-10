"""Unit tests for caption compliance, style consistency, and section ordering/length."""

from __future__ import annotations

from pathlib import Path

from art_style_search.evaluate import (
    _check_section_lengths,
    _check_section_ordering,
    check_caption_compliance,
    compute_style_consistency,
)
from art_style_search.types import Caption

# -- _check_section_ordering --------------------------------------------------


class TestCheckSectionOrdering:
    def test_correct_order(self) -> None:
        text = "[Art Style] bold strokes [Color Palette] warm reds [Composition] centered"
        assert _check_section_ordering(text, ["Art Style", "Color Palette", "Composition"]) == "OK"

    def test_misordered(self) -> None:
        text = "[Composition] centered [Art Style] bold strokes [Color Palette] warm reds"
        result = _check_section_ordering(text, ["Art Style", "Color Palette", "Composition"])
        assert result.startswith("MISORDERED")

    def test_skip_when_fewer_than_two_markers(self) -> None:
        text = "[Art Style] bold strokes and nothing else"
        assert _check_section_ordering(text, ["Art Style", "Color Palette", "Composition"]) == "SKIP"

    def test_skip_when_no_markers(self) -> None:
        text = "No markers here at all."
        assert _check_section_ordering(text, ["Art Style", "Color Palette"]) == "SKIP"

    def test_partial_markers_in_order(self) -> None:
        # Only 2 of 3 are present, but in order
        text = "[Art Style] bold [Composition] centered"
        assert _check_section_ordering(text, ["Art Style", "Color Palette", "Composition"]) == "OK"


# -- _check_section_lengths ---------------------------------------------------


class TestCheckSectionLengths:
    def test_balanced(self) -> None:
        text = "[Art Style] word " * 10 + "[Color Palette] word " * 10 + "[Composition] word " * 10
        assert _check_section_lengths(text, ["Art Style", "Color Palette", "Composition"]) == "OK"

    def test_imbalanced_one_dominates(self) -> None:
        text = "[Art Style] " + "word " * 100 + "[Color Palette] word"
        result = _check_section_lengths(text, ["Art Style", "Color Palette"])
        assert result.startswith("IMBALANCED")
        assert "Art Style" in result

    def test_skip_when_no_sections_found(self) -> None:
        text = "No section markers at all."
        assert _check_section_lengths(text, ["Art Style", "Color Palette"]) == "SKIP"

    def test_empty_when_zero_words(self) -> None:
        text = "[Art Style] [Color Palette]"
        assert _check_section_lengths(text, ["Art Style", "Color Palette"]) == "EMPTY"


# -- compute_style_consistency ------------------------------------------------


class TestComputeStyleConsistency:
    def test_identical_blocks(self) -> None:
        caps = [
            Caption(image_path=Path("a.png"), text="[Art Style] bold expressive oil paint"),
            Caption(image_path=Path("b.png"), text="[Art Style] bold expressive oil paint"),
        ]
        assert compute_style_consistency(caps) == 1.0

    def test_no_overlap(self) -> None:
        caps = [
            Caption(image_path=Path("a.png"), text="[Art Style] bold expressive oil"),
            Caption(image_path=Path("b.png"), text="[Art Style] subtle delicate watercolor"),
        ]
        assert compute_style_consistency(caps) == 0.0

    def test_partial_overlap(self) -> None:
        caps = [
            Caption(image_path=Path("a.png"), text="[Art Style] bold oil paint"),
            Caption(image_path=Path("b.png"), text="[Art Style] bold watercolor paint"),
        ]
        score = compute_style_consistency(caps)
        # Jaccard of {"bold", "oil", "paint"} and {"bold", "watercolor", "paint"}
        # intersection = {"bold", "paint"} = 2, union = {"bold", "oil", "paint", "watercolor"} = 4
        assert abs(score - 0.5) < 1e-9

    def test_fewer_than_two_captions(self) -> None:
        caps = [Caption(image_path=Path("a.png"), text="[Art Style] bold")]
        assert compute_style_consistency(caps) == 0.0

    def test_missing_art_style_block(self) -> None:
        caps = [
            Caption(image_path=Path("a.png"), text="No style block here"),
            Caption(image_path=Path("b.png"), text="Also no style block"),
        ]
        assert compute_style_consistency(caps) == 0.0


# -- check_caption_compliance ------------------------------------------------


class TestCheckCaptionCompliance:
    def test_empty_inputs(self) -> None:
        assert check_caption_compliance([], []) == ""
        assert check_caption_compliance(["section"], []) == ""

    def test_keyword_hits(self) -> None:
        captions = [
            Caption(image_path=Path("a.png"), text="The style foundation uses warm palette colors"),
            Caption(image_path=Path("b.png"), text="The style foundation and composition details"),
        ]
        result = check_caption_compliance(["style_foundation", "color_palette"], captions)
        assert "style_foundation: OK" in result
        # color_palette: keywords are "color" and "palette" — "palette" in first caption
        assert "color_palette" in result

    def test_labeled_section_markers(self) -> None:
        captions = [
            Caption(image_path=Path("a.png"), text="[Art Style] bold [Color] warm"),
            Caption(image_path=Path("b.png"), text="[Art Style] expressive"),
        ]
        result = check_caption_compliance(["art_style"], captions, caption_sections=["Art Style", "Color"])
        assert "[Art Style]: OK" in result
        assert "[Color]" in result
