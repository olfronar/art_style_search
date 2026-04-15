"""Unit tests for caption compliance, style consistency, and section ordering/length."""

from __future__ import annotations

from pathlib import Path

from art_style_search.evaluate import (
    _check_section_lengths,
    _check_section_ordering,
    check_caption_compliance,
    compute_caption_compliance_stats,
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

    def test_primary_section_can_dominate_up_to_sixty_percent(self) -> None:
        text = (
            "[Art Style] "
            + ("style " * 30)
            + "[Subject] "
            + ("subject " * 110)
            + "[Composition] "
            + ("composition " * 30)
            + "[Lighting] "
            + ("lighting " * 30)
        )
        assert _check_section_lengths(text, ["Art Style", "Subject", "Composition", "Lighting"]) == "OK"

    def test_primary_section_above_sixty_percent_is_imbalanced(self) -> None:
        text = (
            "[Art Style] "
            + ("style " * 20)
            + "[Subject] "
            + ("subject " * 130)
            + "[Composition] "
            + ("composition " * 25)
            + "[Lighting] "
            + ("lighting " * 25)
        )
        result = _check_section_lengths(text, ["Art Style", "Subject", "Composition", "Lighting"])
        assert result.startswith("IMBALANCED")
        assert "Subject" in result

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


class TestComputeCaptionComplianceStats:
    def test_returns_structured_rates_for_perfect_compliance(self) -> None:
        captions = [
            Caption(
                image_path=Path("a.png"),
                text=(
                    "[Art Style] style foundation is described with calm precise detail. "
                    "[Frame Map] frame map places the main masses in stable positions. "
                    "[Depth Layers] depth layers are mapped from foreground to background with equal care. "
                    "[Atmosphere] atmosphere remains serene and plainly stated."
                ),
            ),
            Caption(
                image_path=Path("b.png"),
                text=(
                    "[Art Style] style foundation remains explicit with stable descriptive coverage. "
                    "[Frame Map] frame map keeps the composition readable and concrete. "
                    "[Depth Layers] depth layers stay ordered and balanced across the full scene. "
                    "[Atmosphere] atmosphere stays calm and observational."
                ),
            ),
        ]

        stats = compute_caption_compliance_stats(
            ["style_foundation", "frame_map", "depth_layers", "atmosphere"],
            captions,
            caption_sections=["Art Style", "Frame Map", "Depth Layers", "Atmosphere"],
        )

        assert stats.section_topic_coverage == 1.0
        assert stats.section_marker_coverage == 1.0
        assert stats.section_ordering_rate == 1.0
        assert stats.section_balance_rate == 1.0
        assert stats.overall == 1.0

    def test_subject_specificity_rate_is_full_for_rich_subject_blocks(self) -> None:
        detail_chunk = (
            "A young red fox with white socks and a narrow muzzle stands as the main animal subject. "
            "Its amber eyes, nicked left ear, and dark foreleg markings make it immediately identifiable. "
            "It wears a weathered canvas satchel with brass clasps, a thin leather harness, and a wrapped field lantern. "
            "The fox is caught mid-step, turning its shoulders while lifting one paw and twisting toward the viewer. "
            "Its expression is alert but wary, with raised ears, a tight mouth, and a focused sideways glance. "
            "Nearby props include the lantern, a folded map, and broken reeds that frame the animal in context."
        )
        subject_text = " ".join([detail_chunk] * 4)
        captions = [
            Caption(
                image_path=Path("a.png"),
                text=f"[Art Style] shared style rules [Subject] {subject_text} [Composition] low horizon",
            ),
            Caption(
                image_path=Path("b.png"),
                text=f"[Art Style] shared style rules [Subject] {subject_text} [Composition] low horizon",
            ),
        ]

        stats = compute_caption_compliance_stats(
            ["style_foundation", "subject_anchor", "composition"],
            captions,
            caption_sections=["Art Style", "Subject", "Composition"],
        )

        assert stats.subject_specificity_rate == 1.0

    def test_subject_specificity_rate_rejects_long_sections_with_detail_only_up_front(self) -> None:
        detail_chunk = (
            "A young red fox with white socks and a narrow muzzle stands as the main animal subject. "
            "Its amber eyes, nicked left ear, and dark foreleg markings make it immediately identifiable. "
            "It wears a weathered canvas satchel with brass clasps, a thin leather harness, and a wrapped field lantern. "
            "The fox is caught mid-step, turning its shoulders while lifting one paw and twisting toward the viewer. "
            "Its expression is alert but wary, with raised ears, a tight mouth, and a focused sideways glance. "
            "Nearby props include the lantern, a folded map, and broken reeds that frame the animal in context."
        )
        filler = "contour atmosphere tonal surface interval staging backdrop rhythm " * 70
        subject_text = f"{detail_chunk} {detail_chunk} {filler}"
        captions = [
            Caption(
                image_path=Path("a.png"),
                text=f"[Art Style] shared style rules [Subject] {subject_text} [Composition] low horizon",
            ),
        ]

        stats = compute_caption_compliance_stats(
            ["style_foundation", "subject_anchor", "composition"],
            captions,
            caption_sections=["Art Style", "Subject", "Composition"],
        )

        assert stats.subject_specificity_rate == 0.0

    def test_subject_specificity_rate_rejects_short_generic_subject_blocks(self) -> None:
        captions = [
            Caption(
                image_path=Path("a.png"),
                text="[Art Style] shared style rules [Subject] person figure creature object thing [Composition] centered",
            ),
            Caption(
                image_path=Path("b.png"),
                text="[Art Style] shared style rules [Subject] person figure creature [Composition] centered",
            ),
        ]

        stats = compute_caption_compliance_stats(
            ["style_foundation", "subject_anchor", "composition"],
            captions,
            caption_sections=["Art Style", "Subject", "Composition"],
        )

        assert stats.subject_specificity_rate == 0.0
