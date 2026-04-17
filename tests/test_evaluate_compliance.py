"""Unit tests for caption compliance, style consistency, and section ordering/length."""

from __future__ import annotations

from pathlib import Path

from art_style_search.caption_sections import parse_labeled_sections
from art_style_search.evaluate import (
    _lengths_from_parsed,
    _ordering_from_parsed,
    check_caption_compliance,
    compute_canon_fidelity,
    compute_caption_compliance_stats,
    compute_observation_boilerplate_purity,
    compute_style_consistency,
    extract_style_canon,
)
from art_style_search.types import Caption


def _check_section_ordering(caption_text: str, expected_sections: list[str]) -> str:
    return _ordering_from_parsed(parse_labeled_sections(caption_text), expected_sections)


def _check_section_lengths(caption_text: str, expected_sections: list[str]) -> str:
    return _lengths_from_parsed(parse_labeled_sections(caption_text), expected_sections)


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


# -- compute_prompt_copying_score --------------------------------------------


_CANON_FOUNDATION = (
    "The medium class is exactly C stylized 3D CGI. The visual vocabulary is defined by modeled masses, "
    "subdivision-smooth forms, beveled edges, shaders, ambient occlusion, diffuse gradients, rim lighting, "
    "roughness, satin specular, global illumination, focal separation, and render polish. Exclude "
    "class-A and class-B terms such as brushwork, ink contour, cel fill, vector path."
)


class TestExtractStyleCanon:
    def test_empty_meta_returns_empty(self) -> None:
        assert extract_style_canon("") == ""

    def test_missing_section_returns_empty(self) -> None:
        meta = "## subject_anchor\n\nPer-image subject description only."
        assert extract_style_canon(meta) == ""

    def test_extracts_between_headings(self) -> None:
        meta = (
            "## style_foundation\n"
            "_core style rules_\n"
            "\n"
            f"{_CANON_FOUNDATION}\n"
            "\n"
            "## subject_anchor\n"
            "Per-image subject description."
        )
        canon = extract_style_canon(meta)
        assert canon == _CANON_FOUNDATION

    def test_extracts_to_end_of_document(self) -> None:
        meta = f"## style_foundation\n\n{_CANON_FOUNDATION}\n"
        assert extract_style_canon(meta).rstrip() == _CANON_FOUNDATION


class TestComputeCanonFidelity:
    def test_empty_caption_returns_neutral(self) -> None:
        assert compute_canon_fidelity("", _CANON_FOUNDATION) == 1.0

    def test_empty_canon_returns_neutral(self) -> None:
        caption = "[Art Style] verbose block " + ("foo bar " * 50)
        assert compute_canon_fidelity(caption, "") == 1.0

    def test_missing_art_style_block_returns_neutral(self) -> None:
        caption = "[Subject] a character with a red hat. [Composition] centered framing."
        assert compute_canon_fidelity(caption, _CANON_FOUNDATION) == 1.0

    def test_short_art_style_block_returns_neutral(self) -> None:
        caption = "[Art Style] tiny block."
        assert compute_canon_fidelity(caption, _CANON_FOUNDATION) == 1.0

    def test_verbatim_copy_scores_high(self) -> None:
        caption = f"[Art Style] {_CANON_FOUNDATION} [Subject] a character."
        score = compute_canon_fidelity(caption, _CANON_FOUNDATION)
        assert score > 0.7, f"verbatim canon copy should score high (good), got {score}"

    def test_paraphrase_scores_low(self) -> None:
        caption = (
            "[Art Style] "
            "This rendering reads as a polished toy-resin sculpt, where every shape feels inflated into a soft, "
            "almost pillowy mass. Contact creases darken gently without ever tipping into dirt. A single muted "
            "cyan edge catches the top silhouette, detaching the figure from its background, while flat broad "
            "gradients roll across the cheeks and shoulders without turning glossy. Local palette leans into "
            "juicy saturated complementaries rather than realistic desaturation. "
            "[Subject] the child character."
        )
        score = compute_canon_fidelity(caption, _CANON_FOUNDATION)
        assert score < 0.3, f"paraphrased voice should score low (bad), got {score}"


class TestComputeObservationBoilerplatePurity:
    def test_empty_caption_returns_neutral(self) -> None:
        assert compute_observation_boilerplate_purity("", _CANON_FOUNDATION) == 1.0

    def test_empty_canon_returns_neutral(self) -> None:
        caption = "[Subject] a character " + ("foo bar " * 50)
        assert compute_observation_boilerplate_purity(caption, "") == 1.0

    def test_clean_observations_score_high(self) -> None:
        caption = (
            "[Art Style] canon appears here. "
            "[Subject] A small orange fox sits on a mossy stump, ears forward, eyes wide and dark. "
            "A leather satchel hangs across its shoulder. "
            "[Color Palette] saturated rust, cream, and sage against a pastel cyan backdrop. "
            "[Composition] the fox occupies the lower-left third; the stump leads the eye upward. "
            "[Lighting & Atmosphere] warm high key light from camera-right, cool sky-bounce fill."
        )
        score = compute_observation_boilerplate_purity(caption, _CANON_FOUNDATION)
        assert score > 0.7, f"clean per-image observations should score high (good), got {score}"

    def test_canon_pasted_into_observations_scores_low(self) -> None:
        # The captioner mistakenly pastes the canon into every observation block.
        caption = (
            f"[Art Style] {_CANON_FOUNDATION} "
            f"[Subject] {_CANON_FOUNDATION} "
            f"[Color Palette] {_CANON_FOUNDATION} "
            f"[Composition] {_CANON_FOUNDATION} "
            f"[Lighting & Atmosphere] {_CANON_FOUNDATION}"
        )
        score = compute_observation_boilerplate_purity(caption, _CANON_FOUNDATION)
        assert score < 0.3, f"canon-polluted observations should score low (bad), got {score}"
