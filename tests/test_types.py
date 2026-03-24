"""Unit tests for art_style_search.types."""

from __future__ import annotations

from art_style_search.types import (
    AggregatedMetrics,
    ConvergenceReason,
    KnowledgeBase,
    OpenProblem,
    PromptSection,
    PromptTemplate,
    classify_hypothesis,
    composite_score,
    get_category_names,
)

# -- composite_score ----------------------------------------------------------


class TestCompositeScore:
    def test_known_values(self) -> None:
        m = AggregatedMetrics(
            dino_similarity_mean=0.8,
            dino_similarity_std=0.01,
            lpips_distance_mean=0.3,
            lpips_distance_std=0.02,
            hps_score_mean=0.5,
            hps_score_std=0.03,
            aesthetics_score_mean=7.0,
            aesthetics_score_std=0.5,
        )
        expected = 0.4 * 0.8 - 0.2 * 0.3 + 0.2 * 0.5 + 0.2 * (7.0 / 10.0)
        assert composite_score(m) == expected

    def test_zero_metrics(self) -> None:
        m = AggregatedMetrics(
            dino_similarity_mean=0.0,
            dino_similarity_std=0.0,
            lpips_distance_mean=0.0,
            lpips_distance_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        assert composite_score(m) == 0.0

    def test_higher_dino_yields_higher_score(self) -> None:
        base = dict(
            dino_similarity_std=0.0,
            lpips_distance_mean=0.0,
            lpips_distance_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        low = AggregatedMetrics(dino_similarity_mean=0.2, **base)
        high = AggregatedMetrics(dino_similarity_mean=0.9, **base)
        assert composite_score(high) > composite_score(low)

    def test_higher_lpips_lowers_score(self) -> None:
        base = dict(
            dino_similarity_mean=0.5,
            dino_similarity_std=0.0,
            lpips_distance_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        low_lpips = AggregatedMetrics(lpips_distance_mean=0.1, **base)
        high_lpips = AggregatedMetrics(lpips_distance_mean=0.9, **base)
        assert composite_score(low_lpips) > composite_score(high_lpips)

    def test_weights_sum_to_one(self) -> None:
        """Verify the coefficient magnitudes sum to 1.0 (0.4 + 0.2 + 0.2 + 0.2)."""
        assert abs((0.4 + 0.2 + 0.2 + 0.2) - 1.0) < 1e-9

    def test_aesthetics_divided_by_ten(self) -> None:
        """Aesthetics is on a 1-10 scale and should be normalized to 0-1."""
        m = AggregatedMetrics(
            dino_similarity_mean=0.0,
            dino_similarity_std=0.0,
            lpips_distance_mean=0.0,
            lpips_distance_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=10.0,
            aesthetics_score_std=0.0,
        )
        # Only the aesthetics term contributes: 0.2 * (10.0 / 10.0) = 0.2
        assert composite_score(m) == 0.2


# -- PromptTemplate.render ---------------------------------------------------


class TestPromptTemplateRender:
    def test_basic_sections(self) -> None:
        t = PromptTemplate(
            sections=[
                PromptSection(name="style", description="overall style", value="watercolor painting"),
                PromptSection(name="color", description="palette", value="muted earth tones"),
            ]
        )
        assert t.render() == "watercolor painting muted earth tones"

    def test_with_negative_prompt(self) -> None:
        t = PromptTemplate(
            sections=[
                PromptSection(name="style", description="overall style", value="oil painting"),
            ],
            negative_prompt="blurry, low quality",
        )
        assert t.render() == "oil painting Do NOT include: blurry, low quality"

    def test_empty_sections(self) -> None:
        t = PromptTemplate(sections=[])
        assert t.render() == ""

    def test_empty_template_with_negative_prompt(self) -> None:
        t = PromptTemplate(sections=[], negative_prompt="noise")
        assert t.render() == "Do NOT include: noise"

    def test_sections_with_empty_values_are_skipped(self) -> None:
        t = PromptTemplate(
            sections=[
                PromptSection(name="a", description="desc", value=""),
                PromptSection(name="b", description="desc", value="keep me"),
                PromptSection(name="c", description="desc", value=""),
            ]
        )
        assert t.render() == "keep me"

    def test_none_negative_prompt_is_omitted(self) -> None:
        t = PromptTemplate(
            sections=[PromptSection(name="s", description="d", value="hello")],
            negative_prompt=None,
        )
        assert "Avoid" not in t.render()

    def test_empty_string_negative_prompt_is_omitted(self) -> None:
        t = PromptTemplate(
            sections=[PromptSection(name="s", description="d", value="hello")],
            negative_prompt="",
        )
        # Empty string is falsy, so it should not appear
        assert "Avoid" not in t.render()


# -- AggregatedMetrics.summary_dict ------------------------------------------


class TestAggregatedMetricsSummaryDict:
    def test_all_keys_present(self) -> None:
        m = AggregatedMetrics(
            dino_similarity_mean=0.1,
            dino_similarity_std=0.2,
            lpips_distance_mean=0.3,
            lpips_distance_std=0.4,
            hps_score_mean=0.5,
            hps_score_std=0.6,
            aesthetics_score_mean=0.7,
            aesthetics_score_std=0.8,
        )
        d = m.summary_dict()
        expected_keys = {
            "dino_similarity_mean",
            "dino_similarity_std",
            "lpips_distance_mean",
            "lpips_distance_std",
            "hps_score_mean",
            "hps_score_std",
            "aesthetics_score_mean",
            "aesthetics_score_std",
        }
        assert set(d.keys()) == expected_keys

    def test_values_match_fields(self) -> None:
        m = AggregatedMetrics(
            dino_similarity_mean=0.85,
            dino_similarity_std=0.02,
            lpips_distance_mean=0.15,
            lpips_distance_std=0.01,
            hps_score_mean=0.42,
            hps_score_std=0.03,
            aesthetics_score_mean=6.5,
            aesthetics_score_std=0.7,
        )
        d = m.summary_dict()
        assert d["dino_similarity_mean"] == 0.85
        assert d["dino_similarity_std"] == 0.02
        assert d["lpips_distance_mean"] == 0.15
        assert d["lpips_distance_std"] == 0.01
        assert d["hps_score_mean"] == 0.42
        assert d["hps_score_std"] == 0.03
        assert d["aesthetics_score_mean"] == 6.5
        assert d["aesthetics_score_std"] == 0.7

    def test_returns_plain_dict(self) -> None:
        m = AggregatedMetrics(
            dino_similarity_mean=0.0,
            dino_similarity_std=0.0,
            lpips_distance_mean=0.0,
            lpips_distance_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        assert isinstance(m.summary_dict(), dict)

    def test_exactly_eight_entries(self) -> None:
        m = AggregatedMetrics(
            dino_similarity_mean=0.0,
            dino_similarity_std=0.0,
            lpips_distance_mean=0.0,
            lpips_distance_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        assert len(m.summary_dict()) == 8


# -- ConvergenceReason --------------------------------------------------------


class TestConvergenceReason:
    def test_enum_values(self) -> None:
        assert ConvergenceReason.MAX_ITERATIONS.value == "max_iterations"
        assert ConvergenceReason.PLATEAU.value == "plateau"
        assert ConvergenceReason.CLAUDE_STOP.value == "claude_stop"

    def test_enum_has_exactly_three_members(self) -> None:
        assert len(ConvergenceReason) == 3

    def test_members_accessible_by_value(self) -> None:
        assert ConvergenceReason("max_iterations") is ConvergenceReason.MAX_ITERATIONS
        assert ConvergenceReason("plateau") is ConvergenceReason.PLATEAU
        assert ConvergenceReason("claude_stop") is ConvergenceReason.CLAUDE_STOP


# -- classify_hypothesis ------------------------------------------------------


class TestClassifyHypothesis:
    def test_color_keywords(self) -> None:
        categories = ["color_palette", "composition", "technique"]
        assert classify_hypothesis("The color palette is too desaturated", categories) == "color_palette"

    def test_composition_keywords(self) -> None:
        categories = ["color_palette", "composition", "technique"]
        assert classify_hypothesis("The framing and layout need to be tighter", categories) == "composition"

    def test_technique_keywords(self) -> None:
        categories = ["color_palette", "composition", "technique"]
        assert classify_hypothesis("The brushwork medium description is too generic", categories) == "technique"

    def test_synonym_matching(self) -> None:
        categories = ["color_palette", "lighting", "texture"]
        assert classify_hypothesis("The shadow contrast is too harsh", categories) == "lighting"
        assert classify_hypothesis("The surface grain detail is missing", categories) == "texture"

    def test_fallback_to_general(self) -> None:
        categories = ["color_palette", "composition"]
        assert classify_hypothesis("Something completely unrelated xyz", categories) == "general"

    def test_empty_text(self) -> None:
        assert classify_hypothesis("", ["color_palette"]) == "general"

    def test_custom_section_names(self) -> None:
        categories = ["characters", "background"]
        assert classify_hypothesis("The characters need more detail", categories) == "characters"


# -- get_category_names -------------------------------------------------------


class TestGetCategoryNames:
    def test_includes_base_categories(self) -> None:
        template = PromptTemplate(sections=[])
        cats = get_category_names(template)
        assert "color_palette" in cats
        assert "composition" in cats
        assert "technique" in cats

    def test_includes_template_sections(self) -> None:
        template = PromptTemplate(
            sections=[
                PromptSection(name="lighting", description="d", value="v"),
                PromptSection(name="characters", description="d", value="v"),
            ]
        )
        cats = get_category_names(template)
        assert "lighting" in cats
        assert "characters" in cats

    def test_no_duplicates(self) -> None:
        template = PromptTemplate(sections=[PromptSection(name="composition", description="d", value="v")])
        cats = get_category_names(template)
        assert cats.count("composition") == 1

    def test_sorted(self) -> None:
        template = PromptTemplate(sections=[])
        cats = get_category_names(template)
        assert cats == sorted(cats)


# -- KnowledgeBase ------------------------------------------------------------


class TestKnowledgeBase:
    def test_add_hypothesis_increments_id(self) -> None:
        kb = KnowledgeBase()
        h1 = kb.add_hypothesis(
            iteration=1,
            parent_id=None,
            statement="test color",
            experiment="add hex",
            category="color_palette",
            kept=True,
            metric_delta={"dino": 0.02},
            lesson="hex codes work",
            confirmed="hex codes work",
            rejected="",
        )
        h2 = kb.add_hypothesis(
            iteration=2,
            parent_id="H1",
            statement="test texture",
            experiment="add grain",
            category="texture",
            kept=False,
            metric_delta={"dino": -0.01},
            lesson="grain too much",
            confirmed="",
            rejected="grain doesn't help",
        )
        assert h1.id == "H1"
        assert h2.id == "H2"
        assert kb.next_id == 3

    def test_confirmed_hypothesis_updates_category(self) -> None:
        kb = KnowledgeBase()
        kb.add_hypothesis(
            iteration=1,
            parent_id=None,
            statement="test color",
            experiment="add hex",
            category="color_palette",
            kept=True,
            metric_delta={"dino": 0.03},
            lesson="hex codes work",
            confirmed="hex codes work",
            rejected="",
        )
        cat = kb.categories["color_palette"]
        assert "hex codes work" in cat.confirmed_insights
        assert cat.best_dino_delta == 0.03
        assert "H1" in cat.hypothesis_ids

    def test_rejected_hypothesis_updates_category(self) -> None:
        kb = KnowledgeBase()
        kb.add_hypothesis(
            iteration=1,
            parent_id=None,
            statement="detailed brushstrokes won't help",
            experiment="elaborate vocab",
            category="technique",
            kept=False,
            metric_delta={"dino": -0.01},
            lesson="too verbose",
            confirmed="",
            rejected="brushstrokes ignored",
        )
        cat = kb.categories["technique"]
        assert any("brushstrokes" in r for r in cat.rejected_approaches)
        assert cat.confirmed_insights == []

    def test_outcome_determination(self) -> None:
        kb = KnowledgeBase()
        h = kb.add_hypothesis(
            iteration=1,
            parent_id=None,
            statement="s",
            experiment="e",
            category="general",
            kept=True,
            metric_delta={},
            lesson="l",
            confirmed="yes",
            rejected="also yes",
        )
        assert h.outcome == "partial"

        h2 = kb.add_hypothesis(
            iteration=2,
            parent_id=None,
            statement="s2",
            experiment="e2",
            category="general",
            kept=False,
            metric_delta={},
            lesson="l",
            confirmed="",
            rejected="no",
        )
        assert h2.outcome == "rejected"

    def test_render_empty_kb(self) -> None:
        kb = KnowledgeBase()
        assert kb.render_for_claude() == ""

    def test_render_includes_hypothesis_chain(self) -> None:
        kb = KnowledgeBase()
        kb.add_hypothesis(
            iteration=1,
            parent_id=None,
            statement="color accuracy gap",
            experiment="add hex codes",
            category="color_palette",
            kept=True,
            metric_delta={"dino": 0.02},
            lesson="hex codes work",
            confirmed="hex codes work",
            rejected="",
        )
        kb.add_hypothesis(
            iteration=2,
            parent_id="H1",
            statement="color temperature adds further",
            experiment="add warm/cool descriptors",
            category="color_palette",
            kept=True,
            metric_delta={"dino": 0.01},
            lesson="temperature helps",
            confirmed="temperature helps",
            rejected="",
        )
        output = kb.render_for_claude()
        assert "H1" in output
        assert "H2" in output
        assert "builds on H1" in output
        assert "Hypothesis Chain" in output
        assert "Per-Category Status" in output
        assert "color_palette" in output

    def test_render_includes_open_problems(self) -> None:
        kb = KnowledgeBase()
        kb.open_problems = [
            OpenProblem(
                text="Color matching on dark palettes",
                category="color_palette",
                priority="HIGH",
                metric_gap=0.15,
                since_iteration=3,
            ),
        ]
        # Need at least one hypothesis for render to produce anything
        kb.add_hypothesis(
            iteration=1,
            parent_id=None,
            statement="s",
            experiment="e",
            category="general",
            kept=True,
            metric_delta={},
            lesson="l",
            confirmed="yes",
            rejected="",
        )
        output = kb.render_for_claude()
        assert "Open Problems" in output
        assert "HIGH" in output
        assert "Color matching on dark palettes" in output

    def test_render_includes_do_not_repeat(self) -> None:
        kb = KnowledgeBase()
        kb.add_hypothesis(
            iteration=1,
            parent_id=None,
            statement="detailed brushstrokes",
            experiment="elaborate vocab",
            category="technique",
            kept=False,
            metric_delta={"dino": -0.01},
            lesson="too verbose",
            confirmed="",
            rejected="brushstrokes ignored",
        )
        output = kb.render_for_claude()
        assert "Do NOT Repeat" in output
        assert "detailed brushstrokes" in output
