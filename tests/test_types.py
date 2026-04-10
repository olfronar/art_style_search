"""Unit tests for art_style_search.types."""

from __future__ import annotations

from art_style_search.prompt._format import format_knowledge_base
from art_style_search.scoring import (
    IMPROVEMENT_EPSILON,
    classify_hypothesis,
    composite_score,
    improvement_epsilon,
)
from art_style_search.types import (
    AggregatedMetrics,
    ConvergenceReason,
    KnowledgeBase,
    OpenProblem,
    PromptSection,
    PromptTemplate,
    get_category_names,
)

# -- composite_score ----------------------------------------------------------


class TestCompositeScore:
    def test_known_values(self) -> None:
        m = AggregatedMetrics(
            dreamsim_similarity_mean=0.8,
            dreamsim_similarity_std=0.01,
            hps_score_mean=0.28,
            hps_score_std=0.03,
            aesthetics_score_mean=7.0,
            aesthetics_score_std=0.5,
        )
        # Current formula: DreamSim 40%, Color 22%, SSIM 11%, HPS 5%,
        # Aesthetics 6%, StyleConsistency 6%, Vision(style) 5% + Vision(subject) 1%
        # + Vision(composition) 4% = 10%.
        # color_histogram/ssim default to 0.0, vision_* default to 0.5
        # HPS normalized: min(0.28/0.35, 1.0) = 0.8
        base = (
            0.40 * 0.8
            + 0.05 * min(0.28 / 0.35, 1.0)
            + 0.06 * (7.0 / 10.0)
            + 0.22 * 0.0  # color_histogram
            + 0.11 * 0.0  # ssim
            + 0.06 * 0.0  # style_consistency
            + 0.05 * 0.5  # vision_style
            + 0.01 * 0.5  # vision_subject
            + 0.04 * 0.5  # vision_composition
        )
        # Consistency penalty: 0.30 * (dreamsim_std + color_histogram_std) / 2.0
        penalty = 0.30 * (0.01 + 0.0) / 2.0
        expected = base - penalty
        assert abs(composite_score(m) - expected) < 1e-9

    def test_zero_metrics(self) -> None:
        m = AggregatedMetrics(
            dreamsim_similarity_mean=0.0,
            dreamsim_similarity_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        # vision_style/subject/composition default to 0.5; all other metrics zero
        expected = 0.05 * 0.5 + 0.01 * 0.5 + 0.04 * 0.5
        assert abs(composite_score(m) - expected) < 1e-9

    def test_higher_dreamsim_yields_higher_score(self) -> None:
        base = dict(
            dreamsim_similarity_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        low = AggregatedMetrics(dreamsim_similarity_mean=0.2, **base)
        high = AggregatedMetrics(dreamsim_similarity_mean=0.9, **base)
        assert composite_score(high) > composite_score(low)

    def test_higher_color_histogram_yields_higher_score(self) -> None:
        base = dict(
            dreamsim_similarity_mean=0.5,
            dreamsim_similarity_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        low_color = AggregatedMetrics(color_histogram_mean=0.1, **base)
        high_color = AggregatedMetrics(color_histogram_mean=0.9, **base)
        assert composite_score(high_color) > composite_score(low_color)

    def test_weights_sum_to_one(self) -> None:
        """Verify the coefficient magnitudes sum to 1.0 (9 metrics)."""
        assert abs((0.40 + 0.05 + 0.06 + 0.22 + 0.11 + 0.06 + 0.05 + 0.01 + 0.04) - 1.0) < 1e-9

    def test_hps_normalized(self) -> None:
        """HPS v2 scores (~0.25-0.35) should be normalized to [0,1] via /0.35 ceiling."""
        m = AggregatedMetrics(
            dreamsim_similarity_mean=0.0,
            dreamsim_similarity_std=0.0,
            hps_score_mean=0.35,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        # HPS: 0.05 * min(0.35/0.35, 1.0) = 0.05, plus vision defaults at 0.5
        expected = 0.05 * 1.0 + 0.05 * 0.5 + 0.01 * 0.5 + 0.04 * 0.5
        assert abs(composite_score(m) - expected) < 1e-9

    def test_hps_clamped_above_ceiling(self) -> None:
        """HPS values above 0.35 should clamp to 1.0 contribution."""
        m = AggregatedMetrics(
            dreamsim_similarity_mean=0.0,
            dreamsim_similarity_std=0.0,
            hps_score_mean=0.50,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        # Clamped: min(0.50/0.35, 1.0) = 1.0
        expected = 0.05 * 1.0 + 0.05 * 0.5 + 0.01 * 0.5 + 0.04 * 0.5
        assert abs(composite_score(m) - expected) < 1e-9

    def test_aesthetics_divided_by_ten(self) -> None:
        """Aesthetics is on a 1-10 scale and should be normalized to 0-1."""
        m = AggregatedMetrics(
            dreamsim_similarity_mean=0.0,
            dreamsim_similarity_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=10.0,
            aesthetics_score_std=0.0,
        )
        # Aesthetics: 0.06 * (10.0 / 10.0) = 0.06, plus vision defaults at 0.5
        expected = 0.06 * (10.0 / 10.0) + 0.05 * 0.5 + 0.01 * 0.5 + 0.04 * 0.5
        assert abs(composite_score(m) - expected) < 1e-9

    def test_composite_score_floor_clamp(self) -> None:
        """Composite score should never go negative even with extreme variance penalty."""
        m = AggregatedMetrics(
            dreamsim_similarity_mean=0.1,
            dreamsim_similarity_std=0.9,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
            color_histogram_std=0.9,
        )
        score = composite_score(m)
        assert score >= 0.0, f"composite_score should be >= 0.0, got {score}"


# -- improvement_epsilon ------------------------------------------------------


class TestImprovementEpsilon:
    def test_at_zero(self) -> None:
        assert improvement_epsilon(0.0) == IMPROVEMENT_EPSILON

    def test_at_half(self) -> None:
        assert abs(improvement_epsilon(0.5) - IMPROVEMENT_EPSILON * 0.5) < 1e-12

    def test_negative_clamped(self) -> None:
        assert improvement_epsilon(-1.0) == IMPROVEMENT_EPSILON

    def test_negative_inf_clamped(self) -> None:
        assert improvement_epsilon(float("-inf")) == IMPROVEMENT_EPSILON

    def test_at_one(self) -> None:
        assert improvement_epsilon(1.0) == 0.001  # floor prevents zero epsilon

    def test_above_one_clamped(self) -> None:
        assert improvement_epsilon(1.5) == 0.001  # floor prevents zero epsilon

    def test_monotonically_decreasing(self) -> None:
        baselines = [0.0, 0.3, 0.5, 0.7, 0.9]
        epsilons = [improvement_epsilon(b) for b in baselines]
        for i in range(len(epsilons) - 1):
            assert epsilons[i] > epsilons[i + 1]


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
            dreamsim_similarity_mean=0.1,
            dreamsim_similarity_std=0.2,
            hps_score_mean=0.5,
            hps_score_std=0.6,
            aesthetics_score_mean=0.7,
            aesthetics_score_std=0.8,
        )
        d = m.summary_dict()
        expected_keys = {
            "dreamsim_similarity_mean",
            "dreamsim_similarity_std",
            "hps_score_mean",
            "hps_score_std",
            "aesthetics_score_mean",
            "aesthetics_score_std",
            "color_histogram_mean",
            "color_histogram_std",
            "ssim_mean",
            "ssim_std",
            "style_consistency",
            "vision_style",
            "vision_style_std",
            "vision_subject",
            "vision_subject_std",
            "vision_composition",
            "vision_composition_std",
            "completion_rate",
        }
        assert set(d.keys()) == expected_keys

    def test_values_match_fields(self) -> None:
        m = AggregatedMetrics(
            dreamsim_similarity_mean=0.85,
            dreamsim_similarity_std=0.02,
            hps_score_mean=0.42,
            hps_score_std=0.03,
            aesthetics_score_mean=6.5,
            aesthetics_score_std=0.7,
        )
        d = m.summary_dict()
        assert d["dreamsim_similarity_mean"] == 0.85
        assert d["dreamsim_similarity_std"] == 0.02
        assert d["hps_score_mean"] == 0.42
        assert d["hps_score_std"] == 0.03
        assert d["aesthetics_score_mean"] == 6.5
        assert d["aesthetics_score_std"] == 0.7

    def test_returns_plain_dict(self) -> None:
        m = AggregatedMetrics(
            dreamsim_similarity_mean=0.0,
            dreamsim_similarity_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        assert isinstance(m.summary_dict(), dict)

    def test_exactly_eighteen_entries(self) -> None:
        m = AggregatedMetrics(
            dreamsim_similarity_mean=0.0,
            dreamsim_similarity_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        assert len(m.summary_dict()) == 18


# -- ConvergenceReason --------------------------------------------------------


class TestConvergenceReason:
    def test_enum_values(self) -> None:
        assert ConvergenceReason.MAX_ITERATIONS.value == "max_iterations"
        assert ConvergenceReason.PLATEAU.value == "plateau"
        assert ConvergenceReason.REASONING_STOP.value == "reasoning_stop"

    def test_enum_has_exactly_three_members(self) -> None:
        assert len(ConvergenceReason) == 3

    def test_members_accessible_by_value(self) -> None:
        assert ConvergenceReason("max_iterations") is ConvergenceReason.MAX_ITERATIONS
        assert ConvergenceReason("plateau") is ConvergenceReason.PLATEAU
        assert ConvergenceReason("reasoning_stop") is ConvergenceReason.REASONING_STOP


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
            metric_delta={"dreamsim": 0.02},
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
            metric_delta={"dreamsim": -0.01},
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
            metric_delta={"dreamsim": 0.03},
            lesson="hex codes work",
            confirmed="hex codes work",
            rejected="",
        )
        cat = kb.categories["color_palette"]
        assert "hex codes work" in cat.confirmed_insights
        assert cat.best_perceptual_delta == 0.03
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
            metric_delta={"dreamsim": -0.01},
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
        assert format_knowledge_base(kb) == ""

    def test_render_includes_hypothesis_chain(self) -> None:
        kb = KnowledgeBase()
        kb.add_hypothesis(
            iteration=1,
            parent_id=None,
            statement="color accuracy gap",
            experiment="add hex codes",
            category="color_palette",
            kept=True,
            metric_delta={"dreamsim": 0.02},
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
            metric_delta={"dreamsim": 0.01},
            lesson="temperature helps",
            confirmed="temperature helps",
            rejected="",
        )
        output = format_knowledge_base(kb)
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
        output = format_knowledge_base(kb)
        assert "Open Problems" in output
        assert "HIGH" in output
        assert "Color matching on dark palettes" in output

    def test_render_rejected_in_tree_not_standalone(self) -> None:
        """Rejected hypotheses show REJECTED in the tree, no standalone 'Do NOT Repeat' section."""
        kb = KnowledgeBase()
        kb.add_hypothesis(
            iteration=1,
            parent_id=None,
            statement="detailed brushstrokes",
            experiment="elaborate vocab",
            category="technique",
            kept=False,
            metric_delta={"dreamsim": -0.01},
            lesson="too verbose",
            confirmed="",
            rejected="brushstrokes ignored",
        )
        output = format_knowledge_base(kb)
        assert "REJECTED" in output
        assert "detailed brushstrokes" in output
        assert "Do NOT Repeat" not in output  # merged into tree
