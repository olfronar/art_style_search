"""Unit tests for hypothesis diversity mechanisms."""

from __future__ import annotations

from art_style_search.contracts import Lessons, RefinementResult
from art_style_search.prompt._format import suggest_target_categories
from art_style_search.prompt.experiments import enforce_hypothesis_diversity, select_experiment_portfolio
from art_style_search.types import CategoryProgress, KnowledgeBase
from tests.conftest import make_prompt_template


def _make_refinement(
    hypothesis: str,
    target_category: str = "",
    *,
    direction_id: str = "",
    failure_mechanism: str = "",
    intervention_type: str = "",
    risk_level: str = "targeted",
    changed_sections: list[str] | None = None,
) -> RefinementResult:
    return RefinementResult(
        template=make_prompt_template(),
        analysis="test analysis",
        template_changes="test changes",
        should_stop=False,
        hypothesis=hypothesis,
        experiment="test experiment",
        lessons=Lessons(),
        builds_on="",
        open_problems=[],
        changed_section=(changed_sections or [""])[0],
        changed_sections=changed_sections or [],
        target_category=target_category,
        direction_id=direction_id,
        failure_mechanism=failure_mechanism,
        intervention_type=intervention_type,
        risk_level=risk_level,
    )


# ---------------------------------------------------------------------------
# enforce_hypothesis_diversity
# ---------------------------------------------------------------------------


class TestEnforceHypothesisDiversity:
    def test_keeps_diverse_experiments(self) -> None:
        results = [
            _make_refinement("Improve color palette warmth", target_category="color"),
            _make_refinement("Add more texture detail", target_category="texture"),
            _make_refinement("Fix composition balance", target_category="composition"),
        ]
        filtered = enforce_hypothesis_diversity(results, make_prompt_template())
        assert len(filtered) == 3

    def test_drops_duplicate_mechanism_within_category(self) -> None:
        results = [
            _make_refinement(
                "Improve warm tones",
                target_category="color",
                failure_mechanism="Color fields are too diffuse.",
                intervention_type="information_priority",
            ),
            _make_refinement(
                "Fix cool tones",
                target_category="color",
                failure_mechanism="Color fields are too diffuse.",
                intervention_type="information_priority",
            ),
            _make_refinement("Better texture", target_category="texture"),
        ]
        filtered = enforce_hypothesis_diversity(results, make_prompt_template())
        assert len(filtered) == 2
        assert filtered[0].hypothesis == "Improve warm tones"
        assert filtered[1].hypothesis == "Better texture"

    def test_keeps_same_category_when_mechanism_differs(self) -> None:
        results = [
            _make_refinement(
                "Adjust the color palette for warmth",
                target_category="color",
                failure_mechanism="The palette hierarchy is too flat.",
                intervention_type="information_priority",
            ),
            _make_refinement(
                "Change colors to be more vivid",
                target_category="color",
                failure_mechanism="The generator misreads hue ownership on costumes.",
                intervention_type="negative_constraints",
            ),
        ]
        filtered = enforce_hypothesis_diversity(results, make_prompt_template())
        assert len(filtered) == 2

    def test_empty_input(self) -> None:
        filtered = enforce_hypothesis_diversity([], make_prompt_template())
        assert filtered == []

    def test_single_experiment(self) -> None:
        results = [_make_refinement("Improve lighting", target_category="lighting")]
        filtered = enforce_hypothesis_diversity(results, make_prompt_template())
        assert len(filtered) == 1


class TestSelectExperimentPortfolio:
    def test_takes_one_targeted_proposal_per_direction_first(self) -> None:
        results = [
            _make_refinement(
                "Dir1 targeted",
                target_category="subject_anchor",
                direction_id="D1",
                failure_mechanism="subject confusion",
                intervention_type="information_priority",
                risk_level="targeted",
                changed_sections=["subject_anchor"],
            ),
            _make_refinement(
                "Dir1 bold",
                target_category="subject_anchor",
                direction_id="D1",
                failure_mechanism="subject confusion",
                intervention_type="scene_type_split",
                risk_level="bold",
                changed_sections=["subject_anchor", "scene_geometry"],
            ),
            _make_refinement(
                "Dir2 targeted",
                target_category="composition",
                direction_id="D2",
                failure_mechanism="layout drift",
                intervention_type="information_priority",
                risk_level="targeted",
                changed_sections=["composition_blueprint"],
            ),
            _make_refinement(
                "Dir2 bold",
                target_category="composition",
                direction_id="D2",
                failure_mechanism="layout drift",
                intervention_type="section_schema",
                risk_level="bold",
                changed_sections=["composition_blueprint", "environment_staging"],
            ),
            _make_refinement(
                "Dir3 targeted",
                target_category="lighting",
                direction_id="D3",
                failure_mechanism="temperature drift",
                intervention_type="information_priority",
                risk_level="targeted",
                changed_sections=["lighting_rendering"],
            ),
            _make_refinement(
                "Dir3 bold",
                target_category="lighting",
                direction_id="D3",
                failure_mechanism="temperature drift",
                intervention_type="section_schema",
                risk_level="bold",
                changed_sections=["lighting_rendering", "mood_atmosphere"],
            ),
        ]

        selected = select_experiment_portfolio(results, num_experiments=5, num_directions=3)

        assert [r.hypothesis for r in selected[:3]] == ["Dir1 targeted", "Dir2 targeted", "Dir3 targeted"]
        assert [r.hypothesis for r in selected[3:]] == ["Dir1 bold", "Dir2 bold"]

    def test_allows_same_category_across_directions(self) -> None:
        results = [
            _make_refinement(
                "D1 targeted",
                target_category="subject_anchor",
                direction_id="D1",
                failure_mechanism="archetype swaps",
                intervention_type="negative_constraints",
                risk_level="targeted",
                changed_sections=["subject_anchor"],
            ),
            _make_refinement(
                "D2 targeted",
                target_category="subject_anchor",
                direction_id="D2",
                failure_mechanism="salience budgeting",
                intervention_type="information_priority",
                risk_level="targeted",
                changed_sections=["subject_anchor"],
            ),
            _make_refinement(
                "D3 targeted",
                target_category="subject_anchor",
                direction_id="D3",
                failure_mechanism="support relation drift",
                intervention_type="contact_lock",
                risk_level="targeted",
                changed_sections=["subject_anchor"],
            ),
        ]

        selected = select_experiment_portfolio(results, num_experiments=3, num_directions=3)
        assert [r.direction_id for r in selected] == ["D1", "D2", "D3"]


# ---------------------------------------------------------------------------
# suggest_target_categories
# ---------------------------------------------------------------------------


class TestSuggestTargetCategories:
    def test_unexplored_categories_ranked_highest(self) -> None:
        kb = KnowledgeBase()
        categories = ["color", "texture", "composition", "lighting"]
        result = suggest_target_categories(kb, 4, categories)
        # All unexplored → all get 1.0 score, order preserved
        assert set(result) == set(categories)

    def test_diminishing_returns_ranked_lowest(self) -> None:
        kb = KnowledgeBase()
        # "color" has 3 rejections, 0 confirmed
        kb.categories["color"] = CategoryProgress(
            category="color",
            hypothesis_ids=["H1", "H2", "H3"],
            rejected_approaches=["a", "b", "c"],
        )
        categories = ["color", "texture"]
        result = suggest_target_categories(kb, 2, categories)
        # "texture" (unexplored=1.0) should rank above "color" (diminishing=0.1)
        assert result[0] == "texture"
        assert result[1] == "color"

    def test_partial_success_ranked_mid(self) -> None:
        kb = KnowledgeBase()
        kb.categories["texture"] = CategoryProgress(
            category="texture",
            hypothesis_ids=["H1"],
            confirmed_insights=["use dry brush"],
            best_perceptual_delta=0.02,
        )
        categories = ["texture", "lighting"]
        result = suggest_target_categories(kb, 2, categories)
        # "lighting" (unexplored=1.0) > "texture" (partial=0.7)
        assert result[0] == "lighting"
        assert result[1] == "texture"

    def test_num_targets_limits_output(self) -> None:
        kb = KnowledgeBase()
        categories = ["a", "b", "c", "d", "e"]
        result = suggest_target_categories(kb, 3, categories)
        assert len(result) == 3

    def test_empty_categories(self) -> None:
        kb = KnowledgeBase()
        result = suggest_target_categories(kb, 5, [])
        assert result == []
