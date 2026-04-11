"""Unit tests for hypothesis diversity mechanisms."""

from __future__ import annotations

from art_style_search.prompt._format import suggest_target_categories
from art_style_search.prompt._parse import Lessons, RefinementResult
from art_style_search.prompt.experiments import enforce_hypothesis_diversity
from art_style_search.types import CategoryProgress, KnowledgeBase
from tests.conftest import make_prompt_template


def _make_refinement(hypothesis: str, target_category: str = "") -> RefinementResult:
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
        changed_section="",
        target_category=target_category,
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

    def test_drops_duplicate_categories(self) -> None:
        results = [
            _make_refinement("Improve warm tones", target_category="color"),
            _make_refinement("Fix cool tones", target_category="color"),
            _make_refinement("Better texture", target_category="texture"),
        ]
        filtered = enforce_hypothesis_diversity(results, make_prompt_template())
        assert len(filtered) == 2
        assert filtered[0].hypothesis == "Improve warm tones"
        assert filtered[1].hypothesis == "Better texture"

    def test_falls_back_to_keyword_classification(self) -> None:
        results = [
            _make_refinement("Adjust the color palette for warmth"),
            _make_refinement("Change colors to be more vivid"),
        ]
        filtered = enforce_hypothesis_diversity(results, make_prompt_template())
        # Both should classify as "color" by keywords → second should be dropped
        assert len(filtered) == 1

    def test_empty_input(self) -> None:
        filtered = enforce_hypothesis_diversity([], make_prompt_template())
        assert filtered == []

    def test_single_experiment(self) -> None:
        results = [_make_refinement("Improve lighting", target_category="lighting")]
        filtered = enforce_hypothesis_diversity(results, make_prompt_template())
        assert len(filtered) == 1


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
