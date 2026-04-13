"""Unit tests for knowledge.build_caption_diffs and staleness aging."""

from __future__ import annotations

from pathlib import Path

from art_style_search.contracts import ExperimentProposal, Lessons
from art_style_search.knowledge import build_caption_diffs, update_knowledge_base
from art_style_search.types import (
    AggregatedMetrics,
    Caption,
    IterationResult,
    KnowledgeBase,
    OpenProblem,
    PromptTemplate,
)


class TestBuildCaptionDiffs:
    def test_empty_inputs(self) -> None:
        assert build_caption_diffs([], []) == ""
        assert build_caption_diffs([Caption(image_path=Path("a.png"), text="x")], []) == ""
        assert build_caption_diffs([], [Caption(image_path=Path("a.png"), text="x")]) == ""

    def test_unchanged_caption(self) -> None:
        path = Path("img.png")
        prev = [Caption(image_path=path, text="Same caption text")]
        worst = [Caption(image_path=path, text="Same caption text")]
        result = build_caption_diffs(prev, worst)
        assert "UNCHANGED" in result
        assert "img.png" in result

    def test_changed_caption(self) -> None:
        path = Path("img.png")
        prev = [Caption(image_path=path, text="Old description of the image style")]
        worst = [Caption(image_path=path, text="New description of the image style")]
        result = build_caption_diffs(prev, worst)
        assert "PREV:" in result
        assert "NOW:" in result
        assert "img.png" in result

    def test_no_matching_paths(self) -> None:
        prev = [Caption(image_path=Path("a.png"), text="text a")]
        worst = [Caption(image_path=Path("b.png"), text="text b")]
        assert build_caption_diffs(prev, worst) == ""

    def test_mixed_changed_and_unchanged(self) -> None:
        p1, p2 = Path("a.png"), Path("b.png")
        prev = [
            Caption(image_path=p1, text="same"),
            Caption(image_path=p2, text="old text"),
        ]
        worst = [
            Caption(image_path=p1, text="same"),
            Caption(image_path=p2, text="new text"),
        ]
        result = build_caption_diffs(prev, worst)
        assert "UNCHANGED" in result
        assert "PREV:" in result


class TestOpenProblemStaleness:
    """Test that stale open problems get demoted in priority."""

    def _make_kb_with_problems(self, since_iteration: int) -> KnowledgeBase:
        kb = KnowledgeBase()
        kb.open_problems = [
            OpenProblem(
                text="Subject identity collapse",
                category="subject_matter",
                priority="HIGH",
                since_iteration=since_iteration,
            ),
            OpenProblem(
                text="Color flattening", category="color_palette", priority="HIGH", since_iteration=since_iteration
            ),
            OpenProblem(
                text="Minor texture issue", category="texture", priority="MED", since_iteration=since_iteration
            ),
        ]
        return kb

    def test_no_aging_within_5_iterations(self) -> None:

        kb = self._make_kb_with_problems(since_iteration=0)
        template = PromptTemplate()
        agg = AggregatedMetrics(
            dreamsim_similarity_mean=0.5,
            dreamsim_similarity_std=0.05,
            hps_score_mean=0.25,
            hps_score_std=0.02,
            aesthetics_score_mean=6.0,
            aesthetics_score_std=0.5,
        )
        result = IterationResult(
            branch_id=0,
            iteration=5,
            template=template,
            rendered_prompt="",
            image_paths=[],
            per_image_scores=[],
            aggregated=agg,
            claude_analysis="",
            template_changes="",
            kept=False,
            hypothesis="Test hyp",
            experiment="Test exp",
        )
        proposal = ExperimentProposal(
            template=template,
            hypothesis="Test hyp",
            experiment_desc="Test exp",
            builds_on=None,
            open_problems=[],
            lessons=Lessons(),
        )
        update_knowledge_base(kb, result, template, agg, proposal, iteration=5)
        # After 5 iterations, HIGH should still be HIGH
        high_problems = [p for p in kb.open_problems if p.priority == "HIGH"]
        assert len(high_problems) == 2

    def test_high_demoted_to_med_after_5_iterations(self) -> None:

        kb = self._make_kb_with_problems(since_iteration=0)
        template = PromptTemplate()
        agg = AggregatedMetrics(
            dreamsim_similarity_mean=0.5,
            dreamsim_similarity_std=0.05,
            hps_score_mean=0.25,
            hps_score_std=0.02,
            aesthetics_score_mean=6.0,
            aesthetics_score_std=0.5,
        )
        result = IterationResult(
            branch_id=0,
            iteration=6,
            template=template,
            rendered_prompt="",
            image_paths=[],
            per_image_scores=[],
            aggregated=agg,
            claude_analysis="",
            template_changes="",
            kept=False,
            hypothesis="Test hyp",
            experiment="Test exp",
        )
        proposal = ExperimentProposal(
            template=template,
            hypothesis="Test hyp",
            experiment_desc="Test exp",
            builds_on=None,
            open_problems=[],
            lessons=Lessons(),
        )
        update_knowledge_base(kb, result, template, agg, proposal, iteration=6)
        # After 6 iterations, HIGH should be demoted to MED
        high_problems = [p for p in kb.open_problems if p.priority == "HIGH"]
        assert len(high_problems) == 0

    def test_all_demoted_to_low_after_10_iterations(self) -> None:

        kb = self._make_kb_with_problems(since_iteration=0)
        template = PromptTemplate()
        agg = AggregatedMetrics(
            dreamsim_similarity_mean=0.5,
            dreamsim_similarity_std=0.05,
            hps_score_mean=0.25,
            hps_score_std=0.02,
            aesthetics_score_mean=6.0,
            aesthetics_score_std=0.5,
        )
        result = IterationResult(
            branch_id=0,
            iteration=11,
            template=template,
            rendered_prompt="",
            image_paths=[],
            per_image_scores=[],
            aggregated=agg,
            claude_analysis="",
            template_changes="",
            kept=False,
            hypothesis="Test hyp",
            experiment="Test exp",
        )
        proposal = ExperimentProposal(
            template=template,
            hypothesis="Test hyp",
            experiment_desc="Test exp",
            builds_on=None,
            open_problems=[],
            lessons=Lessons(),
        )
        update_knowledge_base(kb, result, template, agg, proposal, iteration=11)
        # After 11 iterations, ALL should be LOW
        low_problems = [p for p in kb.open_problems if p.priority == "LOW"]
        assert len(low_problems) == len(kb.open_problems)


class TestKnowledgeBaseDecisionHandling:
    def _make_result(self, *, kept: bool = False, target_category: str = "") -> IterationResult:
        agg = AggregatedMetrics(
            dreamsim_similarity_mean=0.6,
            dreamsim_similarity_std=0.05,
            hps_score_mean=0.24,
            hps_score_std=0.02,
            aesthetics_score_mean=5.8,
            aesthetics_score_std=0.5,
        )
        return IterationResult(
            branch_id=0,
            iteration=2,
            template=PromptTemplate(),
            rendered_prompt="",
            image_paths=[],
            per_image_scores=[],
            aggregated=agg,
            claude_analysis="",
            template_changes="",
            kept=kept,
            hypothesis="Unclassified hypothesis text",
            experiment="Test experiment",
            target_category=target_category,
        )

    def test_exploration_is_not_recorded_as_confirmed(self) -> None:
        kb = KnowledgeBase()
        proposal = ExperimentProposal(
            template=PromptTemplate(),
            hypothesis="Unclassified hypothesis text",
            experiment_desc="Test experiment",
            builds_on=None,
            open_problems=[],
            lessons=Lessons(new_insight="Maybe useful later"),
            target_category="lighting",
        )

        update_knowledge_base(
            kb,
            self._make_result(kept=True, target_category="lighting"),
            PromptTemplate(),
            None,
            proposal,
            iteration=2,
            decision="exploration",
        )

        hyp = kb.hypotheses[0]
        cat = kb.categories["lighting"]
        assert hyp.outcome == "partial"
        assert cat.confirmed_insights == []
        assert cat.rejected_approaches == []
        assert cat.hypothesis_ids == ["H1"]

    def test_target_category_overrides_keyword_fallback_for_hypothesis_and_problems(self) -> None:
        kb = KnowledgeBase()
        proposal = ExperimentProposal(
            template=PromptTemplate(),
            hypothesis="Completely generic wording",
            experiment_desc="Test experiment",
            builds_on=None,
            open_problems=["Still vague wording"],
            lessons=Lessons(rejected="did not help"),
            target_category="texture",
        )

        update_knowledge_base(
            kb,
            self._make_result(target_category="texture"),
            PromptTemplate(),
            None,
            proposal,
            iteration=1,
            decision="rejected",
        )

        assert kb.hypotheses[0].category == "texture"
        assert kb.open_problems[0].category == "texture"

    def test_near_duplicate_problems_keep_oldest_since_and_strongest_priority(self) -> None:
        kb = KnowledgeBase()
        kb.open_problems = [
            OpenProblem(
                text="Palette defaults still drift toward warm terracotta daylight when the reference is cool, dark, or enclosed.",
                category="color_palette",
                priority="HIGH",
                since_iteration=3,
            )
        ]
        template = PromptTemplate()
        proposal = ExperimentProposal(
            template=template,
            hypothesis="Palette wording tweak",
            experiment_desc="Test experiment",
            builds_on=None,
            open_problems=[
                "[MED] Cool enclosed palettes still drift toward warm terracotta daylight when the reference is dark, cool, or ceiling-dominated."
            ],
            lessons=Lessons(rejected="did not resolve palette drift"),
            target_category="color_palette",
        )

        update_knowledge_base(
            kb,
            self._make_result(target_category="color_palette"),
            template,
            None,
            proposal,
            iteration=8,
            decision="rejected",
        )

        assert len(kb.open_problems) == 1
        problem = kb.open_problems[0]
        assert problem.category == "color_palette"
        assert problem.priority == "HIGH"
        assert problem.since_iteration == 3

    def test_direction_and_mechanism_metadata_persist_on_hypothesis(self) -> None:
        kb = KnowledgeBase()
        proposal = ExperimentProposal(
            template=PromptTemplate(),
            hypothesis="Contrastive subject identity lock",
            experiment_desc="Test experiment",
            builds_on=None,
            open_problems=[],
            lessons=Lessons(confirmed="Contrastive identity cues reduce archetype swaps."),
            target_category="subject_anchor",
            direction_id="D2",
            direction_summary="Character identity disambiguation",
            failure_mechanism="The generator swaps in nearby archetypes when early subject tokens are generic.",
            intervention_type="negative_constraints",
            risk_level="bold",
            expected_primary_metric="vision_subject",
            expected_tradeoff="May over-constrain sparse scenes.",
            changed_sections=["subject_anchor", "scene_geometry"],
            changed_section="subject_anchor",
        )
        result = self._make_result(kept=True, target_category="subject_anchor")
        result.changed_sections = ["subject_anchor", "scene_geometry"]

        update_knowledge_base(
            kb,
            result,
            PromptTemplate(),
            None,
            proposal,
            iteration=4,
            decision="promoted",
        )

        hyp = kb.hypotheses[0]
        cat = kb.categories["subject_anchor"]
        assert hyp.direction_id == "D2"
        assert hyp.direction_summary == "Character identity disambiguation"
        assert hyp.failure_mechanism.startswith("The generator swaps in nearby archetypes")
        assert hyp.intervention_type == "negative_constraints"
        assert hyp.risk_level == "bold"
        assert hyp.expected_primary_metric == "vision_subject"
        assert hyp.expected_tradeoff == "May over-constrain sparse scenes."
        assert hyp.changed_sections == ["subject_anchor", "scene_geometry"]
        assert cat.last_mechanism_tried.startswith("The generator swaps in nearby archetypes")
        assert cat.last_confirmed_mechanism.startswith("The generator swaps in nearby archetypes")
