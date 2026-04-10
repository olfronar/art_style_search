"""Unit tests for knowledge.build_caption_diffs and staleness aging."""

from __future__ import annotations

from pathlib import Path

from art_style_search.experiment import ExperimentProposal
from art_style_search.knowledge import build_caption_diffs, update_knowledge_base
from art_style_search.prompt import Lessons
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
