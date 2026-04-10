"""Tests for per-image alignment, pair reconstruction, and completion tracking."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from art_style_search.report import _per_image_score_for
from art_style_search.scoring import composite_score, per_image_composite
from art_style_search.types import (
    Caption,
    IterationResult,
    MetricScores,
)
from art_style_search.utils import build_ref_gen_pairs
from tests.test_state import make_aggregated_metrics, make_prompt_template


def _make_scores(dreamsim: float) -> MetricScores:
    """Helper: MetricScores with a specific DreamSim value, rest at known defaults."""
    return MetricScores(
        dreamsim_similarity=dreamsim,
        hps_score=0.25,
        aesthetics_score=5.0,
        color_histogram=0.5,
        ssim=0.5,
    )


def _make_result(n: int = 4, dreamsim_values: list[float] | None = None) -> IterationResult:
    """Build an IterationResult with aligned paths/scores/captions for n images."""
    if dreamsim_values is None:
        dreamsim_values = [0.5 + i * 0.05 for i in range(n)]
    assert len(dreamsim_values) == n
    return IterationResult(
        branch_id=0,
        iteration=1,
        template=make_prompt_template(),
        rendered_prompt="test",
        image_paths=[Path(f"/out/{i:02d}.png") for i in range(n)],
        per_image_scores=[_make_scores(ds) for ds in dreamsim_values],
        aggregated=make_aggregated_metrics(),
        claude_analysis="",
        template_changes="",
        kept=True,
        iteration_captions=[
            Caption(image_path=Path(f"/ref/img_{i:03d}.png"), text=f"Caption for image {i}") for i in range(n)
        ],
        n_images_attempted=n,
        n_images_succeeded=n,
    )


# ---------------------------------------------------------------------------
# _per_image_score_for alignment
# ---------------------------------------------------------------------------


class TestPerImageScoreForAlignment:
    def test_score_matches_image_index(self) -> None:
        """per_image_scores[i] should correspond to image with stem i."""
        result = _make_result(5, dreamsim_values=[0.1, 0.2, 0.3, 0.4, 0.5])
        for i in range(5):
            score = _per_image_score_for(result, Path(f"/out/{i:02d}.png"))
            assert score is not None
            assert score.dreamsim_similarity == pytest.approx(0.1 * (i + 1))

    def test_out_of_range_returns_none(self) -> None:
        result = _make_result(3)
        assert _per_image_score_for(result, Path("/out/99.png")) is None

    def test_non_integer_stem_returns_none(self) -> None:
        result = _make_result(3)
        assert _per_image_score_for(result, Path("/out/abc.png")) is None


# ---------------------------------------------------------------------------
# _extract_silent_scores alignment
# ---------------------------------------------------------------------------


class TestExtractSilentScoresAlignment:
    def test_correct_score_for_silent_image(self) -> None:
        """Silent image scores must come from the right index, not a misaligned one."""
        # Lazy import to avoid circular import issues
        from art_style_search.loop import _extract_silent_scores

        result = _make_result(4, dreamsim_values=[0.1, 0.9, 0.2, 0.8])
        # Mark image 1 (DreamSim=0.9) as silent
        silent_set = frozenset({Path("/ref/img_001.png")})
        scores = _extract_silent_scores([result], silent_set)
        assert len(scores) == 1
        # The score should reflect image 1's high DreamSim, not image 0's low DreamSim
        expected = per_image_composite(_make_scores(0.9))
        assert scores[0] == pytest.approx(expected)

    def test_multiple_silent_images(self) -> None:
        from art_style_search.loop import _extract_silent_scores

        result = _make_result(4, dreamsim_values=[0.1, 0.5, 0.3, 0.7])
        silent_set = frozenset({Path("/ref/img_001.png"), Path("/ref/img_003.png")})
        scores = _extract_silent_scores([result], silent_set)
        assert len(scores) == 2
        expected_1 = per_image_composite(_make_scores(0.5))
        expected_3 = per_image_composite(_make_scores(0.7))
        assert sorted(scores) == pytest.approx(sorted([expected_1, expected_3]))


# ---------------------------------------------------------------------------
# Caption diff alignment (zip at loop.py:684)
# ---------------------------------------------------------------------------


class TestCaptionDiffAlignment:
    def test_feedback_filter_preserves_alignment(self) -> None:
        """After filtering by feedback_set, captions still match their scores."""
        result = _make_result(4, dreamsim_values=[0.1, 0.9, 0.2, 0.8])
        feedback_set = frozenset({Path("/ref/img_000.png"), Path("/ref/img_002.png")})

        captions_for_diff = result.iteration_captions
        scores_for_diff = result.per_image_scores
        paired = [
            (c, s) for c, s in zip(captions_for_diff, scores_for_diff, strict=False) if c.image_path in feedback_set
        ]

        # Should get images 0 and 2
        assert len(paired) == 2
        assert paired[0][0].image_path == Path("/ref/img_000.png")
        assert paired[0][1].dreamsim_similarity == pytest.approx(0.1)
        assert paired[1][0].image_path == Path("/ref/img_002.png")
        assert paired[1][1].dreamsim_similarity == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# build_ref_gen_pairs
# ---------------------------------------------------------------------------


class TestBuildRefGenPairs:
    def test_basic_reconstruction(self) -> None:
        result = _make_result(3)
        pairs = build_ref_gen_pairs(result)
        assert len(pairs) == 3
        for i, (ref, gen) in enumerate(pairs):
            assert ref == Path(f"/ref/img_{i:03d}.png")
            assert gen == Path(f"/out/{i:02d}.png")

    def test_with_gaps(self) -> None:
        """Simulate dropped images: stems [00, 02, 05] with captions for all 6."""
        result = IterationResult(
            branch_id=0,
            iteration=1,
            template=make_prompt_template(),
            rendered_prompt="test",
            image_paths=[Path("/out/00.png"), Path("/out/02.png"), Path("/out/05.png")],
            per_image_scores=[_make_scores(0.5)] * 3,
            aggregated=make_aggregated_metrics(),
            claude_analysis="",
            template_changes="",
            kept=True,
            iteration_captions=[
                Caption(image_path=Path(f"/ref/img_{i:03d}.png"), text=f"Caption {i}") for i in range(6)
            ],
        )
        pairs = build_ref_gen_pairs(result)
        assert len(pairs) == 3
        assert pairs[0] == (Path("/ref/img_000.png"), Path("/out/00.png"))
        assert pairs[1] == (Path("/ref/img_002.png"), Path("/out/02.png"))
        assert pairs[2] == (Path("/ref/img_005.png"), Path("/out/05.png"))

    def test_non_integer_stems_skipped(self) -> None:
        result = IterationResult(
            branch_id=0,
            iteration=1,
            template=make_prompt_template(),
            rendered_prompt="test",
            image_paths=[Path("/out/abc.png"), Path("/out/00.png")],
            per_image_scores=[_make_scores(0.5)] * 2,
            aggregated=make_aggregated_metrics(),
            claude_analysis="",
            template_changes="",
            kept=True,
            iteration_captions=[
                Caption(image_path=Path("/ref/img_000.png"), text="Caption 0"),
            ],
        )
        pairs = build_ref_gen_pairs(result)
        assert len(pairs) == 1
        assert pairs[0] == (Path("/ref/img_000.png"), Path("/out/00.png"))

    def test_out_of_range_index_skipped(self) -> None:
        result = IterationResult(
            branch_id=0,
            iteration=1,
            template=make_prompt_template(),
            rendered_prompt="test",
            image_paths=[Path("/out/00.png"), Path("/out/99.png")],
            per_image_scores=[_make_scores(0.5)] * 2,
            aggregated=make_aggregated_metrics(),
            claude_analysis="",
            template_changes="",
            kept=True,
            iteration_captions=[
                Caption(image_path=Path("/ref/img_000.png"), text="Caption 0"),
            ],
        )
        pairs = build_ref_gen_pairs(result)
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# Completion penalty in composite_score
# ---------------------------------------------------------------------------


class TestCompletionPenalty:
    def test_lower_completion_scores_lower(self) -> None:
        base = make_aggregated_metrics()
        full = replace(base, completion_rate=1.0)
        partial = replace(base, completion_rate=0.6)
        assert composite_score(full) > composite_score(partial)

    def test_penalty_proportional(self) -> None:
        base = make_aggregated_metrics()
        rate_90 = replace(base, completion_rate=0.9)
        rate_50 = replace(base, completion_rate=0.5)
        delta_90 = composite_score(replace(base, completion_rate=1.0)) - composite_score(rate_90)
        delta_50 = composite_score(replace(base, completion_rate=1.0)) - composite_score(rate_50)
        assert delta_50 > delta_90
        assert delta_90 == pytest.approx(0.015, abs=1e-6)
        assert delta_50 == pytest.approx(0.075, abs=1e-6)

    def test_full_completion_no_penalty(self) -> None:
        base = make_aggregated_metrics()
        full = replace(base, completion_rate=1.0)
        # Completion penalty should be 0 when rate is 1.0
        no_rate = replace(base)  # default completion_rate=1.0
        assert composite_score(full) == pytest.approx(composite_score(no_rate))


# ---------------------------------------------------------------------------
# Worst-image lookup alignment (prompt/experiments.py:275)
# ---------------------------------------------------------------------------


class TestWorstImageLookup:
    def test_worst_image_gets_correct_caption(self) -> None:
        """The worst-scoring image's caption should match, not a misaligned one."""
        # Image 2 has the worst DreamSim
        result = _make_result(4, dreamsim_values=[0.5, 0.6, 0.1, 0.7])

        # Replicate the logic from prompt/experiments.py:275-286
        idx = min(
            range(len(result.per_image_scores)),
            key=lambda i: result.per_image_scores[i].dreamsim_similarity,
        )
        assert idx == 2  # image 2 is worst
        cap = result.iteration_captions[idx]
        assert cap.image_path == Path("/ref/img_002.png")
        assert "image 2" in cap.text
        assert result.per_image_scores[idx].dreamsim_similarity == pytest.approx(0.1)
