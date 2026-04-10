"""Unit tests for art_style_search.experiment — helper functions."""

from __future__ import annotations

from art_style_search.experiment import (
    _median_metric_scores,
    best_kept_result,
    collect_experiment_results,
)
from tests.test_state import make_iteration_result, make_metric_scores

# ---------------------------------------------------------------------------
# collect_experiment_results
# ---------------------------------------------------------------------------


class TestCollectExperimentResults:
    def test_filters_exceptions(self) -> None:
        good = make_iteration_result(branch_id=0, iteration=1)
        bad = RuntimeError("boom")
        results = collect_experiment_results([good, bad, good], label="test")
        assert len(results) == 2
        assert all(r is good for r in results)

    def test_keeps_all_successes(self) -> None:
        items = [make_iteration_result(branch_id=i, iteration=1) for i in range(3)]
        results = collect_experiment_results(items, label="test")
        assert results == items

    def test_empty_input(self) -> None:
        assert collect_experiment_results([], label="test") == []


# ---------------------------------------------------------------------------
# best_kept_result
# ---------------------------------------------------------------------------


class TestBestKeptResult:
    def test_prefers_kept(self) -> None:
        r0 = make_iteration_result(branch_id=0, iteration=1)
        r0 = r0.__class__(**{**r0.__dict__, "kept": False})
        r1 = make_iteration_result(branch_id=1, iteration=1)
        r1 = r1.__class__(**{**r1.__dict__, "kept": True})
        r2 = make_iteration_result(branch_id=2, iteration=1)
        r2 = r2.__class__(**{**r2.__dict__, "kept": False})

        result = best_kept_result([r0, r1, r2])
        assert result is not None
        assert result.kept is True
        assert result.branch_id == 1

    def test_falls_back_to_first(self) -> None:
        r0 = make_iteration_result(branch_id=0, iteration=1)
        r0 = r0.__class__(**{**r0.__dict__, "kept": False})
        r1 = make_iteration_result(branch_id=1, iteration=1)
        r1 = r1.__class__(**{**r1.__dict__, "kept": False})

        result = best_kept_result([r0, r1])
        assert result is not None
        assert result.branch_id == 0

    def test_empty_returns_none(self) -> None:
        assert best_kept_result([]) is None


# ---------------------------------------------------------------------------
# _median_metric_scores
# ---------------------------------------------------------------------------


class TestMedianMetricScores:
    def test_median_of_three_replicates(self) -> None:
        # 3 replicates, 2 images each — seeds chosen so the median is the middle value
        rep0 = [make_metric_scores(seed=0.0), make_metric_scores(seed=3.0)]
        rep1 = [make_metric_scores(seed=1.0), make_metric_scores(seed=5.0)]
        rep2 = [make_metric_scores(seed=2.0), make_metric_scores(seed=4.0)]

        medians = _median_metric_scores([rep0, rep1, rep2])
        assert len(medians) == 2

        # Image 0: seeds 0, 1, 2 — median is seed=1 values
        expected_img0 = make_metric_scores(seed=1.0)
        assert medians[0].dreamsim_similarity == expected_img0.dreamsim_similarity
        assert medians[0].hps_score == expected_img0.hps_score
        assert medians[0].aesthetics_score == expected_img0.aesthetics_score

        # Image 1: seeds 3, 5, 4 — median is seed=4 values
        expected_img1 = make_metric_scores(seed=4.0)
        assert medians[1].dreamsim_similarity == expected_img1.dreamsim_similarity
        assert medians[1].hps_score == expected_img1.hps_score
        assert medians[1].aesthetics_score == expected_img1.aesthetics_score

    def test_single_replicate_passthrough(self) -> None:
        scores = [make_metric_scores(seed=2.0), make_metric_scores(seed=7.0)]
        medians = _median_metric_scores([scores])
        assert len(medians) == 2

        for original, median in zip(scores, medians, strict=True):
            assert median.dreamsim_similarity == original.dreamsim_similarity
            assert median.hps_score == original.hps_score
            assert median.aesthetics_score == original.aesthetics_score
            assert median.color_histogram == original.color_histogram
            assert median.ssim == original.ssim
            assert median.vision_style == original.vision_style
            assert median.vision_subject == original.vision_subject
            assert median.vision_composition == original.vision_composition
