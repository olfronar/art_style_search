"""Unit tests for adaptive_composite_score."""

from __future__ import annotations

from art_style_search.scoring import adaptive_composite_score, composite_score
from art_style_search.types import AggregatedMetrics


def _make_metrics(**overrides: float) -> AggregatedMetrics:
    defaults = {
        "dreamsim_similarity_mean": 0.5,
        "dreamsim_similarity_std": 0.05,
        "hps_score_mean": 0.2,
        "hps_score_std": 0.01,
        "aesthetics_score_mean": 6.0,
        "aesthetics_score_std": 0.3,
    }
    defaults.update(overrides)
    return AggregatedMetrics(**defaults)


class TestAdaptiveCompositeScore:
    def test_single_result_falls_back_to_composite(self) -> None:
        m = _make_metrics()
        assert adaptive_composite_score(m, [m]) == composite_score(m)

    def test_empty_falls_back(self) -> None:
        m = _make_metrics()
        assert adaptive_composite_score(m, []) == composite_score(m)

    def test_best_scores_highest(self) -> None:
        low = _make_metrics(dreamsim_similarity_mean=0.3, color_histogram_mean=0.2)
        mid = _make_metrics(dreamsim_similarity_mean=0.5, color_histogram_mean=0.4)
        high = _make_metrics(dreamsim_similarity_mean=0.8, color_histogram_mean=0.7)
        all_results = [low, mid, high]

        score_low = adaptive_composite_score(low, all_results)
        score_mid = adaptive_composite_score(mid, all_results)
        score_high = adaptive_composite_score(high, all_results)

        assert score_high > score_mid > score_low

    def test_identical_results_falls_back(self) -> None:
        m = _make_metrics()
        # All identical → zero variance → falls back to composite_score
        assert adaptive_composite_score(m, [m, m, m]) == composite_score(m)

    def test_returns_between_zero_and_one(self) -> None:
        a = _make_metrics(dreamsim_similarity_mean=0.3)
        b = _make_metrics(dreamsim_similarity_mean=0.9)
        all_results = [a, b]
        for m in all_results:
            score = adaptive_composite_score(m, all_results)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"
