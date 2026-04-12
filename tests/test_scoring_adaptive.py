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
        "compliance_topic_coverage": 1.0,
        "compliance_marker_coverage": 1.0,
        "section_ordering_rate": 1.0,
        "section_balance_rate": 1.0,
        "requested_ref_count": 20,
        "actual_ref_count": 20,
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

    def test_penalizes_compliance_and_reference_shortfall(self) -> None:
        complete = _make_metrics()
        incomplete = _make_metrics(
            compliance_topic_coverage=0.25,
            compliance_marker_coverage=0.25,
            section_ordering_rate=0.0,
            section_balance_rate=0.0,
            actual_ref_count=12,
        )

        assert composite_score(complete) > composite_score(incomplete)

    def test_subject_and_style_fidelity_can_outweigh_small_similarity_gain(self) -> None:
        high_similarity_low_fidelity = _make_metrics(
            dreamsim_similarity_mean=0.60,
            color_histogram_mean=0.55,
            ssim_mean=0.55,
            vision_style=0.0,
            vision_subject=0.0,
        )
        lower_similarity_but_faithful = _make_metrics(
            dreamsim_similarity_mean=0.55,
            color_histogram_mean=0.50,
            ssim_mean=0.50,
            vision_style=0.5,
            vision_subject=0.5,
        )

        assert composite_score(lower_similarity_but_faithful) > composite_score(high_similarity_low_fidelity)
