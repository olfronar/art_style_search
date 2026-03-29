"""Unit tests for art_style_search.evaluate.aggregate."""

from __future__ import annotations

import math

from art_style_search.evaluate import aggregate
from art_style_search.types import MetricScores


class TestAggregateEmpty:
    """aggregate with an empty list should return all-zero metrics."""

    def test_returns_zeros(self) -> None:
        result = aggregate([])
        assert result.dino_similarity_mean == 0.0
        assert result.dino_similarity_std == 0.0
        assert result.lpips_distance_mean == 0.0
        assert result.lpips_distance_std == 0.0
        assert result.hps_score_mean == 0.0
        assert result.hps_score_std == 0.0
        assert result.aesthetics_score_mean == 0.0
        assert result.aesthetics_score_std == 0.0


class TestAggregateSingle:
    """aggregate with a single score should return that score as mean, std=0."""

    def test_single_score(self) -> None:
        scores = [MetricScores(dino_similarity=0.8, lpips_distance=0.3, hps_score=0.25, aesthetics_score=6.5)]
        result = aggregate(scores)

        assert result.dino_similarity_mean == 0.8
        assert result.dino_similarity_std == 0.0
        assert result.lpips_distance_mean == 0.3
        assert result.lpips_distance_std == 0.0
        assert result.hps_score_mean == 0.25
        assert result.hps_score_std == 0.0
        assert result.aesthetics_score_mean == 6.5
        assert result.aesthetics_score_std == 0.0


class TestAggregateMultiple:
    """aggregate with multiple scores should compute correct mean and population std."""

    def test_mean_and_std(self) -> None:
        scores = [
            MetricScores(dino_similarity=0.6, lpips_distance=0.4, hps_score=0.20, aesthetics_score=5.0),
            MetricScores(dino_similarity=0.8, lpips_distance=0.2, hps_score=0.30, aesthetics_score=7.0),
            MetricScores(dino_similarity=1.0, lpips_distance=0.6, hps_score=0.10, aesthetics_score=9.0),
        ]
        result = aggregate(scores)

        # Expected means
        assert math.isclose(result.dino_similarity_mean, 0.8, abs_tol=1e-9)
        assert math.isclose(result.lpips_distance_mean, 0.4, abs_tol=1e-9)
        assert math.isclose(result.hps_score_mean, 0.2, abs_tol=1e-9)
        assert math.isclose(result.aesthetics_score_mean, 7.0, abs_tol=1e-9)

        # Expected population std: sqrt(mean((x - mean)^2))
        # dino: values [0.6, 0.8, 1.0], mean=0.8, deviations [-0.2, 0, 0.2], var=0.04/3*2=0.08/3
        expected_dino_std = ((0.2**2 + 0.0**2 + 0.2**2) / 3) ** 0.5
        assert math.isclose(result.dino_similarity_std, expected_dino_std, abs_tol=1e-9)

        # lpips: values [0.4, 0.2, 0.6], mean=0.4, deviations [0, -0.2, 0.2]
        expected_lpips_std = ((0.0**2 + 0.2**2 + 0.2**2) / 3) ** 0.5
        assert math.isclose(result.lpips_distance_std, expected_lpips_std, abs_tol=1e-9)

        # hps: values [0.20, 0.30, 0.10], mean=0.2, deviations [0, 0.1, -0.1]
        expected_hps_std = ((0.0**2 + 0.1**2 + 0.1**2) / 3) ** 0.5
        assert math.isclose(result.hps_score_std, expected_hps_std, abs_tol=1e-9)

        # aesthetics: values [5.0, 7.0, 9.0], mean=7.0, deviations [-2, 0, 2]
        expected_aes_std = ((4.0 + 0.0 + 4.0) / 3) ** 0.5
        assert math.isclose(result.aesthetics_score_std, expected_aes_std, abs_tol=1e-9)
