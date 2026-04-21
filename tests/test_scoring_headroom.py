"""Unit tests for headroom-weighted composite scoring (A6).

Headroom weighting redirects composite weight away from saturated metrics (those already at or
near their ceiling) and onto metrics with remaining headroom. Saturated axes contribute no
marginal utility to the optimizer, so keeping them weighted drowns out the signal from axes
that can actually move.

Formally: ``headroom_weight(m) = w_m * (1 - s_m)`` with re-normalization across all axes.
A metric at ``s=1.0`` (saturated) contributes zero weight; a metric at ``s=0`` contributes its
full static weight. When all axes are fully saturated the function falls back to
``composite_score`` so rankings remain defined.
"""

from __future__ import annotations

from art_style_search.scoring import composite_score, headroom_composite_score
from art_style_search.types import AggregatedMetrics


def _make_metrics(**overrides: float) -> AggregatedMetrics:
    defaults: dict[str, float | int] = {
        "dreamsim_similarity_mean": 0.5,
        "dreamsim_similarity_std": 0.0,
        "hps_score_mean": 0.175,
        "hps_score_std": 0.0,
        "aesthetics_score_mean": 5.0,
        "aesthetics_score_std": 0.0,
        "color_histogram_mean": 0.5,
        "color_histogram_std": 0.0,
        "ssim_mean": 0.5,
        "style_consistency": 0.5,
        "vision_style": 0.5,
        "vision_subject": 0.5,
        "vision_composition": 0.5,
        "vision_medium": 0.5,
        "vision_proportions": 0.5,
        "megastyle_similarity_mean": 0.5,
        "requested_ref_count": 20,
        "actual_ref_count": 20,
    }
    defaults.update(overrides)
    return AggregatedMetrics(**defaults)  # type: ignore[arg-type]


class TestHeadroomCompositeScore:
    def test_uniform_scores_match_base_composite(self) -> None:
        """When every weighted metric is at the same level, (1 - s) is uniform and headroom == base."""
        m = _make_metrics()
        assert headroom_composite_score(m) == composite_score(m)

    def test_amplifies_ranking_on_unique_headroom_axis(self) -> None:
        """When only one axis has headroom (rest saturated), improvements on that axis get amplified
        in headroom score vs base — that's the point of the reweighting."""
        # All axes saturated except DreamSim; vary DreamSim between candidates.
        sat = {
            "hps_score_mean": 0.35,
            "aesthetics_score_mean": 10.0,
            "color_histogram_mean": 1.0,
            "ssim_mean": 1.0,
            "style_consistency": 1.0,
            "vision_style": 1.0,
            "vision_subject": 1.0,
            "vision_composition": 1.0,
            "vision_medium": 1.0,
            "vision_proportions": 1.0,
            "megastyle_similarity_mean": 1.0,
        }
        a = _make_metrics(dreamsim_similarity_mean=0.5, **sat)
        b = _make_metrics(dreamsim_similarity_mean=0.6, **sat)
        base_delta = composite_score(b) - composite_score(a)
        headroom_delta = headroom_composite_score(b) - headroom_composite_score(a)
        # Base delta is gated by dreamsim's static 34% weight; headroom amplifies because dreamsim
        # is the only axis with weight after redistribution.
        assert headroom_delta > base_delta
        assert headroom_delta > 0

    def test_saturated_metric_weight_redirects_to_movable_ones(self) -> None:
        """DreamSim saturated at 1.0, all others at 0.5 → headroom sees only the 0.5-scoring axes,
        returning the base score of those axes (0.5). This is LESS than composite (0.67) because
        the saturated axis's 'free' 0.34 contribution is explicitly excluded from the headroom lens."""
        m = _make_metrics(dreamsim_similarity_mean=1.0)
        base = composite_score(m)
        headroom = headroom_composite_score(m)
        assert base > headroom  # saturated axis's contribution is excluded from headroom
        # Headroom reflects the weighted average of unsaturated axes at their actual scores.
        # All remaining axes are at 0.5, so headroom ≈ 0.5.
        assert abs(headroom - 0.5) < 0.01

    def test_all_saturated_falls_back_to_composite(self) -> None:
        """Degenerate case: every weighted metric at 1.0 → sum of headroom weights is 0.
        Fall back to ``composite_score`` to keep rankings defined."""
        m = _make_metrics(
            dreamsim_similarity_mean=1.0,
            hps_score_mean=0.35,
            aesthetics_score_mean=10.0,
            color_histogram_mean=1.0,
            ssim_mean=1.0,
            style_consistency=1.0,
            vision_style=1.0,
            vision_subject=1.0,
            vision_composition=1.0,
            vision_medium=1.0,
            vision_proportions=1.0,
            megastyle_similarity_mean=1.0,
        )
        assert headroom_composite_score(m) == composite_score(m)

    def test_floor_clamped_to_zero(self) -> None:
        """Like composite_score, headroom output is floor-clamped at 0.0 regardless of penalty depth."""
        m = _make_metrics(
            dreamsim_similarity_std=0.5,  # huge variance penalty
            color_histogram_std=0.5,
            completion_rate=0.0,  # max completion penalty
        )
        assert headroom_composite_score(m) >= 0.0

    def test_penalties_still_applied(self) -> None:
        """Variance, completion, compliance, subject-floor penalties subtract from headroom output."""
        clean = _make_metrics()
        noisy = _make_metrics(dreamsim_similarity_std=0.3, color_histogram_std=0.3)
        assert headroom_composite_score(noisy) < headroom_composite_score(clean)
