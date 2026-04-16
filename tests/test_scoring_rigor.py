"""Unit tests for per_image_composite and paired_promotion_test in scoring.py."""

from __future__ import annotations

from art_style_search.scoring import paired_promotion_test, per_image_composite
from art_style_search.types import MetricScores


def _make_scores(
    ds: float = 0.7,
    hps: float = 0.25,
    aes: float = 6.0,
    color: float = 0.5,
    ssim: float = 0.5,
    vs: float = 0.5,
    vsu: float = 0.5,
    vc: float = 0.5,
    vm: float = 0.5,
    vp: float = 0.5,
) -> MetricScores:
    return MetricScores(
        dreamsim_similarity=ds,
        hps_score=hps,
        aesthetics_score=aes,
        color_histogram=color,
        ssim=ssim,
        vision_style=vs,
        vision_subject=vsu,
        vision_composition=vc,
        vision_medium=vm,
        vision_proportions=vp,
    )


# ---------------------------------------------------------------------------
# TestPerImageComposite
# ---------------------------------------------------------------------------


class TestPerImageComposite:
    def test_weights_sum_to_roughly_one(self) -> None:
        """Perfect scores (all 1.0, aesthetics 10.0) should give close to 1.0.

        The weights are:
          0.34 (dreamsim) + 0.07 (hps) + 0.06 (aes/10) + 0.17 (color) +
          0.10 (ssim) + 0.06 (vs) + 0.07 (vsu) + 0.04 (vc) + 0.02 (vm) + 0.03 (vp) = 0.96
        HPS normalizes as min(raw / 0.35, 1.0), so hps=0.35 -> 1.0.
        Aesthetics normalizes as aes/10, so aes=10 -> 1.0.
        Total with perfect inputs = 0.96 (per_image_composite omits _W_STYLE_CON).
        """
        perfect = _make_scores(ds=1.0, hps=0.35, aes=10.0, color=1.0, ssim=1.0, vs=1.0, vsu=1.0, vc=1.0, vm=1.0, vp=1.0)
        score = per_image_composite(perfect)
        # The weights sum to 0.96, but close to 1.0
        assert 0.90 <= score <= 1.0, f"Expected ~0.96 for perfect scores, got {score}"

    def test_zero_scores_give_zero(self) -> None:
        zeros = _make_scores(ds=0.0, hps=0.0, aes=0.0, color=0.0, ssim=0.0, vs=0.0, vsu=0.0, vc=0.0, vm=0.0, vp=0.0)
        assert per_image_composite(zeros) == 0.0


# ---------------------------------------------------------------------------
# TestPairedPromotionTest
# ---------------------------------------------------------------------------


class TestPairedPromotionTest:
    def test_significant_improvement(self) -> None:
        """Candidate clearly better on DreamSim across 20 paired observations -> passed=True."""
        candidate = [_make_scores(ds=0.8 + i * 0.005) for i in range(20)]
        incumbent = [_make_scores(ds=0.6 + i * 0.005) for i in range(20)]
        result = paired_promotion_test(candidate, incumbent)
        assert result.passed is True
        assert result.effect_size > 0
        assert result.p_value < 0.10
        assert result.ci_lower > 0  # entire CI above zero

    def test_no_improvement(self) -> None:
        """Identical scores -> effect_size ~0, passed=False."""
        same = [_make_scores() for _ in range(20)]
        result = paired_promotion_test(same, same)
        assert result.passed is False
        assert abs(result.effect_size) < 1e-9

    def test_regression(self) -> None:
        """Candidate worse than incumbent -> passed=False."""
        candidate = [_make_scores(ds=0.4 + i * 0.005) for i in range(20)]
        incumbent = [_make_scores(ds=0.7 + i * 0.005) for i in range(20)]
        result = paired_promotion_test(candidate, incumbent)
        assert result.passed is False
        assert result.effect_size < 0

    def test_noisy_small_improvement(self) -> None:
        """Candidate slightly better on some metrics, worse on others.

        The test may or may not pass, but the result should be structurally valid.
        """
        candidate = [_make_scores(ds=0.7 + i * 0.002, color=0.5 - i * 0.001, ssim=0.5 + i * 0.001) for i in range(20)]
        incumbent = [
            _make_scores(ds=0.69 + i * 0.002, color=0.51 - i * 0.001, ssim=0.49 + i * 0.001) for i in range(20)
        ]
        result = paired_promotion_test(candidate, incumbent)
        # Structurally valid regardless of pass/fail
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.p_value <= 1.0
        assert result.ci_lower <= result.ci_upper
