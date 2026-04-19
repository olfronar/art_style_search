"""Unit tests for A1 paired-replicate promotion gate.

The gate replaces the single-shot ``composite > baseline + epsilon`` check with a
replicate-based rule that requires both:

1. **Hard dominance** — candidate's min composite across replicates ≥ baseline's max composite
   across replicates. Branch-level std ~0.05 vs epsilon ~0.0015 means single-shot noise dwarfs
   the threshold; requiring the worst-case candidate to beat the best-case incumbent cuts SNR
   from ~30x to ~5x with 3 replicates.
2. **Epsilon clearance** — candidate's median composite > baseline's median + epsilon. This
   preserves the adaptive-epsilon semantics; dominance without a meaningful effect size is
   still noise.

Both must hold for promotion. Neither condition alone is sufficient.
"""

from __future__ import annotations

from art_style_search.scoring import replicate_promotion_decision


class TestReplicatePromotionDecision:
    def test_clear_dominance_and_effect_size_promotes(self) -> None:
        """Candidate's min (0.62) beats baseline's max (0.60), and median delta (0.05) > epsilon."""
        candidate = [0.62, 0.63, 0.65]  # min=0.62, median=0.63
        baseline = [0.55, 0.58, 0.60]  # max=0.60, median=0.58
        assert replicate_promotion_decision(candidate, baseline, epsilon=0.002) == "promoted"

    def test_overlapping_distributions_rejects(self) -> None:
        """Candidate's min (0.55) < baseline's max (0.60) → hard dominance fails even if median is higher."""
        candidate = [0.55, 0.65, 0.70]  # min=0.55, median=0.65
        baseline = [0.50, 0.58, 0.60]  # max=0.60, median=0.58
        # Median delta 0.07 clears epsilon easily, but the dominance test fails.
        assert replicate_promotion_decision(candidate, baseline, epsilon=0.002) == "rejected"

    def test_dominant_but_sub_epsilon_rejects(self) -> None:
        """Candidate dominates but median delta < epsilon → effect size fails."""
        candidate = [0.6005, 0.6010, 0.6015]  # min=0.6005, median=0.6010
        baseline = [0.5995, 0.5998, 0.6000]  # max=0.6000, median=0.5998
        # Min (0.6005) > max (0.6000) passes dominance; median delta = 0.0012 < epsilon 0.002.
        assert replicate_promotion_decision(candidate, baseline, epsilon=0.002) == "rejected"

    def test_edge_equality_at_max_min_boundary_rejects(self) -> None:
        """min == max is a tie, not dominance (strict >)."""
        candidate = [0.60, 0.65, 0.70]  # min=0.60
        baseline = [0.50, 0.55, 0.60]  # max=0.60
        # Ties on the dominance boundary should not count as passing dominance.
        assert replicate_promotion_decision(candidate, baseline, epsilon=0.002) == "rejected"

    def test_empty_baseline_uses_effect_size_only(self) -> None:
        """No baseline replicates (first iteration) → only median > epsilon matters."""
        candidate = [0.50, 0.52, 0.55]  # median=0.52
        assert replicate_promotion_decision(candidate, [], epsilon=0.002) == "promoted"

    def test_empty_candidate_rejects(self) -> None:
        """No candidate replicates → cannot promote."""
        assert replicate_promotion_decision([], [0.50, 0.55], epsilon=0.002) == "rejected"

    def test_single_replicate_each_still_works(self) -> None:
        """With n=1 per side the gate degenerates to min > max (candidate > baseline) + epsilon check."""
        candidate = [0.65]
        baseline = [0.60]
        assert replicate_promotion_decision(candidate, baseline, epsilon=0.002) == "promoted"
        # Below-epsilon single-shot delta → rejected
        assert replicate_promotion_decision([0.601], [0.600], epsilon=0.002) == "rejected"
