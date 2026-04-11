"""Composite scoring and hypothesis-category classification.

Split out of ``types.py`` so the pure data layer can stay small.  The
functions here consume ``AggregatedMetrics`` (a dataclass defined in
``types.py``) and read-only category lists, so the import edge only goes
``scoring → types``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from art_style_search.types import AggregatedMetrics, MetricScores, PromotionTestResult
from art_style_search.utils import CATEGORY_SYNONYMS as _CATEGORY_SYNONYMS

_HPS_CEILING = 0.35  # default empirical max for HPS v2 scores; used to normalize to [0, 1]

# Fixed metric weights for composite scoring.
# Base weights sum to 1.00 for composite_score (experiment-level).
# per_image_composite omits _W_STYLE_CON → its weights sum to 0.94 (max output 0.94).
_W_DREAMSIM = 0.40
_W_HPS = 0.05
_W_AESTHETICS = 0.06
_W_COLOR = 0.22
_W_SSIM = 0.11
_W_STYLE_CON = 0.06
_W_VISION_STYLE = 0.05
_W_VISION_SUBJECT = 0.01
_W_VISION_COMP = 0.04
_W_VARIANCE_PENALTY = 0.30
_W_COMPLETION_PENALTY = 0.15

# Improvement must exceed this threshold to be accepted (filters generation noise)
IMPROVEMENT_EPSILON = 0.005
_EPSILON_FLOOR = 0.001  # Minimum epsilon to prevent false-positive improvements at high baselines


def improvement_epsilon(baseline: float) -> float:
    """Threshold that shrinks as baseline score climbs, with a minimum floor.

    At baseline 0.45 → ~0.00275. At 0.51 → ~0.00245. At 0.90 → 0.001 (floor).
    When baseline is -inf (no prior metrics), falls back to IMPROVEMENT_EPSILON.
    """
    clamped = min(max(baseline, 0.0), 1.0)
    return max(IMPROVEMENT_EPSILON * (1.0 - clamped), _EPSILON_FLOOR)


def _normalize_hps(raw: float, ceiling: float = _HPS_CEILING) -> float:
    """Normalize raw HPS v2 score to [0, 1] using the empirical ceiling."""
    return min(raw / ceiling, 1.0)


def composite_score(m: AggregatedMetrics) -> float:
    """Fixed-weight composite score used for absolute quality comparison.

    All metrics normalized to ~[0, 1] before weighting.
    Weights: DreamSim 40%, Color 22%, SSIM 11%, HPS 5%,
    Aesthetics 6%, StyleConsistency 6%, Vision(style) 5% + Vision(subject) 1%
    + Vision(composition) 4% = 10%.  Total = 1.00.
    Includes a consistency penalty based on per-image score variance.
    """
    base = (
        _W_DREAMSIM * m.dreamsim_similarity_mean
        + _W_HPS * _normalize_hps(m.hps_score_mean)
        + _W_AESTHETICS * (m.aesthetics_score_mean / 10.0)
        + _W_COLOR * m.color_histogram_mean
        + _W_SSIM * m.ssim_mean
        + _W_STYLE_CON * m.style_consistency
        + _W_VISION_STYLE * m.vision_style
        + _W_VISION_SUBJECT * m.vision_subject
        + _W_VISION_COMP * m.vision_composition
    )
    # Penalize inconsistency: high std across images means unreliable reproduction
    variance_penalty = _W_VARIANCE_PENALTY * (m.dreamsim_similarity_std + m.color_histogram_std) / 2.0
    # Penalize incomplete experiments: missing images should not inflate scores
    completion_penalty = (1.0 - m.completion_rate) * _W_COMPLETION_PENALTY
    return max(0.0, base - variance_penalty - completion_penalty)


def adaptive_composite_score(
    target: AggregatedMetrics,
    all_results: list[AggregatedMetrics],
) -> float:
    """Score with adaptive weights proportional to cross-experiment variance.

    Metrics that differentiate between experiments get higher weight.
    Falls back to fixed weights for single-experiment evaluation.
    """
    if len(all_results) < 2:
        return composite_score(target)

    # All metrics are higher=better after normalization
    metric_extractors: list[Callable[[AggregatedMetrics], float]] = [
        lambda r: r.dreamsim_similarity_mean,
        lambda r: _normalize_hps(r.hps_score_mean),
        lambda r: r.aesthetics_score_mean / 10.0,
        lambda r: r.color_histogram_mean,
        lambda r: r.ssim_mean,
        lambda r: r.style_consistency,
        lambda r: r.vision_style,
        lambda r: r.vision_subject,
        lambda r: r.vision_composition,
    ]

    weighted_sum = 0.0
    total_weight = 0.0

    for extractor in metric_extractors:
        values = [extractor(r) for r in all_results]
        target_val = extractor(target)

        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

        if std < 1e-8:
            continue

        vmin, vmax = min(values), max(values)
        rng = vmax - vmin
        if rng < 1e-8:
            continue
        normalized = (target_val - vmin) / rng

        weighted_sum += std * normalized
        total_weight += std

    if total_weight < 1e-8:
        return composite_score(target)

    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Hypothesis category classification
# ---------------------------------------------------------------------------

# Synonym map for category auto-classification — defined in utils.py so it can
# be shared with types.get_category_names and loop._should_honor_stop without
# creating a scoring ↔ types import cycle.


def classify_hypothesis(text: str, categories: list[str]) -> str:
    """Auto-classify a hypothesis into a category via keyword matching."""
    text_lower = text.lower()
    best_cat = "general"
    best_score = 0
    for cat in categories:
        score = 0
        # Match words from the category name itself
        for word in cat.replace("_", " ").split():
            if word in text_lower:
                score += 1
        # Match synonyms
        for synonym in _CATEGORY_SYNONYMS.get(cat, []):
            if synonym in text_lower:
                score += 1
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat if best_score > 0 else "general"


# ---------------------------------------------------------------------------
# Per-image composite score (for statistical testing)
# ---------------------------------------------------------------------------


def per_image_composite(s: MetricScores) -> float:
    """Compute a per-image composite score using the same weights as ``composite_score``.

    Unlike ``composite_score`` (which operates on aggregated means), this computes
    the score for a single image — no variance penalty since there's only one observation.
    Omits ``_W_STYLE_CON`` (style consistency is experiment-level), so max output is 0.94.
    """
    return (
        _W_DREAMSIM * s.dreamsim_similarity
        + _W_HPS * _normalize_hps(s.hps_score)
        + _W_AESTHETICS * (s.aesthetics_score / 10.0)
        + _W_COLOR * s.color_histogram
        + _W_SSIM * s.ssim
        + _W_VISION_STYLE * s.vision_style
        + _W_VISION_SUBJECT * s.vision_subject
        + _W_VISION_COMP * s.vision_composition
    )


# ---------------------------------------------------------------------------
# Statistical testing for promotion decisions (rigorous mode)
# ---------------------------------------------------------------------------

_PROMOTION_ALPHA = 0.10  # relaxed threshold for internal trustworthiness
_BOOTSTRAP_RESAMPLES = 2000


def paired_promotion_test(
    candidate_scores: list[MetricScores],
    incumbent_scores: list[MetricScores],
) -> PromotionTestResult:
    """Wilcoxon signed-rank test on per-image composite scores.

    Returns a ``PromotionTestResult`` with one-sided p-value (H1: candidate > incumbent),
    effect size (mean paired difference), and bootstrap 95% CI on the mean difference.
    Falls back to a sign test when Wilcoxon assumptions are violated (too many ties).

    Note: uses ``per_image_composite`` which has max 0.94 (no style_consistency term),
    so paired differences are on a [0, 0.94] scale rather than [0, 1.0].
    """
    import numpy as np
    from scipy import stats as sp_stats

    n = min(len(candidate_scores), len(incumbent_scores))
    diffs = np.array(
        [per_image_composite(candidate_scores[i]) - per_image_composite(incumbent_scores[i]) for i in range(n)]
    )

    effect_size = float(np.mean(diffs))

    # Bootstrap 95% CI on mean difference
    rng = np.random.default_rng(42)
    boot_means = np.array(
        [float(np.mean(rng.choice(diffs, size=n, replace=True))) for _ in range(_BOOTSTRAP_RESAMPLES)]
    )
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Wilcoxon signed-rank test (one-sided: candidate > incumbent)
    try:
        stat_result = sp_stats.wilcoxon(diffs, alternative="greater")
        statistic = float(stat_result.statistic)
        p_value = float(stat_result.pvalue)
    except ValueError:
        # Falls back to sign test if all differences are zero or too many ties
        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        n_nonzero = n_pos + n_neg
        if n_nonzero == 0:
            return PromotionTestResult(
                statistic=0.0,
                p_value=1.0,
                effect_size=effect_size,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                passed=False,
            )
        sign_result = sp_stats.binomtest(n_pos, n_nonzero, 0.5, alternative="greater")
        statistic = float(n_pos)
        p_value = float(sign_result.pvalue)

    passed = p_value < _PROMOTION_ALPHA and effect_size > 0
    return PromotionTestResult(
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        passed=passed,
    )
