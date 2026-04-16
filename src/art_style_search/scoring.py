"""Composite scoring and hypothesis-category classification.

Split out of ``types.py`` so the pure data layer can stay small.  The
functions here consume ``AggregatedMetrics`` (a dataclass defined in
``types.py``) and read-only category lists, so the import edge only goes
``scoring → types``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from art_style_search.types import AggregatedMetrics, MetricScores, PromotionTestResult, compliance_components_mean
from art_style_search.utils import CATEGORY_SYNONYMS as _CATEGORY_SYNONYMS

_HPS_CEILING = 0.35  # default empirical max for HPS v2 scores; used to normalize to [0, 1]

# Fixed metric weights for composite scoring.
# Base weights sum to 1.00 for composite_score (experiment-level).
# per_image_composite omits _W_STYLE_CON → its weights sum to 0.96 (max output 0.96).
_W_DREAMSIM = 0.34
_W_HPS = 0.07
_W_AESTHETICS = 0.06
_W_COLOR = 0.17
_W_SSIM = 0.10
_W_STYLE_CON = 0.04
# Vision slice rebalanced to make room for the medium-class + proportions dims.
# The medium verdict diagnoses a root cause of many style misses (2D/3D misclassification)
# and the proportions verdict diagnoses subject-fidelity misses at the anatomy level, so
# weight reductions on vision_style (-0.02) and vision_subject (-0.03) fund them cleanly.
_W_VISION_STYLE = 0.06
_W_VISION_SUBJECT = 0.07
_W_VISION_COMP = 0.04
_W_VISION_MEDIUM = 0.02
_W_VISION_PROPORTIONS = 0.03
_W_VARIANCE_PENALTY = 0.30
_W_COMPLETION_PENALTY = 0.15
_W_COMPLIANCE_PENALTY = 0.08
_W_REF_SHORTFALL_PENALTY = 0.04
_VISION_SUBJECT_FLOOR = 0.35
_W_VISION_SUBJECT_FLOOR_PENALTY = 0.05

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


def compliance_mean(m: AggregatedMetrics) -> float:
    """Unweighted mean of the six compliance components on an AggregatedMetrics."""
    return compliance_components_mean(
        m.compliance_topic_coverage,
        m.compliance_marker_coverage,
        m.section_ordering_rate,
        m.section_balance_rate,
        m.subject_specificity_rate,
        m.style_boilerplate_purity,
    )


# Pairs of (label, attribute) driving `metric_deltas`.  Any metric that appears here must be an
# attribute of AggregatedMetrics and meaningful as a diff vs baseline.
_METRIC_DELTA_ATTRS: tuple[tuple[str, str], ...] = (
    ("dreamsim_similarity_mean", "dreamsim_similarity_mean"),
    ("color_histogram_mean", "color_histogram_mean"),
    ("ssim_mean", "ssim_mean"),
    ("hps_score_mean", "hps_score_mean"),
    ("aesthetics_score_mean", "aesthetics_score_mean"),
    ("vision_style", "vision_style"),
    ("vision_subject", "vision_subject"),
    ("vision_composition", "vision_composition"),
    ("vision_medium", "vision_medium"),
    ("vision_proportions", "vision_proportions"),
    ("style_consistency", "style_consistency"),
    ("completion_rate", "completion_rate"),
)


def metric_deltas(metrics: AggregatedMetrics, baseline: AggregatedMetrics) -> dict[str, float]:
    """Per-metric diffs `metrics - baseline` for every tracked metric plus compliance."""
    deltas = {label: getattr(metrics, attr) - getattr(baseline, attr) for label, attr in _METRIC_DELTA_ATTRS}
    deltas["compliance"] = compliance_mean(metrics) - compliance_mean(baseline)
    return deltas


def _normalize_hps(raw: float, ceiling: float = _HPS_CEILING) -> float:
    """Normalize raw HPS v2 score to [0, 1] using the empirical ceiling."""
    return min(raw / ceiling, 1.0)


def composite_score(m: AggregatedMetrics) -> float:
    """Fixed-weight composite score used for absolute quality comparison.

    All metrics normalized to ~[0, 1] before weighting.
    Weights: DreamSim 34%, HPS 7%, Aesthetics 6%, Color 17%, SSIM 10%,
    StyleConsistency 4%, Vision(style) 6%, Vision(subject) 7%,
    Vision(composition) 4%, Vision(medium) 2%, Vision(proportions) 3%.
    Total = 1.00.
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
        + _W_VISION_MEDIUM * m.vision_medium
        + _W_VISION_PROPORTIONS * m.vision_proportions
    )
    # Penalize inconsistency: high std across images means unreliable reproduction
    variance_penalty = _W_VARIANCE_PENALTY * (m.dreamsim_similarity_std + m.color_histogram_std) / 2.0
    # Penalize incomplete experiments: missing images should not inflate scores
    completion_penalty = (1.0 - m.completion_rate) * _W_COMPLETION_PENALTY
    compliance_score = compliance_components_mean(
        m.compliance_topic_coverage,
        m.compliance_marker_coverage,
        m.section_ordering_rate,
        m.section_balance_rate,
        m.subject_specificity_rate,
        m.style_boilerplate_purity,
    )
    compliance_penalty = (1.0 - compliance_score) * _W_COMPLIANCE_PENALTY

    ref_shortfall = 0.0
    if m.requested_ref_count > 0:
        ref_shortfall = max(m.requested_ref_count - m.actual_ref_count, 0) / m.requested_ref_count
    ref_shortfall_penalty = ref_shortfall * _W_REF_SHORTFALL_PENALTY
    subject_floor_penalty = 0.0
    if m.vision_subject < _VISION_SUBJECT_FLOOR:
        shortfall = (_VISION_SUBJECT_FLOOR - m.vision_subject) / _VISION_SUBJECT_FLOOR
        subject_floor_penalty = shortfall * _W_VISION_SUBJECT_FLOOR_PENALTY
    return max(
        0.0,
        base
        - variance_penalty
        - completion_penalty
        - compliance_penalty
        - ref_shortfall_penalty
        - subject_floor_penalty,
    )


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
        lambda r: r.vision_medium,
        lambda r: r.vision_proportions,
        lambda r: compliance_components_mean(
            r.compliance_topic_coverage,
            r.compliance_marker_coverage,
            r.section_ordering_rate,
            r.section_balance_rate,
            r.subject_specificity_rate,
            r.style_boilerplate_purity,
        ),
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
    Omits ``_W_STYLE_CON`` (style consistency is experiment-level), so max output is 0.96.
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
        + _W_VISION_MEDIUM * s.vision_medium
        + _W_VISION_PROPORTIONS * s.vision_proportions
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
