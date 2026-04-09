"""Composite scoring and hypothesis-category classification.

Split out of ``types.py`` so the pure data layer can stay small.  The
functions here consume ``AggregatedMetrics`` (a dataclass defined in
``types.py``) and read-only category lists, so the import edge only goes
``scoring → types``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from art_style_search.types import AggregatedMetrics
from art_style_search.utils import CATEGORY_SYNONYMS as _CATEGORY_SYNONYMS

_HPS_CEILING = 0.35  # default empirical max for HPS v2 scores; used to normalize to [0, 1]

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
    Aesthetics 6%, StyleConsistency 4%, Vision 4%+4%+4%=12%.  Total = 1.00.
    Includes a consistency penalty based on per-image score variance.
    """
    base = (
        0.40 * m.dreamsim_similarity_mean
        + 0.05 * _normalize_hps(m.hps_score_mean)
        + 0.06 * (m.aesthetics_score_mean / 10.0)
        + 0.22 * m.color_histogram_mean
        + 0.11 * m.ssim_mean
        + 0.04 * m.style_consistency
        + 0.04 * m.vision_style
        + 0.04 * m.vision_subject
        + 0.04 * m.vision_composition
    )
    # Penalize inconsistency: high std across images means unreliable reproduction
    variance_penalty = 0.30 * (m.dreamsim_similarity_std + m.color_histogram_std) / 2.0
    return base - variance_penalty


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
