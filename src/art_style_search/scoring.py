"""Composite scoring and hypothesis-category classification.

Split out of ``types.py`` so the pure data layer can stay small.  The
functions here consume ``AggregatedMetrics`` (a dataclass defined in
``types.py``) and read-only category lists, so the import edge only goes
``scoring → types``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from art_style_search.types import AggregatedMetrics, MetricScores, compliance_components_mean
from art_style_search.utils import CATEGORY_SYNONYMS as _CATEGORY_SYNONYMS

_HPS_CEILING = 0.35  # default empirical max for HPS v2 scores; used to normalize to [0, 1]

# Fixed metric weights for composite scoring.
# Base weights sum to 1.00 for composite_score (experiment-level).
# per_image_composite omits _W_STYLE_CON (experiment-level only) but includes _W_MEGASTYLE
# (per-image paired similarity) → its weights sum to 0.97 (max output 0.97).
# Rebalanced for the canon-first regime: since every caption's [Art Style] block is expected
# to be a verbatim copy of the meta-prompt's ``style_foundation`` canon, ``style_consistency``
# (Jaccard across [Art Style] blocks) is now a regression alarm — divergence means the canon
# contract is slipping. Funded by halving the pixel-level SSIM weight, which is already
# largely subsumed by DreamSim (perceptual) + color_histogram (color).
_W_DREAMSIM = 0.34
_W_HPS = 0.07
_W_AESTHETICS = 0.06
_W_COLOR = 0.17
_W_SSIM = 0.06
# style_consistency demoted 0.08 → 0.03 with the MegaStyle-Encoder addition: the 754-pair
# homescapes sweep found token-overlap on caption [Art Style] blocks has Spearman ≈0 with
# image-space style similarity, so it was carrying essentially noise at 0.08. The 0.05
# funded _W_MEGASTYLE, which measures the thing style_consistency was supposed to proxy.
_W_STYLE_CON = 0.03
# MegaStyle-Encoder cosine similarity (ref vs gen) in SigLIP SoViT-400M embedding space
# fine-tuned on 1.4M style-paired images (arxiv 2604.08364). Independent axis vs DreamSim
# (content) and the Gemini vision judge (ternary); Spearman ≈0 with both across 754 pairs,
# so it adds a continuous, content-disentangled style-space signal not captured elsewhere.
# Promoted 0.05 → 0.08 after the homescapes run showed vision_style (ternary, Gemini)
# systematically demoting branches with the highest MegaStyle — two nominal "style" signals
# anti-correlating in practice. MegaStyle is continuous/cheap, vision_style is coarse/costly;
# making MegaStyle the primary style weight (0.08 vs vision_style 0.03) lets fine gradients
# drive selection while keeping the Gemini verdict as a regression alarm alongside
# style_consistency (also 0.03).
_W_MEGASTYLE = 0.08
# Vision slice rebalanced to make room for the medium-class + proportions dims.
# The medium verdict diagnoses a root cause of many style misses (2D/3D misclassification)
# and the proportions verdict diagnoses subject-fidelity misses at the anatomy level, so
# weight reductions on vision_style (-0.02) and vision_subject (-0.03) fund them cleanly.
# vision_style further demoted 0.06 → 0.03 (funding MegaStyle's promotion above) — see the
# _W_MEGASTYLE comment for rationale.
_W_VISION_STYLE = 0.03
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


def replicate_promotion_decision(
    candidate_scores: list[float],
    baseline_scores: list[float],
    *,
    epsilon: float,
) -> str:
    """A1 paired-replicate promotion gate.

    Requires BOTH conditions to promote:

    1. **Hard dominance** — ``min(candidate) > max(baseline)``. With 3 replicates per side
       this cuts SNR from ~30x to ~5x vs a single-shot epsilon check, letting genuinely
       better candidates clear the gate without being drowned by branch-level variance.
    2. **Effect-size clearance** — ``median(candidate) > median(baseline) + epsilon``.
       Dominance without a meaningful median delta is still noise — the candidate just
       happened to sample above the incumbent on every replicate.

    Returns ``"promoted"`` or ``"rejected"``. An empty candidate list always rejects;
    an empty baseline list (first iteration, no prior metrics) skips the dominance check
    and falls back to median-vs-epsilon only.
    """
    if not candidate_scores:
        return "rejected"

    import statistics

    candidate_median = statistics.median(candidate_scores)
    if not baseline_scores:
        return "promoted" if candidate_median > epsilon else "rejected"

    baseline_median = statistics.median(baseline_scores)
    candidate_min = min(candidate_scores)
    baseline_max = max(baseline_scores)
    dominates = candidate_min > baseline_max
    clears_epsilon = candidate_median > baseline_median + epsilon
    return "promoted" if dominates and clears_epsilon else "rejected"


def compliance_mean(m: AggregatedMetrics) -> float:
    """Unweighted mean of the six compliance components on an AggregatedMetrics."""
    return compliance_components_mean(
        m.compliance_topic_coverage,
        m.compliance_marker_coverage,
        m.section_ordering_rate,
        m.section_balance_rate,
        m.subject_specificity_rate,
        m.style_canon_fidelity,
        m.observation_boilerplate_purity,
    )


# Pairs of (label, attribute) driving `metric_deltas`.  Any metric that appears here must be an
# attribute of AggregatedMetrics and meaningful as a diff vs baseline.
_METRIC_DELTA_ATTRS: tuple[tuple[str, str], ...] = (
    ("dreamsim_similarity_mean", "dreamsim_similarity_mean"),
    ("color_histogram_mean", "color_histogram_mean"),
    ("ssim_mean", "ssim_mean"),
    ("hps_score_mean", "hps_score_mean"),
    ("aesthetics_score_mean", "aesthetics_score_mean"),
    ("megastyle_similarity_mean", "megastyle_similarity_mean"),
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


# Labels emitted by :func:`metric_deltas` excluding the `completion_rate` counter — these are
# the perceptual / content / consistency axes callers gate or rank experiments against.
DELTA_METRIC_LABELS: tuple[str, ...] = tuple(label for label, _ in _METRIC_DELTA_ATTRS if label != "completion_rate")


def _normalize_hps(raw: float, ceiling: float = _HPS_CEILING) -> float:
    """Normalize raw HPS v2 score to [0, 1] using the empirical ceiling."""
    return min(raw / ceiling, 1.0)


# ---------------------------------------------------------------------------
# Weighted-axis terms (shared by composite + headroom composite)
# ---------------------------------------------------------------------------

# Each entry is ``(weight, score_extractor)``. Keeping the list in one place means composite_score
# and headroom_composite_score can't drift — both iterate the same axes with the same normalization.
_COMPOSITE_AXES: tuple[tuple[float, Callable[[AggregatedMetrics], float]], ...] = (
    (_W_DREAMSIM, lambda m: m.dreamsim_similarity_mean),
    (_W_HPS, lambda m: _normalize_hps(m.hps_score_mean)),
    (_W_AESTHETICS, lambda m: m.aesthetics_score_mean / 10.0),
    (_W_COLOR, lambda m: m.color_histogram_mean),
    (_W_SSIM, lambda m: m.ssim_mean),
    (_W_STYLE_CON, lambda m: m.style_consistency),
    (_W_MEGASTYLE, lambda m: m.megastyle_similarity_mean),
    (_W_VISION_STYLE, lambda m: m.vision_style),
    (_W_VISION_SUBJECT, lambda m: m.vision_subject),
    (_W_VISION_COMP, lambda m: m.vision_composition),
    (_W_VISION_MEDIUM, lambda m: m.vision_medium),
    (_W_VISION_PROPORTIONS, lambda m: m.vision_proportions),
)


def _composite_penalties(m: AggregatedMetrics) -> float:
    """Sum of all composite penalties (variance, completion, compliance, ref-shortfall, subject-floor).

    Same for ``composite_score`` and ``headroom_composite_score`` — only the forward base term differs.
    """
    variance_penalty = _W_VARIANCE_PENALTY * (m.dreamsim_similarity_std + m.color_histogram_std) / 2.0
    completion_penalty = (1.0 - m.completion_rate) * _W_COMPLETION_PENALTY
    compliance_score = compliance_components_mean(
        m.compliance_topic_coverage,
        m.compliance_marker_coverage,
        m.section_ordering_rate,
        m.section_balance_rate,
        m.subject_specificity_rate,
        m.style_canon_fidelity,
        m.observation_boilerplate_purity,
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
    return variance_penalty + completion_penalty + compliance_penalty + ref_shortfall_penalty + subject_floor_penalty


def composite_score(m: AggregatedMetrics) -> float:
    """Fixed-weight composite score used for absolute quality comparison.

    All metrics normalized to ~[0, 1] before weighting.
    Weights: DreamSim 34%, HPS 7%, Aesthetics 6%, Color 17%, SSIM 6%,
    StyleConsistency 3%, MegaStyle 8%, Vision(style) 3%, Vision(subject) 7%,
    Vision(composition) 4%, Vision(medium) 2%, Vision(proportions) 3%.
    Total = 1.00.
    Includes a consistency penalty based on per-image score variance.
    """
    base = sum(weight * extractor(m) for weight, extractor in _COMPOSITE_AXES)
    return max(0.0, base - _composite_penalties(m))


def headroom_composite_score(m: AggregatedMetrics) -> float:
    """A6: headroom-weighted composite that redirects weight away from saturated axes.

    Saturated metrics (those at or near 1.0) carry no marginal utility for the optimizer —
    keeping their static weight drowns out signal from axes that can still move. This variant
    reweights each axis by ``w_m * (1 - s_m)`` then renormalizes so the weights sum to the
    original ``Σ w_m`` (preserving headline-score scale). When every axis is fully saturated
    we fall back to ``composite_score`` to keep rankings defined.

    Penalties (variance, completion, compliance, ref-shortfall, subject-floor) are applied
    identically to ``composite_score``.
    """
    static_total = sum(weight for weight, _ in _COMPOSITE_AXES)
    raw_headroom = [(weight * (1.0 - extractor(m)), extractor) for weight, extractor in _COMPOSITE_AXES]
    headroom_total = sum(weight for weight, _ in raw_headroom)
    if headroom_total <= 0:
        return composite_score(m)
    scale = static_total / headroom_total
    base = sum((weight * scale) * extractor(m) for weight, extractor in raw_headroom)
    return max(0.0, base - _composite_penalties(m))


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
        lambda r: r.megastyle_similarity_mean,
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
            r.style_canon_fidelity,
            r.observation_boilerplate_purity,
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
# Per-image composite score
# ---------------------------------------------------------------------------


def per_image_composite(s: MetricScores) -> float:
    """Compute a per-image composite score using the same weights as ``composite_score``.

    Unlike ``composite_score`` (which operates on aggregated means), this computes
    the score for a single image — no variance penalty since there's only one observation.
    Omits ``_W_STYLE_CON`` (style consistency is experiment-level only) but includes
    ``_W_MEGASTYLE`` (per-image paired similarity), so max output is 0.97.
    """
    return (
        _W_DREAMSIM * s.dreamsim_similarity
        + _W_HPS * _normalize_hps(s.hps_score)
        + _W_AESTHETICS * (s.aesthetics_score / 10.0)
        + _W_COLOR * s.color_histogram
        + _W_SSIM * s.ssim
        + _W_MEGASTYLE * s.megastyle_similarity
        + _W_VISION_STYLE * s.vision_style
        + _W_VISION_SUBJECT * s.vision_subject
        + _W_VISION_COMP * s.vision_composition
        + _W_VISION_MEDIUM * s.vision_medium
        + _W_VISION_PROPORTIONS * s.vision_proportions
    )
