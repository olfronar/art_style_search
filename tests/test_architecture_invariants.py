"""Architecture-invariant tests — drift prevention for the classes of bug observed in audit.

These tests pin properties of the codebase that span multiple files and are easy to break
silently: composite-weight sums, state-codec round-trips, canon-validator exclusions, and
dead-code detection for features CLAUDE.md claims are shipped.

Each one catches a drift class the audit surfaced:
- Weight rebalancing without summing (``TestCompositeWeightSums``).
- New dataclass field missed by the codec (``TestCodecReflection``).
- Accidental tightening of the MUST/NEVER exclusion in ``_CANON_METHODOLOGY_PATTERNS``
  (``TestCanonMethodologyExclusions``).
- A2/A6-style "defined but never called" drift where a feature is documented as shipped
  in CLAUDE.md but no production code path exercises it (``TestShippedFeaturesHaveProductionCallers``).
- ``--protocol classic`` becoming observationally identical to ``--protocol short``
  (``TestProtocolDivergesInPromotionGate``).
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from art_style_search.prompt._parse import _CANON_METHODOLOGY_PATTERNS, validate_template
from art_style_search.scoring import (
    _COMPOSITE_AXES,
    _W_STYLE_CON,
    per_image_composite,
)
from art_style_search.state_codec import (
    _aggregated_metrics_from_dict,
    _iteration_result_from_dict,
    _metric_scores_from_dict,
    to_dict,
)
from art_style_search.types import AggregatedMetrics, MetricScores
from tests.conftest import (
    make_aggregated_metrics,
    make_iteration_result,
    make_metric_scores,
)
from tests.test_prompt import _make_valid_template

# ---------------------------------------------------------------------------
# Composite-weight invariants
# ---------------------------------------------------------------------------


class TestCompositeWeightSums:
    """The 12-axis weight table must sum to 1.00 — rebalancing without updating the
    full set is the easiest way to accidentally make composite_score non-normalized.

    Also pins the per_image_composite ceiling at 0.97 (= 1.00 - _W_STYLE_CON). If a
    future metric joins the per-image composite, this test fails until the CLAUDE.md
    claim is updated.
    """

    def test_composite_axes_weights_sum_to_one(self) -> None:
        total = sum(weight for weight, _ in _COMPOSITE_AXES)
        assert abs(total - 1.0) < 1e-9, f"_COMPOSITE_AXES sum = {total}, expected 1.0"

    def test_per_image_composite_max_equals_aggregate_minus_style_consistency(self) -> None:
        """per_image_composite omits _W_STYLE_CON but matches aggregate weights otherwise.

        An all-ones MetricScores must produce exactly (1.0 - _W_STYLE_CON).
        """
        perfect = MetricScores(
            dreamsim_similarity=1.0,
            hps_score=1.0,  # normalized /0.35 then clamped to 1.0
            aesthetics_score=10.0,  # normalized /10 → 1.0
            color_histogram=1.0,
            ssim=1.0,
            vision_style=1.0,
            vision_subject=1.0,
            vision_composition=1.0,
            vision_medium=1.0,
            vision_proportions=1.0,
            megastyle_similarity=1.0,
        )
        expected = 1.0 - _W_STYLE_CON
        actual = per_image_composite(perfect)
        assert abs(actual - expected) < 1e-9, (
            f"per_image_composite(perfect) = {actual}, expected {expected} (= 1.0 - _W_STYLE_CON={_W_STYLE_CON})"
        )


# ---------------------------------------------------------------------------
# State-codec reflection round-trip — catches silent field drops
# ---------------------------------------------------------------------------


class TestCodecReflection:
    """Regression guard for the v9 MegaStyle-codec class of bug.

    The codec's ``_*_from_dict`` helpers read fields manually with ``.get(..., default)``.
    When a new field is added to a dataclass but forgotten in the decoder, the fallback
    silently returns the default — losing data on round-trip. These tests use
    ``dataclasses.fields`` to assert that *every* field round-trips with the factory's
    distinctive sentinel values. Factories (``tests/conftest.py``) must seed each field
    with a non-default value so a drop surfaces as inequality.
    """

    def _assert_round_trip(self, obj, from_dict) -> None:
        payload = to_dict(obj)
        restored = from_dict(payload)
        for f in fields(obj):
            original = getattr(obj, f.name)
            round_tripped = getattr(restored, f.name)
            assert round_tripped == original, (
                f"{type(obj).__name__}.{f.name} did not round-trip: "
                f"original={original!r}, restored={round_tripped!r}. "
                "Likely cause: the codec decoder doesn't read this field. "
                "Add an explicit `d.get(...)` entry in state_codec.py."
            )

    def test_metric_scores_round_trip_preserves_all_fields(self) -> None:
        self._assert_round_trip(make_metric_scores(seed=7.0), _metric_scores_from_dict)

    def test_aggregated_metrics_round_trip_preserves_all_fields(self) -> None:
        self._assert_round_trip(make_aggregated_metrics(seed=3.0), _aggregated_metrics_from_dict)

    def test_iteration_result_round_trip_preserves_all_fields(self) -> None:
        # IterationResult nests MetricScores + AggregatedMetrics + PromptTemplate, so this
        # catches drops in any of them when they flow through the history codec path.
        self._assert_round_trip(make_iteration_result(branch_id=2, iteration=4), _iteration_result_from_dict)

    def test_factory_seeds_every_metric_scores_field_with_non_default(self) -> None:
        """Meta-check: without this, a new field added with a default and forgotten by
        both the factory and the codec would round-trip successfully via default=default,
        masking the codec drop. This test forces the factory to seed every field.
        """
        defaults = MetricScores(dreamsim_similarity=0.0, hps_score=0.0, aesthetics_score=0.0)
        sample = make_metric_scores(seed=7.0)
        fields_matching_default = [
            f.name
            for f in fields(sample)
            if f.name not in {"is_fallback", "style_gap"} and getattr(sample, f.name) == getattr(defaults, f.name)
        ]
        assert not fields_matching_default, (
            "make_metric_scores() seeds these fields with the dataclass default, "
            "which would mask a codec round-trip drop: "
            f"{fields_matching_default}. Seed them with distinctive non-default values in conftest.py."
        )

    def test_factory_seeds_every_aggregated_metrics_field_with_non_default(self) -> None:
        """Same meta-check for AggregatedMetrics — forces the factory to seed every
        future-added field so codec round-trip tests have real signal.
        """
        defaults = AggregatedMetrics(
            dreamsim_similarity_mean=0.0,
            dreamsim_similarity_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )
        sample = make_aggregated_metrics(seed=3.0)
        fields_matching_default = [
            f.name for f in fields(sample) if getattr(sample, f.name) == getattr(defaults, f.name)
        ]
        assert not fields_matching_default, (
            "make_aggregated_metrics() seeds these fields with the dataclass default: "
            f"{fields_matching_default}. Seed them with distinctive non-default values in conftest.py."
        )


# ---------------------------------------------------------------------------
# Canon validator: MUST/NEVER/Ensure exclusion is load-bearing
# ---------------------------------------------------------------------------


class TestCanonMethodologyExclusions:
    """`_CANON_METHODOLOGY_PATTERNS` carefully excludes `never`/`always`/`ensure` from its
    sentence-initial imperative list so declarative style invariants ("NEVER outline eyes",
    "Always ground shadows in warm ochre", "Ensure edge rhythm varies") pass.

    If someone adds any of those verbs to the alternation, every valid canon would be
    rejected — a silent, very-hard-to-debug regression. These explicit positive tests
    pin the exclusion.
    """

    @pytest.mark.parametrize(
        "invariant_line",
        [
            "NEVER outline eyes.",
            "never desaturate the shadows below midtone.",
            "Always ground shadows in warm ochre.",
            "always close silhouettes with a single confident outer line.",
            "Ensure edge rhythm varies with surface material.",
            "ensure the rim light opposes the key direction.",
        ],
    )
    def test_style_invariants_pass_methodology_check(self, invariant_line: str) -> None:
        for pattern in _CANON_METHODOLOGY_PATTERNS:
            match = pattern.search(invariant_line)
            assert match is None, (
                f"Invariant line {invariant_line!r} was rejected by _CANON_METHODOLOGY_PATTERNS "
                f"pattern {pattern.pattern!r}. The never/always/ensure exclusion is load-bearing — "
                "if a new imperative was added to the alternation, MUST/NEVER invariants will all fail."
            )

    def test_canon_with_must_never_ensure_invariants_validates(self) -> None:
        """Full integration: a valid canon with MUST/NEVER/Ensure lines passes validate_template."""
        t = _make_valid_template()
        t.sections[0].value = (
            "How to Draw: lineless 2D illustration mimicking stylized 3D rendering. "
            "Construction: silhouette primitives merged into beveled volumes. "
            "Line policy: zero linework; separation by value and hue.\n"
            "Shading & Light: saturated albedo, tight AO, feathered midtones, crisp rim light.\n"
            "Color Principle: high-key candy palette; complementary anchoring; shadows hue-shift cooler.\n"
            "Surface & Texture: zero grain; matte-fondant material vocabulary; every edge rounded.\n"
            "Style Invariants: MUST bevel every edge. NEVER outline or cel-band any form. "
            "Always hue-shift shadows cooler. Ensure every silhouette closes with a single curve. "
            "NEVER use pure black outside pupils.\n"
        ) + "Further concrete style assertions. " * 60
        assert validate_template(t) == [], (
            "Canon with MUST/NEVER/Always/Ensure invariants was rejected — the never/always/ensure "
            "exclusion in _CANON_METHODOLOGY_PATTERNS may have been accidentally tightened."
        )


# ---------------------------------------------------------------------------
# Shipped-vs-scaffolded — CLAUDE.md claims must have production entry points
# ---------------------------------------------------------------------------


def _grep_non_test_call_sites(symbol: str) -> list[str]:
    """Return non-test files under ``src/`` that call ``<symbol>(`` or reference it.

    Approximate but sufficient for "is this function wired in production?" — matches
    explicit call syntax, so imports-without-use still fail (which is what we want).
    AST walking would be more precise but this suffices for the drift we're guarding.
    """
    import re
    from pathlib import Path

    src_root = Path(__file__).resolve().parent.parent / "src"
    pattern = re.compile(rf"\b{re.escape(symbol)}\s*\(")
    hits: list[str] = []
    for py_file in src_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if pattern.search(text):
            hits.append(str(py_file.relative_to(src_root.parent.parent)))
    return hits


class TestShippedFeaturesHaveProductionCallers:
    """CLAUDE.md lists A2 (diff-based canon editing) and A6 (headroom-weighted scoring) as
    shipped under the classic protocol. "Shipped" means: a non-test production file calls
    the feature's entry point. These tests fail if someone un-wires the feature without
    also updating CLAUDE.md.

    The 2026-04-21 audit caught both as silently dead — ``apply_canon_ops`` and
    ``headroom_composite_score`` had definitions + tests but zero production call sites.
    """

    def test_apply_canon_ops_has_non_test_caller(self) -> None:
        hits = _grep_non_test_call_sites("apply_canon_ops")
        assert hits, (
            "apply_canon_ops has no production callers in src/. Either wire it into an "
            "iteration phase (expected location: prompt/json_contracts.py) or remove the "
            "A2 claim from CLAUDE.md and delete the module."
        )

    def test_headroom_composite_score_has_non_test_caller(self) -> None:
        hits = _grep_non_test_call_sites("headroom_composite_score")
        assert hits, (
            "headroom_composite_score has no production callers in src/. Either wire it "
            "into the promotion gate (expected location: workflow/policy.py::_promotion_score) "
            "or remove the A6 claim from CLAUDE.md."
        )


# ---------------------------------------------------------------------------
# Protocol divergence — --protocol classic must change observable behavior
# ---------------------------------------------------------------------------


class TestProtocolDivergesInPromotionGate:
    """The 2026-04-21 audit found the protocol flag was decorative — every iteration-behavior
    branch keyed off ``--replicates`` instead. After cycle 4, classic diverges via the
    promotion-score function (A6 headroom). This test pins that divergence so a future
    refactor that accidentally unifies the branch fails immediately.
    """

    def test_classic_and_short_produce_different_promotion_scores(self) -> None:
        from art_style_search.workflow.policy import _promotion_score
        from tests.conftest import make_aggregated_metrics

        m = make_aggregated_metrics(seed=1.0)
        short_score = _promotion_score(m, protocol="short")
        classic_score = _promotion_score(m, protocol="classic")
        assert short_score != classic_score, (
            "_promotion_score returns identical values for short vs classic on a realistic "
            "metrics fixture. The protocol flag has lost its A6 divergence — the run-time "
            "behavior is now indistinguishable between protocols. Either wire back the "
            "divergence or rename the flag to reflect reality."
        )

    def test_promotion_decision_scoring_function_label_differs_by_protocol(self) -> None:
        """The audit label on promotion_log.jsonl must reflect which function decided
        the promotion — short='composite', classic='headroom'."""
        from art_style_search.workflow.policy import _scoring_function_name

        assert _scoring_function_name("short") != _scoring_function_name("classic"), (
            "scoring_function label is the same for short and classic protocols — "
            "auditors reading promotion_log.jsonl will see no divergence between runs."
        )
