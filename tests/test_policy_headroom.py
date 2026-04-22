"""Integration tests for A6 — headroom scoring wired into the classic promotion gate.

Pins: ``_promotion_score`` dispatches on protocol; classic ∘ A1 composes so every replicate
score uses headroom; ``PromotionDecision.scoring_function`` records which branch decided.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from art_style_search.scoring import composite_score, headroom_composite_score
from art_style_search.types import AggregatedMetrics
from art_style_search.workflow.iteration_execution import IterationRanking
from art_style_search.workflow.policy import _apply_iteration_result, _promotion_score
from tests.conftest import make_aggregated_metrics, make_iteration_result, make_loop_state

if TYPE_CHECKING:
    from art_style_search.workflow.context import RunContext


def _config_stub(protocol: str, run_dir, replicates: int = 1):
    from art_style_search.config import Config

    (run_dir / "refs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    return Config(
        run_dir=run_dir,
        log_dir=run_dir / "logs",
        output_dir=run_dir / "outputs",
        state_file=run_dir / "state.json",
        reference_dir=run_dir / "refs",
        max_iterations=5,
        plateau_window=5,
        num_branches=3,
        aspect_ratio="1:1",
        num_fixed_refs=20,
        caption_model="fake",
        generator_model="fake",
        reasoning_model="fake",
        reasoning_provider="anthropic",
        reasoning_base_url="",
        gemini_concurrency=5,
        eval_concurrency=2,
        seed=42,
        protocol=protocol,
        replicates=replicates,
        anthropic_api_key="fake",
        google_api_key="fake",
        zai_api_key="",
        openai_api_key="",
    )


class TestPromotionScoreDispatch:
    """The `_promotion_score` helper dispatches on protocol."""

    def test_short_protocol_uses_composite_score(self) -> None:
        m = make_aggregated_metrics(seed=1.0)
        assert _promotion_score(m, protocol="short") == composite_score(m)

    def test_classic_protocol_uses_headroom_composite_score(self) -> None:
        m = make_aggregated_metrics(seed=1.0)
        assert _promotion_score(m, protocol="classic") == headroom_composite_score(m)

    def test_short_and_classic_diverge_on_realistic_metrics(self) -> None:
        """Sanity check: the dispatch actually produces different numbers for a realistic case.

        This is the load-bearing assertion — if composite == headroom on the factory fixture,
        the A6 wiring is observationally indistinguishable from A0 and the drift-prevention
        tests in cycle 5 can't catch a future unwiring.
        """
        m = make_aggregated_metrics(seed=1.0)
        short_score = _promotion_score(m, protocol="short")
        classic_score = _promotion_score(m, protocol="classic")
        assert short_score != classic_score, (
            "short and classic produce identical scores on the factory fixture — "
            "the protocol flag's A6 divergence is not observable"
        )


class TestPromotionGateUsesHeadroomWhenClassic:
    """`_apply_iteration_result` must promote/reject based on the protocol-appropriate score."""

    def _make_ranking_with(
        self, best_metrics: AggregatedMetrics, baseline_metrics: AggregatedMetrics
    ) -> IterationRanking:
        best_exp = make_iteration_result(branch_id=0, iteration=1)
        best_exp.aggregated = best_metrics
        # Ranking is populated by _score_and_rank; we bypass it here to pin specific scores.
        return IterationRanking(
            exp_results=[best_exp],
            adaptive_scores={id(best_exp): 0.5},
            best_exp=best_exp,
            best_score=composite_score(best_metrics),  # will be recomputed inside the test
            baseline_score=composite_score(baseline_metrics),
            epsilon=0.001,
            synth_result=None,
        )

    def test_classic_protocol_scores_ranking_with_headroom(self, tmp_path) -> None:
        """A ranking whose composite barely beats baseline but whose headroom gap is large
        must promote under classic (headroom redistributes weight to movable axes)."""
        # Build a baseline that's highly saturated (most axes near 1.0). Its composite is high
        # but its headroom is low — any candidate with measurable improvement on unsaturated axes
        # should look better under headroom weighting.
        saturated = replace(
            make_aggregated_metrics(seed=0.0),
            dreamsim_similarity_mean=0.98,
            dreamsim_similarity_std=0.005,
            color_histogram_mean=0.98,
            color_histogram_std=0.005,
            megastyle_similarity_mean=0.98,
            vision_subject=0.3,  # unsaturated — the real headroom
        )
        # Candidate improves vision_subject (low-saturation axis, large headroom-weighted gain)
        # but dreamsim drifts a hair. Composite barely moves. Headroom composite moves more.
        candidate = replace(saturated, vision_subject=0.85, dreamsim_similarity_mean=0.975)
        state = make_loop_state(iteration=1)
        state.best_metrics = saturated
        state.best_template = state.current_template  # non-empty so replicate path wouldn't trigger
        ctx_config = _config_stub("classic", tmp_path)
        ranking = self._make_ranking_with(candidate, saturated)
        # Pin the scores the gate actually uses.
        ranking.best_score = _promotion_score(candidate, protocol="classic")
        ranking.baseline_score = _promotion_score(saturated, protocol="classic")
        assert ranking.best_score > ranking.baseline_score + ranking.epsilon, (
            f"test fixture broken: headroom scores are indistinguishable "
            f"({ranking.best_score} vs {ranking.baseline_score})"
        )

        decision = _apply_iteration_result(state, ranking, ctx_config)

        assert decision == "promoted"

    def test_promotion_decision_records_scoring_function(self, tmp_path) -> None:
        """The promotion_log must record which scoring function decided the promotion."""
        import json

        state = make_loop_state(iteration=1)
        ctx_config = _config_stub("classic", tmp_path)
        # Any ranking where the candidate clearly beats baseline under headroom.
        m_cand = make_aggregated_metrics(seed=5.0)
        m_base = make_aggregated_metrics(seed=0.0)
        ranking = self._make_ranking_with(m_cand, m_base)
        ranking.best_score = _promotion_score(m_cand, protocol="classic")
        ranking.baseline_score = _promotion_score(m_base, protocol="classic")
        state.best_metrics = m_base
        ctx_config.run_dir.mkdir(parents=True, exist_ok=True)

        _apply_iteration_result(state, ranking, ctx_config)

        log_path = ctx_config.run_dir / "promotion_log.jsonl"
        assert log_path.exists(), "promotion_log.jsonl was not written"
        line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        record = json.loads(line)
        assert record.get("scoring_function") == "headroom", (
            f"Expected scoring_function=headroom under classic protocol; got {record!r}"
        )


class TestReplicateGateComposesWithHeadroomWhenClassic:
    """A1 paired-replicate scores must use headroom under classic (A1 ∘ A6)."""

    async def _build_ranking_with_replicate_scores(self, tmp_path, protocol: str):
        """Invoke _run_replicate_gate with a stub replicate_experiment that returns
        controlled per-replicate AggregatedMetrics, and return the resulting ranking."""
        import art_style_search.workflow.iteration_execution as ie

        # Known replicates — two replicates with distinct metrics.
        from art_style_search.types import ReplicatedEvaluation

        rep_a = make_aggregated_metrics(seed=2.0)
        rep_b = make_aggregated_metrics(seed=7.0)
        median = make_aggregated_metrics(seed=4.5)

        async def fake_replicate(
            *,
            template,
            branch_id,
            iteration,
            fixed_refs,
            config,
            n_replicates,
            services,
            existing_result=None,
        ):
            return ReplicatedEvaluation(
                template=template,
                branch_id=branch_id,
                replicate_scores=[[], []],
                replicate_aggregated=[rep_a, rep_b],
                median_per_image=[],
                median_aggregated=median,
            )

        from unittest.mock import patch

        best_exp = make_iteration_result(branch_id=0, iteration=1)
        ranking = IterationRanking(
            exp_results=[best_exp],
            adaptive_scores={id(best_exp): 0.5},
            best_exp=best_exp,
            best_score=composite_score(best_exp.aggregated),
            baseline_score=composite_score(best_exp.aggregated),
            epsilon=0.001,
            synth_result=None,
        )
        state = make_loop_state(iteration=1)
        state.best_template = state.current_template  # non-empty

        class _Ctx:
            def __init__(self, config):
                self.config = config
                self.services = None

        ctx = cast("RunContext", _Ctx(_config_stub(protocol, tmp_path, replicates=2)))

        with patch.object(ie, "replicate_experiment", fake_replicate):
            await ie._run_replicate_gate(ranking, state, ctx, iteration=1)

        return ranking, rep_a, rep_b

    @pytest.mark.asyncio
    async def test_classic_populates_headroom_replicate_scores(self, tmp_path) -> None:
        ranking, rep_a, rep_b = await self._build_ranking_with_replicate_scores(tmp_path, "classic")
        expected = [headroom_composite_score(rep_a), headroom_composite_score(rep_b)]
        assert ranking.best_replicate_scores == expected, (
            "classic replicate gate must compute per-replicate scores with headroom"
        )

    @pytest.mark.asyncio
    async def test_short_populates_composite_replicate_scores(self, tmp_path) -> None:
        ranking, rep_a, rep_b = await self._build_ranking_with_replicate_scores(tmp_path, "short")
        expected = [composite_score(rep_a), composite_score(rep_b)]
        assert ranking.best_replicate_scores == expected, (
            "short replicate gate must keep plain composite (no protocol divergence)"
        )
