"""Integration tests for _run_replicate_gate (A1 orchestration).

Covers failure-mode fallback and the no-op path, addressing the coverage gap flagged by
the Phase-8 code review. Specifically: when incumbent replication fails mid-run, the gate
MUST fall all the way back to single-shot (leave both replicate-score fields as None),
never populate with an empty baseline list — that would trip the empty-baseline branch
of replicate_promotion_decision and silently auto-promote candidates that haven't
actually dominated.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from art_style_search.workflow.iteration_execution import IterationRanking, _run_replicate_gate
from tests.conftest import make_aggregated_metrics, make_iteration_result, make_loop_state


def _make_ranking(best_exp, baseline_score: float) -> IterationRanking:
    return IterationRanking(
        exp_results=[best_exp],
        adaptive_scores={id(best_exp): 0.65},
        best_exp=best_exp,
        best_score=0.65,
        baseline_score=baseline_score,
        epsilon=0.002,
    )


def _make_ctx(tmp_path: Path, replicates: int) -> MagicMock:
    """Minimal stub RunContext with just the fields _run_replicate_gate touches."""
    ctx = MagicMock()
    ctx.config = MagicMock()
    ctx.config.replicates = replicates
    ctx.config.num_fixed_refs = 20
    ctx.config.log_dir = tmp_path
    ctx.config.output_dir = tmp_path
    ctx.services = MagicMock()
    return ctx


class TestReplicateGateOrchestration:
    @pytest.mark.asyncio
    async def test_replicates_equals_one_skips_gate_entirely(self, tmp_path: Path) -> None:
        """config.replicates < 2 → early return; both replicate-score fields stay None."""
        state = make_loop_state()
        state.best_metrics = make_aggregated_metrics()
        best_exp = make_iteration_result(branch_id=0, iteration=1)
        ranking = _make_ranking(best_exp, baseline_score=0.55)
        ctx = _make_ctx(tmp_path, replicates=1)

        await _run_replicate_gate(ranking, state, ctx, iteration=1)

        assert ranking.best_replicate_scores is None
        assert ranking.baseline_replicate_scores is None

    @pytest.mark.asyncio
    async def test_incumbent_failure_falls_back_to_single_shot(self, tmp_path: Path, monkeypatch) -> None:
        """If incumbent replication raises, leave BOTH score fields None — the policy layer
        then uses single-shot. Leaving ``baseline_replicate_scores = []`` with a populated
        candidate would silently auto-promote via the empty-baseline branch of
        replicate_promotion_decision (bug flagged in code review)."""
        state = make_loop_state()
        state.best_metrics = make_aggregated_metrics()
        # best_template must be non-empty for the incumbent-replication task to be scheduled.
        state.best_template.sections = [MagicMock(name="fake_section")]
        best_exp = make_iteration_result(branch_id=0, iteration=1)
        ranking = _make_ranking(best_exp, baseline_score=0.55)
        ctx = _make_ctx(tmp_path, replicates=3)

        async def fake_replicate_experiment(
            *, template, branch_id, iteration, fixed_refs, config, services, n_replicates=3, **kwargs
        ):
            if branch_id == 900:  # incumbent
                msg = "incumbent captioning died"
                raise RuntimeError(msg)
            return MagicMock(
                replicate_aggregated=[make_aggregated_metrics()] * 3,
                median_aggregated=make_aggregated_metrics(),
                median_per_image=[],
            )

        monkeypatch.setattr(
            "art_style_search.workflow.iteration_execution.replicate_experiment",
            fake_replicate_experiment,
        )

        await _run_replicate_gate(ranking, state, ctx, iteration=1)

        # BOTH must be None so policy layer skips the replicate branch.
        assert ranking.best_replicate_scores is None, (
            "incumbent failure must not leave candidate replicates populated — that would silently "
            "auto-promote the candidate via the empty-baseline fallback"
        )
        assert ranking.baseline_replicate_scores is None

    @pytest.mark.asyncio
    async def test_candidate_failure_falls_back_to_single_shot(self, tmp_path: Path, monkeypatch) -> None:
        """Symmetric: if candidate replication raises, leave both fields None."""
        state = make_loop_state()
        state.best_metrics = make_aggregated_metrics()
        state.best_template.sections = [MagicMock(name="fake_section")]
        best_exp = make_iteration_result(branch_id=0, iteration=1)
        ranking = _make_ranking(best_exp, baseline_score=0.55)
        ctx = _make_ctx(tmp_path, replicates=3)

        async def fake_replicate_experiment(*args, **kwargs):
            msg = "candidate captioning died"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            "art_style_search.workflow.iteration_execution.replicate_experiment",
            fake_replicate_experiment,
        )

        await _run_replicate_gate(ranking, state, ctx, iteration=1)

        assert ranking.best_replicate_scores is None
        assert ranking.baseline_replicate_scores is None

    @pytest.mark.asyncio
    async def test_no_incumbent_template_populates_empty_baseline(self, tmp_path: Path, monkeypatch) -> None:
        """First iteration (empty best_template.sections) → no incumbent task scheduled.
        Candidate replicates populate, baseline stays []. This is the legitimate bootstrap
        path — replicate_promotion_decision handles empty baseline via median-vs-epsilon."""
        state = make_loop_state()
        state.best_template.sections = []  # fresh state, no incumbent
        best_exp = make_iteration_result(branch_id=0, iteration=0)
        ranking = _make_ranking(best_exp, baseline_score=float("-inf"))
        ctx = _make_ctx(tmp_path, replicates=3)

        async def fake_replicate_experiment(*args, **kwargs):
            return MagicMock(
                replicate_aggregated=[make_aggregated_metrics()] * 3,
                median_aggregated=make_aggregated_metrics(),
                median_per_image=[],
            )

        monkeypatch.setattr(
            "art_style_search.workflow.iteration_execution.replicate_experiment",
            fake_replicate_experiment,
        )

        await _run_replicate_gate(ranking, state, ctx, iteration=0)

        assert ranking.best_replicate_scores is not None
        assert len(ranking.best_replicate_scores) == 3
        assert ranking.baseline_replicate_scores == []
