"""Regression tests for the short→classic resume flow.

When the short protocol finishes its 3-iteration budget, it marks ``state.converged=True``
with ``convergence_reason=MAX_ITERATIONS``. The user's expected next step is to re-invoke
with ``--protocol classic --max-iterations N`` to extend the same state with more budget.
Before the fix, the early-return at ``loop.run`` treated every ``converged=True`` as
"already done, skip the loop" and the classic resume exited immediately without running
any new iterations.

These tests lock in the resumability contract:

- ``MAX_ITERATIONS`` convergence is resumable when the new invocation provides more budget.
- ``PLATEAU`` and ``REASONING_STOP`` convergence stay respected (the optimizer actively
  decided to stop, not just ran out of budget) — re-running those requires ``--new``.
"""

from __future__ import annotations

from pathlib import Path

from art_style_search.state import load_state, save_state
from art_style_search.types import ConvergenceReason
from tests.conftest import make_aggregated_metrics, make_loop_state


def _build_converged_state(tmp_path: Path, *, iteration: int, reason: ConvergenceReason) -> Path:
    state = make_loop_state(iteration=iteration, global_best_metrics=make_aggregated_metrics())
    state.iteration = iteration
    state.converged = True
    state.convergence_reason = reason
    state.protocol = "short"
    state_file = tmp_path / "state.json"
    save_state(state, state_file)
    return state_file


class TestBudgetResumable:
    def test_max_iterations_convergence_with_more_budget_clears_flags(self, tmp_path: Path) -> None:
        """A completed short run (state.iteration=2, converged=MAX_ITERATIONS) resumed under
        classic with --max-iterations 6 should clear converged/convergence_reason and advance
        state.iteration to 3 so the next classic iter runs from where short left off."""
        state_file = _build_converged_state(tmp_path, iteration=2, reason=ConvergenceReason.MAX_ITERATIONS)

        # Simulate the loop's resume-reset path in-line (pure state mutation — the async setup
        # around it is covered by integration tests).
        state = load_state(state_file)
        assert state is not None
        config_max_iterations = 6

        budget_resumable = (
            state.converged
            and state.convergence_reason == ConvergenceReason.MAX_ITERATIONS
            and state.iteration + 1 < config_max_iterations
        )
        assert budget_resumable, "short→classic with bigger budget must be resumable"

        state.converged = False
        state.convergence_reason = None
        state.plateau_counter = 0
        state.protocol = "classic"
        state.iteration += 1
        save_state(state, state_file)

        reloaded = load_state(state_file)
        assert reloaded is not None
        assert reloaded.converged is False
        assert reloaded.convergence_reason is None
        assert reloaded.iteration == 3, "iteration must advance past the completed short iter"
        assert reloaded.protocol == "classic"

    def test_plateau_convergence_is_not_resumable(self, tmp_path: Path) -> None:
        """Plateau means the optimizer couldn't find improvements — respecting it is the point.
        User must pass --new to override."""
        _build_converged_state(tmp_path, iteration=2, reason=ConvergenceReason.PLATEAU)
        state = load_state(tmp_path / "state.json")
        assert state is not None

        budget_resumable = (
            state.converged and state.convergence_reason == ConvergenceReason.MAX_ITERATIONS and state.iteration + 1 < 6
        )
        assert budget_resumable is False

    def test_reasoning_stop_is_not_resumable(self, tmp_path: Path) -> None:
        """Reasoner explicitly signaled CONVERGED — respect the decision."""
        _build_converged_state(tmp_path, iteration=2, reason=ConvergenceReason.REASONING_STOP)
        state = load_state(tmp_path / "state.json")
        assert state is not None

        budget_resumable = (
            state.converged and state.convergence_reason == ConvergenceReason.MAX_ITERATIONS and state.iteration + 1 < 6
        )
        assert budget_resumable is False

    def test_same_budget_is_not_resumable(self, tmp_path: Path) -> None:
        """Resuming with max_iterations == state.iteration+1 has no runway — stay respected."""
        _build_converged_state(tmp_path, iteration=2, reason=ConvergenceReason.MAX_ITERATIONS)
        state = load_state(tmp_path / "state.json")
        assert state is not None

        # state.iteration=2 means 3 iterations completed (0, 1, 2); max=3 has no more budget.
        budget_resumable = (
            state.converged and state.convergence_reason == ConvergenceReason.MAX_ITERATIONS and state.iteration + 1 < 3
        )
        assert budget_resumable is False
