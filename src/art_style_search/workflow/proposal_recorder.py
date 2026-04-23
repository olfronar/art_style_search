"""Accumulator for raw proposal fates in one iteration.

Tracks every sketch emitted by ``brainstorm_experiment_sketches`` along with the stage at
which it was kept or dropped (``deduped_stage1``, ``deduped_stage2``, ``not_picked``,
``invalid_template``, ``executed``). Serialized once per iteration via
``state.save_iteration_proposals`` into ``runs/<name>/logs/iter_NNN_proposals.json`` so the
report can render the full ideation trail — not just portfolio survivors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from art_style_search.contracts import ExperimentProposal, ExperimentSketch, RefinementResult

ProposalFate = Literal[
    "brainstormed",
    "trimmed",
    "deduped_stage1",
    "deduped_stage2",
    "not_picked",
    "invalid_template",
    "executed",
]


@dataclass
class ProposalRecord:
    """One reasoner-emitted sketch plus whatever survived of its downstream fate."""

    rank: int
    sketch: ExperimentSketch
    refinement: RefinementResult | None = None
    proposal: ExperimentProposal | None = None
    fate: ProposalFate = "brainstormed"
    fate_reason: str | None = None
    branch_id: int | None = None


@dataclass
class ProposalBatchRecorder:
    """Mutable per-iteration accumulator threaded through the proposal pipeline."""

    iteration: int
    records: list[ProposalRecord] = field(default_factory=list)

    def record_brainstorm(self, ranked_sketches: list[ExperimentSketch]) -> None:
        """Seed the recorder with the post-rank (pre-trim) sketch order."""
        self.records = [
            ProposalRecord(rank=rank, sketch=sketch, fate="brainstormed") for rank, sketch in enumerate(ranked_sketches)
        ]

    def mark_trimmed(self, ranks: list[int]) -> None:
        for r in ranks:
            self._set_fate(r, "trimmed")

    def mark_deduped_stage1(self, rank: int, reason: str) -> None:
        self._set_fate(rank, "deduped_stage1", reason=reason)

    def attach_refinements(self, pairs: list[tuple[int, RefinementResult]]) -> None:
        for rank, refinement in pairs:
            record = self._by_rank(rank)
            if record is not None:
                record.refinement = refinement

    def mark_deduped_stage2(self, rank: int, reason: str) -> None:
        self._set_fate(rank, "deduped_stage2", reason=reason)

    def attach_proposal(self, rank: int, proposal: ExperimentProposal) -> None:
        record = self._by_rank(rank)
        if record is not None:
            record.proposal = proposal

    def mark_not_picked(self, rank: int) -> None:
        self._set_fate(rank, "not_picked")

    def mark_invalid(self, rank: int, errors: list[str]) -> None:
        self._set_fate(rank, "invalid_template", reason="; ".join(errors))

    def mark_executed(self, rank: int, branch_id: int) -> None:
        record = self._by_rank(rank)
        if record is not None:
            record.fate = "executed"
            record.branch_id = branch_id

    def refinement_to_rank(self) -> dict[int, int]:
        """Identity map from ``id(refinement)`` to rank for helpers that receive RefinementResult lists."""
        return {id(r.refinement): r.rank for r in self.records if r.refinement is not None}

    def proposal_to_rank(self) -> dict[int, int]:
        """Identity map from ``id(proposal)`` to rank."""
        return {id(r.proposal): r.rank for r in self.records if r.proposal is not None}

    def _by_rank(self, rank: int) -> ProposalRecord | None:
        if 0 <= rank < len(self.records):
            return self.records[rank]
        return None

    def _set_fate(self, rank: int, fate: ProposalFate, *, reason: str | None = None) -> None:
        record = self._by_rank(rank)
        if record is None:
            return
        record.fate = fate
        if reason is not None:
            record.fate_reason = reason
