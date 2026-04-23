"""Tests for ``ProposalBatchRecorder`` + its state-codec round-trip."""

from __future__ import annotations

import pytest

from art_style_search.contracts import ExperimentSketch, Lessons, RefinementResult
from art_style_search.state_codec import proposal_batch_from_dict, to_dict
from art_style_search.types import PromptSection, PromptTemplate
from art_style_search.workflow.proposal_recorder import ProposalBatchRecorder


def _sketch(
    hypothesis: str, direction: str = "D1", risk: str = "targeted", cat: str = "subject_anchor"
) -> ExperimentSketch:
    return ExperimentSketch(
        hypothesis=hypothesis,
        target_category=cat,
        failure_mechanism="mech",
        intervention_type="intervention",
        direction_id=direction,
        direction_summary=f"Direction {direction} summary",
        risk_level=risk,
        expected_primary_metric="dreamsim_similarity_mean",
        builds_on="",
    )


def _refinement(hypothesis: str, direction: str = "D1", risk: str = "targeted") -> RefinementResult:
    return RefinementResult(
        template=PromptTemplate(
            sections=[PromptSection(name="style_foundation", description="d", value="v")],
            negative_prompt=None,
            caption_sections=[],
            caption_length_target=0,
        ),
        analysis="",
        template_changes="",
        should_stop=False,
        hypothesis=hypothesis,
        experiment="exp",
        lessons=Lessons(),
        builds_on=None,
        open_problems=[],
        target_category="subject_anchor",
        direction_id=direction,
        direction_summary=f"Direction {direction} summary",
        failure_mechanism="mech",
        intervention_type="intervention",
        risk_level=risk,
    )


class TestProposalBatchRecorder:
    def test_walks_realistic_fate_sequence(self) -> None:
        sketches = [_sketch(f"hyp {i}", direction=["D1", "D2", "D3"][i % 3]) for i in range(9)]
        recorder = ProposalBatchRecorder(iteration=3)
        recorder.record_brainstorm(sketches)

        assert [r.rank for r in recorder.records] == list(range(9))
        assert all(r.fate == "brainstormed" for r in recorder.records)

        # Trim the last two.
        recorder.mark_trimmed([7, 8])
        assert recorder.records[7].fate == "trimmed"
        assert recorder.records[8].fate == "trimmed"

        # Stage-1 dedup kills rank 1.
        recorder.mark_deduped_stage1(1, "category=subject_anchor; mechanism=mech; intervention=intervention")
        assert recorder.records[1].fate == "deduped_stage1"
        assert recorder.records[1].fate_reason is not None
        assert "category=subject_anchor" in recorder.records[1].fate_reason

        # Attach refinements to surviving ranks (0, 2..6).
        survivor_ranks = [0, 2, 3, 4, 5, 6]
        pairs = [(rank, _refinement(sketches[rank].hypothesis, sketches[rank].direction_id)) for rank in survivor_ranks]
        recorder.attach_refinements(pairs)
        assert all(recorder.records[r].refinement is not None for r in survivor_ranks)
        assert recorder.refinement_to_rank() == {
            id(rec.refinement): rec.rank for rec in recorder.records if rec.refinement
        }

        # Stage-2 dedup kills rank 4.
        recorder.mark_deduped_stage2(4, "duplicate mechanism")
        assert recorder.records[4].fate == "deduped_stage2"

        # Not-picked for rank 5, 6. Invalid template for rank 3.
        recorder.mark_not_picked(5)
        recorder.mark_not_picked(6)
        recorder.mark_invalid(3, ["targeted experiments must change exactly 1 section"])
        assert recorder.records[3].fate == "invalid_template"
        assert "1 section" in (recorder.records[3].fate_reason or "")

        # Executed: rank 0 as branch 0, rank 2 as branch 1.
        recorder.mark_executed(0, 0)
        recorder.mark_executed(2, 1)
        assert recorder.records[0].fate == "executed"
        assert recorder.records[0].branch_id == 0
        assert recorder.records[2].branch_id == 1

        final_fates = [r.fate for r in recorder.records]
        assert final_fates.count("executed") == 2
        assert final_fates.count("trimmed") == 2
        assert final_fates.count("deduped_stage1") == 1
        assert final_fates.count("deduped_stage2") == 1
        assert final_fates.count("not_picked") == 2
        assert final_fates.count("invalid_template") == 1

    def test_out_of_range_rank_is_noop(self) -> None:
        recorder = ProposalBatchRecorder(iteration=0)
        recorder.record_brainstorm([_sketch("only one")])
        recorder.mark_deduped_stage1(42, "bad rank")
        # Original record untouched.
        assert recorder.records[0].fate == "brainstormed"


class TestCodecRoundTrip:
    def test_full_round_trip_preserves_fates_and_details(self) -> None:
        sketches = [_sketch(f"hyp {i}", direction=["D1", "D2", "D3"][i % 3]) for i in range(5)]
        recorder = ProposalBatchRecorder(iteration=7)
        recorder.record_brainstorm(sketches)
        recorder.mark_deduped_stage1(1, "collision X")
        refinement = _refinement("hyp 0", direction="D1")
        recorder.attach_refinements([(0, refinement)])
        recorder.mark_executed(0, 0)
        recorder.mark_not_picked(2)
        recorder.mark_invalid(3, ["err A", "err B"])

        from art_style_search.state_codec import proposal_batch_to_dict

        payload = proposal_batch_to_dict(recorder)
        assert payload["iteration"] == 7
        assert len(payload["records"]) == 5
        assert "refinement" not in payload["records"][0]
        assert "proposal" not in payload["records"][0]

        restored = proposal_batch_from_dict(payload)
        assert restored.iteration == 7
        assert [r.fate for r in restored.records] == [r.fate for r in recorder.records]
        assert [r.rank for r in restored.records] == [r.rank for r in recorder.records]
        assert restored.records[0].branch_id == 0
        assert restored.records[0].refinement is None  # intentionally not persisted
        assert restored.records[0].proposal is None  # intentionally not persisted
        assert restored.records[1].fate_reason == "collision X"
        assert restored.records[3].fate == "invalid_template"
        assert restored.records[3].fate_reason == "err A; err B"
        assert restored.records[2].fate == "not_picked"

    def test_invalid_fate_defaults_to_brainstormed(self) -> None:
        payload = {
            "iteration": 0,
            "records": [
                {
                    "rank": 0,
                    "sketch": to_dict(_sketch("h")),
                    "fate": "not_a_real_fate",
                }
            ],
        }
        restored = proposal_batch_from_dict(payload)
        assert restored.records[0].fate == "brainstormed"


@pytest.mark.parametrize("fate", ["executed", "not_picked", "deduped_stage1", "deduped_stage2", "invalid_template"])
def test_single_stage_transitions(fate: str) -> None:
    recorder = ProposalBatchRecorder(iteration=0)
    recorder.record_brainstorm([_sketch("h")])
    method = {
        "executed": lambda: recorder.mark_executed(0, 0),
        "not_picked": lambda: recorder.mark_not_picked(0),
        "deduped_stage1": lambda: recorder.mark_deduped_stage1(0, "r"),
        "deduped_stage2": lambda: recorder.mark_deduped_stage2(0, "r"),
        "invalid_template": lambda: recorder.mark_invalid(0, ["r"]),
    }[fate]
    method()
    assert recorder.records[0].fate == fate
