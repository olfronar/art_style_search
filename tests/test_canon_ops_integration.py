"""Integration tests for A2 — canon_ops wired through the expansion validator.

Pure ``apply_canon_ops`` is tested in ``tests/test_canon_ops.py``. These tests pin the
wiring: validator applies payload ops to the prior canon before template validation;
ops are authoritative over the payload value; malformed ops fall back with a warning;
ops flow through to ``CanonEditLedgerEntry.canon_ops``.
"""

from __future__ import annotations

import logging

import pytest

from art_style_search.prompt.json_contracts import validate_expansion_payload
from tests.test_prompt import _make_valid_template


def _base_payload(valid_template, *, value_override: str | None = None, canon_ops=None) -> dict:
    """Build a minimally-valid expansion payload. Override style_foundation.value or attach ops."""
    sections = []
    for i, section in enumerate(valid_template.sections):
        value = section.value
        if i == 0 and value_override is not None:
            value = value_override
        sections.append({"name": section.name, "description": section.description, "value": value})
    payload = {
        "analysis": "Short analysis.",
        "lessons": None,
        "hypothesis": "Tighten canon via a single targeted edit.",
        "builds_on": None,
        "experiment": "Replace a silhouette rule in the canon.",
        "changed_section": "style_foundation",
        "changed_sections": ["style_foundation"],
        "target_category": "style_foundation",
        "direction_id": "D1",
        "direction_summary": "Canon tightening",
        "failure_mechanism": "Canon hand-wave on silhouette construction.",
        "intervention_type": "canon_edit",
        "risk_level": "targeted",
        "expected_primary_metric": "megastyle_similarity_mean",
        "expected_tradeoff": "None expected for a surgical edit.",
        "open_problems": [],
        "template_changes": "Surgical canon edit via replace_sentence.",
        "template": {
            "sections": sections,
            "negative_prompt": "avoid blur",
            "caption_sections": list(valid_template.caption_sections),
            "caption_length_target": 500,
        },
    }
    if canon_ops is not None:
        payload["canon_ops"] = canon_ops
    return payload


class TestCanonOpsAppliedInExpansion:
    def test_canon_ops_apply_to_prior_canon_and_overwrite_section_value(self) -> None:
        valid_template = _make_valid_template()
        prior_canon = valid_template.sections[0].value
        assert "silhouette primitives" in prior_canon

        ops = [{"op": "replace_sentence", "match": "silhouette primitives", "replace": "bold geometric shapes"}]
        payload = _base_payload(valid_template, canon_ops=ops)

        result = validate_expansion_payload(payload, prior_canon=prior_canon)

        new_foundation_value = result.template.sections[0].value
        assert "bold geometric shapes" in new_foundation_value, (
            "canon_ops were not applied to prior_canon — expected 'bold geometric shapes' in style_foundation.value"
        )
        assert "silhouette primitives" not in new_foundation_value
        # Ops echoed back for ledger consumption.
        assert result.canon_ops == ops

    def test_canon_ops_are_authoritative_over_payload_value(self) -> None:
        """When payload sends both a full style_foundation.value and canon_ops, ops win.

        The reasoner may send a stub value when using ops; we must not let a stale value
        leak into the final template.
        """
        valid_template = _make_valid_template()
        prior_canon = valid_template.sections[0].value
        stub_value = "STUB — should be discarded when canon_ops apply cleanly"

        ops = [{"op": "replace_sentence", "match": "silhouette primitives", "replace": "bold geometric shapes"}]
        payload = _base_payload(valid_template, value_override=stub_value, canon_ops=ops)

        result = validate_expansion_payload(payload, prior_canon=prior_canon)

        assert "STUB" not in result.template.sections[0].value
        assert "bold geometric shapes" in result.template.sections[0].value

    def test_missing_canon_ops_falls_through_to_payload_value(self) -> None:
        """Back-compat: payloads without canon_ops use the payload's style_foundation.value."""
        valid_template = _make_valid_template()
        payload = _base_payload(valid_template)  # no canon_ops key

        result = validate_expansion_payload(payload, prior_canon="doesn't matter")

        # Unchanged from valid_template fixture.
        assert "silhouette primitives" in result.template.sections[0].value
        assert result.canon_ops == []

    def test_empty_canon_ops_list_falls_through_to_payload_value(self) -> None:
        """Explicit empty list is treated as "no ops" — don't mutate payload value."""
        valid_template = _make_valid_template()
        payload = _base_payload(valid_template, canon_ops=[])

        result = validate_expansion_payload(payload, prior_canon="doesn't matter")

        assert "silhouette primitives" in result.template.sections[0].value
        assert result.canon_ops == []

    def test_malformed_canon_ops_fall_back_to_payload_value_and_log_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A match-not-found op is a reasoner bug; we don't propagate the error, we fall back.

        Fallback requires the payload's style_foundation.value to still be valid (the reasoner
        sent a usable full rewrite alongside speculative ops).
        """
        valid_template = _make_valid_template()
        prior_canon = valid_template.sections[0].value

        bad_ops = [{"op": "replace_sentence", "match": "this string is NOT in the canon", "replace": "something"}]
        payload = _base_payload(valid_template, canon_ops=bad_ops)

        with caplog.at_level(logging.WARNING):
            result = validate_expansion_payload(payload, prior_canon=prior_canon)

        # Fallback: payload value (from valid_template) is preserved.
        assert "silhouette primitives" in result.template.sections[0].value
        # Warning emitted so observers see the fallback.
        assert any("canon_ops" in rec.message.lower() for rec in caplog.records), (
            f"expected a canon_ops fallback warning; got: {[rec.message for rec in caplog.records]}"
        )

    def test_canon_ops_producing_invalid_canon_rejects_entire_expansion(self) -> None:
        """Ops apply cleanly but result trips _CANON_METHODOLOGY_PATTERNS → validator rejects.

        Guards against the reasoner sneaking methodology/scaffolding text through the ops
        channel (e.g. replacing a rule with audit scaffolding like "MANDATORY ...").
        """
        valid_template = _make_valid_template()
        prior_canon = valid_template.sections[0].value

        # Inject "verbatim" via ops — hits `\b(?:verbatim|paraphrase)\b` in _CANON_METHODOLOGY_PATTERNS.
        # Methodology vocabulary in the canon is exactly what the validator guards against; the A2
        # channel must not be a back door around it.
        ops = [
            {
                "op": "replace_sentence",
                "match": "silhouette primitives",
                "replace": "silhouette primitives (captioner must paste verbatim)",
            }
        ]
        payload = _base_payload(valid_template, canon_ops=ops)

        with pytest.raises(ValueError, match="methodology"):
            validate_expansion_payload(payload, prior_canon=prior_canon)

    def test_refinement_result_exposes_canon_ops_field(self) -> None:
        """RefinementResult.canon_ops must always be a list (empty when payload omits it)."""
        valid_template = _make_valid_template()
        payload = _base_payload(valid_template)

        result = validate_expansion_payload(payload)

        # Type check: attribute exists and is a list.
        assert isinstance(result.canon_ops, list)
        assert result.canon_ops == []


class TestExpandSketchesThreadsPriorCanon:
    """expand_experiment_sketches must bind the incumbent canon as prior_canon for each
    sketch's validator, so the reasoner's canon_ops land against the canon the reasoner
    was shown."""

    def _build_fake_client_and_capture_validators(self, fake_payload: dict):
        captured = {"validators": []}

        class FakeReasoningClient:
            async def call_json(self, *, validator, **_kwargs):
                captured["validators"].append(validator)
                return validator(fake_payload)

        return FakeReasoningClient(), captured

    async def _invoke_expand(self, client, current_template) -> None:
        from art_style_search.contracts import ExperimentSketch
        from art_style_search.prompt.experiments import expand_experiment_sketches
        from art_style_search.types import KnowledgeBase, StyleProfile

        sketch = ExperimentSketch(
            hypothesis="Sharpen silhouette rule via targeted ops.",
            target_category="style_foundation",
            failure_mechanism="Canon hand-wave.",
            intervention_type="canon_edit",
            direction_id="D1",
            direction_summary="Canon tightening",
            risk_level="targeted",
            expected_primary_metric="megastyle_similarity_mean",
        )
        await expand_experiment_sketches(
            style_profile=StyleProfile(
                color_palette="",
                composition="",
                technique="",
                mood_atmosphere="",
                subject_matter="",
                influences="",
                gemini_raw_analysis="",
                claude_raw_analysis="",
            ),
            current_template=current_template,
            knowledge_base=KnowledgeBase(),
            best_metrics=None,
            last_results=None,
            client=client,
            model="test-model",
            sketches=[sketch],
        )

    @pytest.mark.asyncio
    async def test_expand_binds_current_template_canon_as_prior_canon(self) -> None:
        valid_template = _make_valid_template()
        prior_canon = valid_template.sections[0].value
        assert "silhouette primitives" in prior_canon

        ops = [{"op": "replace_sentence", "match": "silhouette primitives", "replace": "bold geometric shapes"}]
        fake_payload = _base_payload(valid_template, canon_ops=ops)

        client, captured = self._build_fake_client_and_capture_validators(fake_payload)
        await self._invoke_expand(client, current_template=valid_template)

        assert captured["validators"], "expand_experiment_sketches did not call client.call_json"
        # Invoke the captured validator directly — it should already have prior_canon bound.
        result = captured["validators"][0](fake_payload)
        assert "bold geometric shapes" in result.template.sections[0].value
        assert "silhouette primitives" not in result.template.sections[0].value


class TestCanonOpsTravelThroughLedger:
    """canon_ops must travel from RefinementResult → ExperimentProposal → IterationResult →
    CanonEditLedgerEntry, and survive state.json round-trip. This is the cross-iteration
    feedback loop — the next iteration's reasoner sees "here's what I tried last time and
    what the metrics did" with the ops themselves, not just before/after excerpts.
    """

    def test_canon_edit_ledger_entry_has_canon_ops_field(self) -> None:
        from art_style_search.types import CanonEditLedgerEntry

        entry = CanonEditLedgerEntry(
            iteration=3,
            prior_canon_excerpt="prior",
            new_canon_excerpt="new",
            changed_sections=["style_foundation"],
            hypothesis_summary="test",
            metric_deltas={"megastyle_similarity_mean": 0.02},
            accepted=True,
            canon_ops=[{"op": "replace_sentence", "match": "a", "replace": "b"}],
        )
        assert entry.canon_ops == [{"op": "replace_sentence", "match": "a", "replace": "b"}]

    def test_experiment_proposal_has_canon_ops_field(self) -> None:
        from art_style_search.contracts import ExperimentProposal, Lessons
        from art_style_search.types import PromptSection, PromptTemplate

        proposal = ExperimentProposal(
            template=PromptTemplate(sections=[PromptSection("s", "d", "v")]),
            hypothesis="",
            experiment_desc="",
            builds_on=None,
            open_problems=[],
            lessons=Lessons(),
            canon_ops=[{"op": "add_sentence", "where": "end", "value": "x"}],
        )
        assert proposal.canon_ops == [{"op": "add_sentence", "where": "end", "value": "x"}]

    def test_iteration_result_has_canon_ops_field(self) -> None:
        from art_style_search.types import AggregatedMetrics, IterationResult, PromptSection, PromptTemplate

        result = IterationResult(
            branch_id=0,
            iteration=0,
            template=PromptTemplate(sections=[PromptSection("s", "d", "v")]),
            rendered_prompt="",
            image_paths=[],
            per_image_scores=[],
            aggregated=AggregatedMetrics(
                dreamsim_similarity_mean=0.0,
                dreamsim_similarity_std=0.0,
                hps_score_mean=0.0,
                hps_score_std=0.0,
                aesthetics_score_mean=0.0,
                aesthetics_score_std=0.0,
            ),
            claude_analysis="",
            template_changes="",
            kept=False,
            canon_ops=[{"op": "replace_slot", "value": "all new"}],
        )
        assert result.canon_ops == [{"op": "replace_slot", "value": "all new"}]

    def test_legacy_v9_state_loads_canon_ops_as_empty_list(self, tmp_path) -> None:
        """Pre-v10 state.json (no canon_ops field in ledger entries) must load cleanly."""
        import json

        from art_style_search.state import load_state

        state_file = tmp_path / "state.json"
        legacy_ledger_entry = {
            "iteration": 1,
            "prior_canon_excerpt": "prior",
            "new_canon_excerpt": "new",
            "changed_sections": ["style_foundation"],
            "hypothesis_summary": "old hypothesis",
            "metric_deltas": {"vision_style": 0.01},
            "accepted": True,
            # canon_ops deliberately omitted — this is the v9 shape
        }
        legacy_payload = {
            "_schema_version": 9,
            "iteration": 1,
            "current_template": {
                "sections": [],
                "negative_prompt": None,
                "caption_sections": [],
                "caption_length_target": 0,
            },
            "best_template": {
                "sections": [],
                "negative_prompt": None,
                "caption_sections": [],
                "caption_length_target": 0,
            },
            "best_metrics": None,
            "knowledge_base": {"hypotheses": [], "categories": {}, "open_problems": [], "next_id": 1},
            "captions": [],
            "style_profile": {
                "color_palette": "",
                "composition": "",
                "technique": "",
                "mood_atmosphere": "",
                "subject_matter": "",
                "influences": "",
                "gemini_raw_analysis": "",
                "claude_raw_analysis": "",
            },
            "fixed_references": [],
            "seed": 42,
            "protocol": "classic",
            "canon_edit_ledger": [legacy_ledger_entry],
        }
        state_file.write_text(json.dumps(legacy_payload), encoding="utf-8")

        loaded = load_state(state_file)

        assert loaded is not None
        assert len(loaded.canon_edit_ledger) == 1
        # Legacy entry gets an empty canon_ops list (not None, not missing).
        assert loaded.canon_edit_ledger[0].canon_ops == []
        # Existing fields preserved.
        assert loaded.canon_edit_ledger[0].prior_canon_excerpt == "prior"
        assert loaded.canon_edit_ledger[0].accepted is True

    def test_canon_edit_ledger_entry_round_trips_canon_ops(self) -> None:
        """State codec must persist canon_ops on a ledger entry (forward compat test)."""
        from art_style_search.state_codec import _canon_edit_ledger_entry_from_dict, to_dict
        from art_style_search.types import CanonEditLedgerEntry

        ops = [
            {"op": "replace_sentence", "match": "foo", "replace": "bar"},
            {"op": "add_sentence", "where": "end", "value": "baz"},
        ]
        entry = CanonEditLedgerEntry(
            iteration=5,
            prior_canon_excerpt="prior content",
            new_canon_excerpt="new content",
            changed_sections=["style_foundation"],
            hypothesis_summary="sharpen silhouette",
            metric_deltas={"megastyle_similarity_mean": 0.02},
            accepted=True,
            canon_ops=ops,
        )
        payload = to_dict(entry)
        restored = _canon_edit_ledger_entry_from_dict(payload)
        assert restored.canon_ops == ops

    def test_append_canon_edit_ledger_records_canon_ops_from_iteration_result(self, tmp_path) -> None:
        """When the best IterationResult carries canon_ops, the ledger entry carries them too."""
        from art_style_search.types import IterationResult, PromptSection, PromptTemplate
        from art_style_search.workflow.iteration_execution import IterationRanking
        from art_style_search.workflow.iteration_persistence import append_canon_edit_ledger
        from tests.conftest import make_aggregated_metrics, make_loop_state

        prior_canon = (
            "How to Draw: silhouette primitives, construction order, line policy. "
            "Shading: soft diffuse. "
            "Color Principle: warm earth tones. "
            "Surface: paper grain. "
            "Style Invariants: NEVER outline eyes. "
        ) + ("Foundation rules. " * 100)
        new_canon = prior_canon.replace("silhouette primitives", "bold geometric shapes")

        ops = [{"op": "replace_sentence", "match": "silhouette primitives", "replace": "bold geometric shapes"}]
        template = PromptTemplate(
            sections=[
                PromptSection(name="style_foundation", description="Canon", value=new_canon),
                PromptSection(
                    name="subject_anchor", description="Subject", value="Proportions: 3.2 heads tall, chibi."
                ),
            ]
        )
        result = IterationResult(
            branch_id=0,
            iteration=1,
            template=template,
            rendered_prompt=template.render(),
            image_paths=[],
            per_image_scores=[],
            aggregated=make_aggregated_metrics(seed=1.0),
            claude_analysis="",
            template_changes="",
            kept=True,
            canon_ops=ops,
        )
        ranking = IterationRanking(
            exp_results=[result],
            adaptive_scores={id(result): 0.5},
            best_exp=result,
            best_score=0.5,
            baseline_score=0.4,
            epsilon=0.005,
            synth_result=None,
        )
        state = make_loop_state(iteration=1)
        state.canon_edit_ledger = []

        append_canon_edit_ledger(
            state,
            ranking,
            prior_canon=prior_canon,
            baseline_metrics=state.best_metrics,
            decision="promoted",
            iteration=1,
        )

        assert len(state.canon_edit_ledger) == 1
        assert state.canon_edit_ledger[0].canon_ops == ops
