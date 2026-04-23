"""Unit tests for workflow.iteration_proposals."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from art_style_search.config import Config
from art_style_search.contracts import ExperimentSketch, Lessons, RefinementResult
from art_style_search.types import (
    AggregatedMetrics,
    KnowledgeBase,
    LoopState,
    PromptSection,
    PromptTemplate,
    StyleProfile,
)
from art_style_search.workflow.context import RunContext
from art_style_search.workflow.iteration_proposals import _propose_iteration_experiments


def _valid_template() -> PromptTemplate:
    return PromptTemplate(
        sections=[
            PromptSection(
                name="style_foundation",
                description="rules",
                value=(
                    "How to Draw: silhouette primitives, construction order, line policy, "
                    "shading layers, signature quirk. "
                )
                + "Shared style rules. " * 125,
            ),
            PromptSection(
                name="subject_anchor",
                description="subject",
                value=(
                    "Proportions: 3.2 heads tall, chibi archetype, stubby limbs. "
                    "Distinguishing Features: species, hair/fur, markings, apparel, props. "
                )
                + "Subject rules. " * 125,
            ),
            PromptSection(name="composition_blueprint", description="layout", value="Layout rules. " * 130),
            PromptSection(name="lighting_rendering", description="light", value="Lighting rules. " * 130),
            PromptSection(name="environment_staging", description="environment", value="Environment rules. " * 130),
            PromptSection(name="color_palette", description="palette", value="Palette rules. " * 130),
            PromptSection(name="texture_language", description="texture", value="Texture rules. " * 130),
            PromptSection(name="negative_constraints", description="avoid", value="Negative rules. " * 130),
        ],
        caption_sections=["Art Style", "Subject", "Composition", "Lighting", "Texture"],
        caption_length_target=500,
    )


def _current_template_with_face_hands_pose() -> PromptTemplate:
    return PromptTemplate(
        sections=[
            PromptSection(
                name="style_foundation",
                description="rules",
                value=(
                    "How to Draw: silhouette primitives, construction order, line policy, "
                    "shading layers, signature quirk. "
                )
                + "Shared style rules. " * 125,
            ),
            PromptSection(
                name="subject_anchor",
                description="subject",
                value=(
                    "Proportions: 3.2 heads tall, chibi archetype, stubby limbs. "
                    "Distinguishing Features: species, hair/fur, markings, apparel, props. "
                )
                + "Subject rules. " * 125,
            ),
            PromptSection(name="face_hands_pose", description="anatomy", value="Pose rules. " * 130),
            PromptSection(name="global_layout_grid", description="layout", value="Layout rules. " * 130),
            PromptSection(name="palette_temperature", description="palette", value="Palette rules. " * 130),
            PromptSection(name="lighting_rendering", description="light", value="Lighting rules. " * 130),
            PromptSection(name="environment_staging", description="environment", value="Environment rules. " * 130),
            PromptSection(name="negative_constraints", description="avoid", value="Negative rules. " * 130),
        ],
        caption_sections=["Art Style", "Subject", "Pose", "Layout", "Lighting"],
        caption_length_target=500,
    )


def _make_config(tmp_path: Path) -> Config:
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        reference_dir=ref_dir,
        output_dir=tmp_path / "outputs",
        log_dir=tmp_path / "logs",
        state_file=tmp_path / "state.json",
        run_dir=tmp_path,
        max_iterations=10,
        plateau_window=5,
        num_branches=9,
        aspect_ratio="1:1",
        num_fixed_refs=4,
        caption_model="caption",
        generator_model="generator",
        reasoning_model="reasoner",
        reasoning_provider="anthropic",
        reasoning_base_url="",
        gemini_concurrency=2,
        eval_concurrency=2,
        seed=42,
        protocol="classic",
        anthropic_api_key="anthropic",
        google_api_key="google",
        zai_api_key="",
        openai_api_key="",
        raw_proposals=9,
    )


def _make_state() -> LoopState:
    template = _valid_template()
    return LoopState(
        iteration=1,
        current_template=template,
        best_template=template,
        best_metrics=AggregatedMetrics(
            dreamsim_similarity_mean=0.6,
            dreamsim_similarity_std=0.02,
            hps_score_mean=0.25,
            hps_score_std=0.01,
            aesthetics_score_mean=5.9,
            aesthetics_score_std=0.3,
        ),
        knowledge_base=KnowledgeBase(),
        captions=[],
        style_profile=StyleProfile(
            color_palette="Muted palette.",
            composition="Simple composition.",
            technique="Painterly stylization.",
            mood_atmosphere="Quiet.",
            subject_matter="Characters and environments.",
            influences="Stylized mobile illustration.",
            gemini_raw_analysis="Gemini analysis",
            claude_raw_analysis="Reasoning analysis",
        ),
    )


def _make_refinement(direction_id: str, idx: int, *, risk_level: str, changed_sections: list[str]) -> RefinementResult:
    return RefinementResult(
        template=_valid_template(),
        analysis=f"Analysis {direction_id}-{idx}",
        template_changes=f"Changed {', '.join(changed_sections)}",
        should_stop=False,
        hypothesis=f"{direction_id} hypothesis {idx}",
        experiment=f"{direction_id} experiment {idx}",
        lessons=Lessons(),
        builds_on=None,
        open_problems=[],
        changed_section=changed_sections[0],
        changed_sections=changed_sections,
        target_category="subject_anchor"
        if direction_id == "D1"
        else "composition"
        if direction_id == "D2"
        else "lighting",
        direction_id=direction_id,
        direction_summary=f"Direction {direction_id}",
        failure_mechanism=f"{direction_id} failure mechanism",
        intervention_type="information_priority" if risk_level == "targeted" else "section_schema",
        risk_level=risk_level,
        expected_primary_metric="vision_subject",
        expected_tradeoff="May reduce naturalness.",
    )


def _make_sketch(
    direction_id: str, idx: int, *, mechanism: str, intervention_type: str = "information_priority"
) -> ExperimentSketch:
    return ExperimentSketch(
        hypothesis=f"{direction_id} sketch {idx}",
        target_category="subject_anchor"
        if direction_id == "D1"
        else "composition"
        if direction_id == "D2"
        else "lighting",
        failure_mechanism=mechanism,
        intervention_type=intervention_type,
        direction_id=direction_id,
        direction_summary=f"Direction {direction_id}",
        risk_level="targeted",
        expected_primary_metric="vision_subject",
        builds_on="",
    )


@pytest.mark.asyncio
async def test_propose_iteration_experiments_requests_raw_batch_and_selects_portfolio(
    monkeypatch, tmp_path: Path
) -> None:
    state = _make_state()
    ctx = RunContext(
        config=_make_config(tmp_path),
        gemini_client=MagicMock(),
        reasoning_client=MagicMock(),
        registry=MagicMock(),
        gemini_semaphore=MagicMock(),
        eval_semaphore=MagicMock(),
        services=MagicMock(),
    )
    seen_num_sketches: list[int] = []

    refinements = [
        _make_refinement("D1", 0, risk_level="targeted", changed_sections=["subject_anchor"]),
        _make_refinement("D1", 1, risk_level="bold", changed_sections=["subject_anchor", "composition_blueprint"]),
        _make_refinement("D1", 2, risk_level="bold", changed_sections=["subject_anchor", "environment_staging"]),
        _make_refinement("D2", 0, risk_level="targeted", changed_sections=["composition_blueprint"]),
        _make_refinement("D2", 1, risk_level="bold", changed_sections=["composition_blueprint", "environment_staging"]),
        _make_refinement("D2", 2, risk_level="bold", changed_sections=["composition_blueprint", "lighting_rendering"]),
        _make_refinement("D3", 0, risk_level="targeted", changed_sections=["lighting_rendering"]),
        _make_refinement("D3", 1, risk_level="bold", changed_sections=["lighting_rendering", "environment_staging"]),
        _make_refinement("D3", 2, risk_level="bold", changed_sections=["lighting_rendering", "subject_anchor"]),
    ]

    async def fake_brainstorm(*args, **kwargs):
        seen_num_sketches.append(kwargs["num_sketches"])
        return [
            _make_sketch(
                ref.direction_id, idx, mechanism=ref.failure_mechanism, intervention_type=ref.intervention_type
            )
            for idx, ref in enumerate(refinements)
        ], False

    async def fake_rank(*args, **kwargs):
        return kwargs.get("sketches", args[0])

    sketch_to_refinement = {f"{ref.direction_id} sketch {idx}": ref for idx, ref in enumerate(refinements)}

    async def fake_expand(*args, **kwargs):
        sketches = kwargs["sketches"]
        return [sketch_to_refinement.get(s.hypothesis) for s in sketches]

    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.brainstorm_experiment_sketches", fake_brainstorm)
    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.rank_experiment_sketches", fake_rank)
    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.expand_experiment_sketches", fake_expand)
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.enforce_hypothesis_diversity",
        lambda refinements, template, **_: refinements,
    )

    proposals, should_stop, _ = await _propose_iteration_experiments(state, ctx, "", "", "")

    assert should_stop is False
    assert seen_num_sketches == [18]
    # A4 quota: 3 distinct target_categories (subject_anchor, composition, lighting), cap 2 each = 6 proposals.
    # Under the old regime (no quota) this test expected 9; the under-fill is the intended mode-collapse signal.
    assert len(proposals) == 6
    assert [proposal.direction_id for proposal in proposals[:3]] == ["D1", "D2", "D3"]
    assert [proposal.risk_level for proposal in proposals[:3]] == ["targeted", "targeted", "targeted"]
    assert [proposal.risk_level for proposal in proposals[3:]] == ["bold", "bold", "bold"]
    # Each category appears exactly twice (one targeted + one bold).
    categories = [p.target_category for p in proposals]
    for cat in ("subject_anchor", "composition", "lighting"):
        assert categories.count(cat) == 2


@pytest.mark.asyncio
async def test_propose_iteration_experiments_keeps_proposal_with_removed_incumbent_section_and_caption_schema_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state = _make_state()
    state.current_template = _current_template_with_face_hands_pose()
    state.best_template = state.current_template
    ctx = RunContext(
        config=_make_config(tmp_path),
        gemini_client=MagicMock(),
        reasoning_client=MagicMock(),
        registry=MagicMock(),
        gemini_semaphore=MagicMock(),
        eval_semaphore=MagicMock(),
        services=MagicMock(),
    )

    proposed_template = _current_template_with_face_hands_pose()
    proposed_template.sections[2] = PromptSection(
        name="scene_type_and_asset_class",
        description="scene taxonomy",
        value="Scene taxonomy rules. " * 130,
    )
    proposed_template.caption_sections = ["Art Style", "Subject", "Scene Type", "Layout", "Lighting"]

    refinement = RefinementResult(
        template=proposed_template,
        analysis="Analysis",
        template_changes="Changed face_hands_pose and caption schema",
        should_stop=False,
        hypothesis="Replace anatomy-only logic with scene taxonomy.",
        experiment="Rename anatomy section and revise caption schema.",
        lessons=Lessons(),
        builds_on=None,
        open_problems=[],
        changed_section="face_hands_pose",
        changed_sections=["face_hands_pose", "caption_sections"],
        target_category="subject_anchor",
        direction_id="D1",
        direction_summary="Schema rewrite",
        failure_mechanism="Anatomy-only logic is too narrow.",
        intervention_type="section_schema",
        risk_level="bold",
        expected_primary_metric="vision_subject",
        expected_tradeoff="May reduce continuity with older captions.",
    )

    async def fake_brainstorm(*args, **kwargs):
        return [
            _make_sketch(
                "D1", 0, mechanism=refinement.failure_mechanism, intervention_type=refinement.intervention_type
            )
        ], False

    async def fake_rank(*args, **kwargs):
        return kwargs.get("sketches", args[0])

    async def fake_expand(*args, **kwargs):
        return [refinement]

    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.brainstorm_experiment_sketches", fake_brainstorm)
    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.rank_experiment_sketches", fake_rank)
    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.expand_experiment_sketches", fake_expand)
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.enforce_hypothesis_diversity",
        lambda refinements, template, **_: refinements,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.select_experiment_portfolio",
        lambda refinements, **kwargs: refinements,
    )

    proposals, should_stop, _ = await _propose_iteration_experiments(state, ctx, "", "", "")

    assert should_stop is False
    assert len(proposals) == 1
    assert proposals[0].changed_sections == ["face_hands_pose", "caption_sections"]


@pytest.mark.asyncio
async def test_propose_iteration_experiments_recovers_caption_structure_alias_from_template_diff(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state = _make_state()
    ctx = RunContext(
        config=_make_config(tmp_path),
        gemini_client=MagicMock(),
        reasoning_client=MagicMock(),
        registry=MagicMock(),
        gemini_semaphore=MagicMock(),
        eval_semaphore=MagicMock(),
        services=MagicMock(),
    )

    proposed_template = _valid_template()
    proposed_template.caption_sections = ["Art Style", "Subject", "Silhouette", "Lighting"]

    refinement = RefinementResult(
        template=proposed_template,
        analysis="Analysis",
        template_changes="Reworked caption section ordering",
        should_stop=False,
        hypothesis="If caption structure changes, section salience will improve.",
        experiment="Change caption structure ordering.",
        lessons=Lessons(),
        builds_on=None,
        open_problems=[],
        changed_section="caption_structure",
        changed_sections=["caption_structure"],
        target_category="caption_structure",
        direction_id="D1",
        direction_summary="Schema rewrite",
        failure_mechanism="Section order hides the most important locks.",
        intervention_type="section_schema",
        risk_level="targeted",
        expected_primary_metric="vision_subject",
        expected_tradeoff="May reduce continuity with prior captions.",
    )

    async def fake_brainstorm(*args, **kwargs):
        return [
            _make_sketch(
                "D1", 0, mechanism=refinement.failure_mechanism, intervention_type=refinement.intervention_type
            )
        ], False

    async def fake_rank(*args, **kwargs):
        return kwargs.get("sketches", args[0])

    async def fake_expand(*args, **kwargs):
        return [refinement]

    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.brainstorm_experiment_sketches", fake_brainstorm)
    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.rank_experiment_sketches", fake_rank)
    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.expand_experiment_sketches", fake_expand)
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.enforce_hypothesis_diversity",
        lambda refinements, template, **_: refinements,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.select_experiment_portfolio",
        lambda refinements, **kwargs: refinements,
    )

    proposals, should_stop, _ = await _propose_iteration_experiments(state, ctx, "", "", "")

    assert should_stop is False
    assert len(proposals) == 1
    assert proposals[0].changed_section == "caption_sections"
    assert proposals[0].changed_sections == ["caption_sections"]


@pytest.mark.asyncio
async def test_propose_iteration_experiments_forwards_iteration_context_to_reasoning_calls(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state = _make_state()
    state.iteration = 2
    state.plateau_counter = 4
    ctx = RunContext(
        config=_make_config(tmp_path),
        gemini_client=MagicMock(),
        reasoning_client=MagicMock(),
        registry=MagicMock(),
        gemini_semaphore=MagicMock(),
        eval_semaphore=MagicMock(),
        services=MagicMock(),
    )
    brainstorm_kwargs: dict[str, object] = {}
    expand_kwargs: dict[str, object] = {}
    refinement = _make_refinement("D1", 0, risk_level="targeted", changed_sections=["subject_anchor"])

    async def fake_brainstorm(*args, **kwargs):
        brainstorm_kwargs.update(kwargs)
        return [_make_sketch("D1", 0, mechanism=refinement.failure_mechanism)], False

    async def fake_rank(*args, **kwargs):
        return kwargs.get("sketches", args[0])

    async def fake_expand(*args, **kwargs):
        expand_kwargs.update(kwargs)
        return [refinement]

    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.brainstorm_experiment_sketches", fake_brainstorm)
    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.rank_experiment_sketches", fake_rank)
    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.expand_experiment_sketches", fake_expand)
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.enforce_hypothesis_diversity",
        lambda refinements, template, **_: refinements,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.select_experiment_portfolio",
        lambda refinements, **kwargs: refinements,
    )

    proposals, should_stop, _ = await _propose_iteration_experiments(state, ctx, "", "", "")

    assert should_stop is False
    assert len(proposals) == 1
    assert brainstorm_kwargs["iteration"] == 2
    assert brainstorm_kwargs["plateau_counter"] == 4
    assert brainstorm_kwargs["is_first_iteration"] is False
    assert expand_kwargs["iteration"] == 2
    assert expand_kwargs["plateau_counter"] == 4
    assert expand_kwargs["is_first_iteration"] is False


@pytest.mark.asyncio
async def test_propose_iteration_experiments_logs_recovery_summary(monkeypatch, tmp_path: Path, caplog) -> None:
    state = _make_state()
    ctx = RunContext(
        config=_make_config(tmp_path),
        gemini_client=MagicMock(),
        reasoning_client=MagicMock(),
        registry=MagicMock(),
        gemini_semaphore=MagicMock(),
        eval_semaphore=MagicMock(),
        services=MagicMock(),
    )

    recoverable_template = _valid_template()
    recoverable_template.caption_sections = ["Art Style", "Subject", "Silhouette", "Lighting"]

    refinements = [
        RefinementResult(
            template=recoverable_template,
            analysis="Analysis",
            template_changes="Reworked caption section ordering",
            should_stop=False,
            hypothesis="Recoverable schema change",
            experiment="Change caption structure ordering.",
            lessons=Lessons(),
            builds_on=None,
            open_problems=[],
            changed_section="caption_structure",
            changed_sections=["caption_structure"],
            target_category="caption_structure",
            direction_id="D1",
            direction_summary="Schema rewrite",
            failure_mechanism="Section order hides the most important locks.",
            intervention_type="section_schema",
            risk_level="targeted",
            expected_primary_metric="vision_subject",
            expected_tradeoff="May reduce continuity with prior captions.",
        ),
        RefinementResult(
            template=_valid_template(),
            analysis="Analysis",
            template_changes="Unknown metadata only",
            should_stop=False,
            hypothesis="Unrecoverable change metadata",
            experiment="Emit bad changed sections.",
            lessons=Lessons(),
            builds_on=None,
            open_problems=[],
            changed_section="totally_unknown_field",
            changed_sections=["totally_unknown_field"],
            target_category="general",
            direction_id="D2",
            direction_summary="Broken metadata",
            failure_mechanism="None",
            intervention_type="information_priority",
            risk_level="targeted",
            expected_primary_metric="vision_subject",
            expected_tradeoff="None",
        ),
    ]

    async def fake_brainstorm(*args, **kwargs):
        return [
            _make_sketch(
                "D1", 0, mechanism=refinements[0].failure_mechanism, intervention_type=refinements[0].intervention_type
            ),
            _make_sketch(
                "D2", 1, mechanism=refinements[1].failure_mechanism, intervention_type=refinements[1].intervention_type
            ),
        ], False

    async def fake_rank(*args, **kwargs):
        return kwargs.get("sketches", args[0])

    async def fake_expand(*args, **kwargs):
        return list(refinements)

    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.brainstorm_experiment_sketches", fake_brainstorm)
    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.rank_experiment_sketches", fake_rank)
    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.expand_experiment_sketches", fake_expand)
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.enforce_hypothesis_diversity",
        lambda refinements, template, **_: refinements,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.select_experiment_portfolio",
        lambda refinements, **kwargs: refinements,
    )

    with caplog.at_level("INFO"):
        proposals, should_stop, _ = await _propose_iteration_experiments(state, ctx, "", "", "")

    assert should_stop is False
    assert len(proposals) == 1
    assert "Proposal validation summary" in caplog.text


@pytest.mark.asyncio
async def test_propose_iteration_experiments_honors_brainstorm_stop_before_ranking_or_expansion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state = _make_state()
    ctx = RunContext(
        config=_make_config(tmp_path),
        gemini_client=MagicMock(),
        reasoning_client=MagicMock(),
        registry=MagicMock(),
        gemini_semaphore=MagicMock(),
        eval_semaphore=MagicMock(),
        services=MagicMock(),
    )
    calls: list[str] = []

    async def fake_brainstorm(*args, **kwargs):
        calls.append("brainstorm")
        return [_make_sketch("D1", 0, mechanism="Identity drift")], True

    async def fake_rank(*args, **kwargs):
        calls.append("rank")
        return []

    async def fake_expand(*args, **kwargs):
        calls.append("expand")
        return []

    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.brainstorm_experiment_sketches",
        fake_brainstorm,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.rank_experiment_sketches",
        fake_rank,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.expand_experiment_sketches",
        fake_expand,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals._should_honor_stop", lambda *args, **kwargs: True
    )

    proposals, should_stop, _ = await _propose_iteration_experiments(state, ctx, "", "", "")

    assert proposals == []
    assert should_stop is True
    assert calls == ["brainstorm"]


@pytest.mark.asyncio
async def test_propose_iteration_experiments_deduplicates_ranked_sketches_before_expansion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state = _make_state()
    ctx = RunContext(
        config=_make_config(tmp_path),
        gemini_client=MagicMock(),
        reasoning_client=MagicMock(),
        registry=MagicMock(),
        gemini_semaphore=MagicMock(),
        eval_semaphore=MagicMock(),
        services=MagicMock(),
    )
    expand_inputs: list[list[ExperimentSketch]] = []

    async def fake_brainstorm(*args, **kwargs):
        return [
            _make_sketch("D1", 0, mechanism="Identity drift"),
            _make_sketch("D1", 1, mechanism="Identity drift"),
        ], False

    async def fake_rank(*args, **kwargs):
        return [
            _make_sketch("D1", 0, mechanism="Identity drift"),
            _make_sketch("D1", 1, mechanism="Identity drift"),
        ]

    async def fake_expand(*args, **kwargs):
        expand_inputs.append(list(kwargs["sketches"]))
        return [_make_refinement("D1", 0, risk_level="targeted", changed_sections=["subject_anchor"])]

    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.brainstorm_experiment_sketches",
        fake_brainstorm,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.rank_experiment_sketches",
        fake_rank,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.expand_experiment_sketches",
        fake_expand,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.enforce_hypothesis_diversity",
        lambda refinements, template, **_: refinements,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.select_experiment_portfolio",
        lambda refinements, **kwargs: refinements,
    )

    proposals, should_stop, _ = await _propose_iteration_experiments(state, ctx, "", "", "")

    assert should_stop is False
    assert len(proposals) == 1
    assert len(expand_inputs) == 1
    assert len(expand_inputs[0]) == 1
