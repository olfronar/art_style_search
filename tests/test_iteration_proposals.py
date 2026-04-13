"""Unit tests for workflow.iteration_proposals."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from art_style_search.config import Config
from art_style_search.contracts import Lessons, RefinementResult
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
            PromptSection(name="style_foundation", description="rules", value="Shared style rules."),
            PromptSection(name="subject_anchor", description="subject", value="Subject rules."),
            PromptSection(name="composition_blueprint", description="layout", value="Layout rules."),
            PromptSection(name="lighting_rendering", description="light", value="Lighting rules."),
            PromptSection(name="environment_staging", description="environment", value="Environment rules."),
        ],
        caption_sections=["Art Style", "Subject", "Composition", "Lighting"],
        caption_length_target=500,
    )


def _current_template_with_face_hands_pose() -> PromptTemplate:
    return PromptTemplate(
        sections=[
            PromptSection(name="style_foundation", description="rules", value="Shared style rules."),
            PromptSection(name="subject_anchor", description="subject", value="Subject rules."),
            PromptSection(name="face_hands_pose", description="anatomy", value="Pose rules."),
            PromptSection(name="global_layout_grid", description="layout", value="Layout rules."),
            PromptSection(name="palette_temperature", description="palette", value="Palette rules."),
        ],
        caption_sections=["Art Style", "Subject", "Pose", "Layout"],
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
        target_category="subject_anchor" if direction_id == "D1" else "composition" if direction_id == "D2" else "lighting",
        direction_id=direction_id,
        direction_summary=f"Direction {direction_id}",
        failure_mechanism=f"{direction_id} failure mechanism",
        intervention_type="information_priority" if risk_level == "targeted" else "section_schema",
        risk_level=risk_level,
        expected_primary_metric="vision_subject",
        expected_tradeoff="May reduce naturalness.",
    )


@pytest.mark.asyncio
async def test_propose_iteration_experiments_requests_raw_batch_and_selects_portfolio(monkeypatch, tmp_path: Path) -> None:
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
    seen_num_experiments: list[int] = []

    async def fake_propose_experiments(*args, **kwargs):
        seen_num_experiments.append(kwargs["num_experiments"])
        return [
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

    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.propose_experiments", fake_propose_experiments)
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.enforce_hypothesis_diversity",
        lambda refinements, template: refinements,
    )

    proposals, should_stop = await _propose_iteration_experiments(state, ctx, "", "", "")

    assert should_stop is False
    assert seen_num_experiments == [9]
    assert len(proposals) == 9
    assert [proposal.direction_id for proposal in proposals[:3]] == ["D1", "D2", "D3"]
    assert [proposal.risk_level for proposal in proposals[:3]] == ["targeted", "targeted", "targeted"]
    assert [proposal.risk_level for proposal in proposals[3:]] == ["bold", "bold", "bold", "bold", "bold", "bold"]


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

    proposed_template = PromptTemplate(
        sections=[
            PromptSection(name="style_foundation", description="rules", value="Shared style rules."),
            PromptSection(name="subject_anchor", description="subject", value="Subject rules."),
            PromptSection(
                name="scene_type_and_asset_class",
                description="scene taxonomy",
                value="Scene taxonomy rules.",
            ),
            PromptSection(name="global_layout_grid", description="layout", value="Layout rules."),
            PromptSection(name="palette_temperature", description="palette", value="Palette rules."),
        ],
        caption_sections=["Art Style", "Subject", "Scene Type", "Layout"],
        caption_length_target=500,
    )

    async def fake_propose_experiments(*args, **kwargs):
        return [
            RefinementResult(
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
        ]

    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.propose_experiments", fake_propose_experiments)
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.enforce_hypothesis_diversity",
        lambda refinements, template: refinements,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.select_experiment_portfolio",
        lambda refinements, **kwargs: refinements,
    )

    proposals, should_stop = await _propose_iteration_experiments(state, ctx, "", "", "")

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

    async def fake_propose_experiments(*args, **kwargs):
        return [
            RefinementResult(
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
        ]

    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.propose_experiments", fake_propose_experiments)
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.enforce_hypothesis_diversity",
        lambda refinements, template: refinements,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.select_experiment_portfolio",
        lambda refinements, **kwargs: refinements,
    )

    proposals, should_stop = await _propose_iteration_experiments(state, ctx, "", "", "")

    assert should_stop is False
    assert len(proposals) == 1
    assert proposals[0].changed_section == "caption_sections"
    assert proposals[0].changed_sections == ["caption_sections"]


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

    async def fake_propose_experiments(*args, **kwargs):
        return [
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

    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.propose_experiments", fake_propose_experiments)
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.enforce_hypothesis_diversity",
        lambda refinements, template: refinements,
    )
    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.select_experiment_portfolio",
        lambda refinements, **kwargs: refinements,
    )

    with caplog.at_level("INFO"):
        proposals, should_stop = await _propose_iteration_experiments(state, ctx, "", "", "")

    assert should_stop is False
    assert len(proposals) == 1
    assert "Proposal validation summary" in caplog.text
