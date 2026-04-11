"""End-to-end orchestration tests for loop.py with fake external dependencies.

Exercises the full ``run()`` entry point with monkeypatched API calls so that
no real network or GPU work happens.  Three scenarios:

1. A converged run resumes and exits immediately.
2. A fresh run completes zero-step + one iteration.
3. A pre-built state with a high plateau triggers exploration.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from art_style_search.config import Config
from art_style_search.contracts import Lessons, RefinementResult
from art_style_search.evaluate import aggregate
from art_style_search.state import save_state
from art_style_search.types import (
    AggregatedMetrics,
    Caption,
    KnowledgeBase,
    LoopState,
    MetricScores,
    PromptSection,
    PromptTemplate,
    ReviewResult,
    StyleProfile,
    VisionDimensionScore,
    VisionScores,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A template that passes validate_template(): first section is style_foundation,
# first caption section is Art Style, >= 4 sections, caption_length_target in range.
_SECTION_NAMES = [
    ("style_foundation", "Core style identity"),
    ("color_palette", "Dominant colors"),
    ("composition", "Layout and framing"),
    ("technique", "Art medium and rendering"),
    ("lighting", "Light sources and shadows"),
]


def _valid_template() -> PromptTemplate:
    """Build a PromptTemplate that passes all validate_template checks."""
    sections = [
        PromptSection(
            name=name,
            description=desc,
            value=f"Detailed {desc.lower()} with embedded style rules. " * 4,
        )
        for name, desc in _SECTION_NAMES
    ]
    return PromptTemplate(
        sections=sections,
        negative_prompt="photorealistic, 3D render",
        caption_sections=["Art Style", "Color Palette", "Composition", "Technique"],
        caption_length_target=500,
    )


def _make_ref_images(directory: Path, n: int = 4) -> list[Path]:
    """Create small test PNG files and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n):
        p = directory / f"ref_{i:02d}.png"
        img = Image.new("RGB", (64, 64), color=(i * 50 % 256, 100, 150))
        img.save(p)
        paths.append(p)
    return paths


def _make_config(tmp_path: Path, *, max_iterations: int = 1, num_refs: int = 4) -> Config:
    """Build a Config rooted in tmp_path with minimal settings."""
    ref_dir = tmp_path / "refs"
    _make_ref_images(ref_dir, n=num_refs)

    run_dir = tmp_path / "runs" / "test_run"
    output_dir = run_dir / "outputs"
    log_dir = run_dir / "logs"
    state_file = run_dir / "state.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        reference_dir=ref_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        state_file=state_file,
        run_dir=run_dir,
        max_iterations=max_iterations,
        plateau_window=5,
        num_branches=2,
        aspect_ratio="1:1",
        num_fixed_refs=num_refs,
        caption_model="fake-caption-model",
        generator_model="fake-gen-model",
        reasoning_model="fake-reasoning-model",
        reasoning_provider="anthropic",
        reasoning_base_url="",
        gemini_concurrency=5,
        eval_concurrency=2,
        seed=42,
        protocol="classic",
        anthropic_api_key="fake-key",
        google_api_key="fake-key",
        zai_api_key="",
        openai_api_key="",
    )


def _make_style_profile() -> StyleProfile:
    return StyleProfile(
        color_palette="Muted earth tones.",
        composition="Low horizon, asymmetric balance.",
        technique="Wet-on-wet watercolor.",
        mood_atmosphere="Contemplative, quiet.",
        subject_matter="Rural landscapes.",
        influences="Andrew Wyeth.",
        gemini_raw_analysis="Gemini analysis text",
        claude_raw_analysis="Claude analysis text",
    )


def _make_aggregated(seed: float = 0.0) -> AggregatedMetrics:
    return AggregatedMetrics(
        dreamsim_similarity_mean=0.65 + seed * 0.01,
        dreamsim_similarity_std=0.03,
        hps_score_mean=0.24 + seed * 0.002,
        hps_score_std=0.01,
        aesthetics_score_mean=5.5 + seed * 0.1,
        aesthetics_score_std=0.4,
        color_histogram_mean=0.50 + seed * 0.01,
        color_histogram_std=0.02,
        ssim_mean=0.50 + seed * 0.01,
        ssim_std=0.02,
        completion_rate=1.0,
    )


def _make_metric_scores(seed: float = 0.0) -> MetricScores:
    return MetricScores(
        dreamsim_similarity=0.65 + seed * 0.01,
        hps_score=0.24 + seed * 0.002,
        aesthetics_score=5.5 + seed * 0.1,
        color_histogram=0.50 + seed * 0.01,
        ssim=0.50 + seed * 0.01,
    )


def _make_vision_scores() -> VisionScores:
    return VisionScores(
        style=VisionDimensionScore("style", 0.5, "partial match"),
        subject=VisionDimensionScore("subject", 0.5, "partial match"),
        composition=VisionDimensionScore("composition", 0.5, "partial match"),
    )


def _fake_caption(ref_path: Path) -> Caption:
    return Caption(
        image_path=ref_path,
        text=(
            "[Art Style] Watercolor painting with muted earth tones. "
            "[Color Palette] Ochre, burnt sienna, slate blue. "
            "[Composition] Low horizon, asymmetric balance. "
            "[Technique] Wet-on-wet watercolor rendering. " * 3
        ),
    )


def _build_refinement_result(template: PromptTemplate, idx: int) -> RefinementResult:
    """Build a RefinementResult that propose_experiments would return."""
    categories = ["color_palette", "composition", "technique", "lighting", "texture"]
    cat = categories[idx % len(categories)]
    return RefinementResult(
        template=template,
        analysis=f"Analysis for experiment {idx}",
        template_changes=f"Changed section {cat}",
        should_stop=False,
        hypothesis=f"Testing {cat} improvement",
        experiment=f"Modify {cat} section for better fidelity",
        lessons=Lessons(confirmed="", rejected="", new_insight=""),
        builds_on=None,
        open_problems=[],
        changed_section=cat,
        target_category=cat,
    )


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


def _apply_all_patches(monkeypatch, tmp_path: Path, ref_paths: list[Path]):
    """Monkeypatch all external dependencies for loop.run()."""

    async def fake_caption_references(
        reference_paths, *, model=None, client=None, cache_dir=None, semaphore=None, prompt=None, cache_key=""
    ):
        return [_fake_caption(p) for p in reference_paths]

    async def fake_pairwise(pairs_a, pairs_b, *, client=None, model=None, semaphore=None):
        return "A is slightly better", 0.6

    fake_services = SimpleNamespace(
        captioning=SimpleNamespace(caption_references=fake_caption_references),
        evaluation=SimpleNamespace(pairwise_compare=fake_pairwise),
    )

    # 1. Mock _setup_run_context to avoid real client construction + model loading
    from art_style_search.workflow.context import RunContext

    async def fake_setup_run_context(config):
        return RunContext(
            config=config,
            gemini_client=MagicMock(),
            reasoning_client=MagicMock(),
            registry=MagicMock(),
            gemini_semaphore=asyncio.Semaphore(5),
            eval_semaphore=asyncio.Semaphore(2),
            services=fake_services,
        )

    monkeypatch.setattr("art_style_search.loop._setup_run_context", fake_setup_run_context)

    # 2. Mock analyze_style (zero-step analysis)
    async def fake_analyze_style(
        reference_paths, captions, *, gemini_client, reasoning_client, caption_model, reasoning_model, cache_path
    ):
        return _make_style_profile(), _valid_template()

    monkeypatch.setattr("art_style_search.workflow.zero_step.analyze_style", fake_analyze_style)

    # 3. Mock propose_initial_templates (zero-step diverse templates)
    async def fake_propose_initial_templates(style_profile, num_branches, *, client, model):
        return [_valid_template() for _ in range(num_branches)]

    monkeypatch.setattr("art_style_search.prompt.initial.propose_initial_templates", fake_propose_initial_templates)
    monkeypatch.setattr("art_style_search.workflow.zero_step.propose_initial_templates", fake_propose_initial_templates)

    # 4. Mock run_experiment (the core per-experiment pipeline)
    call_counter = {"n": 0}

    async def fake_run_experiment(
        experiment_id,
        template,
        iteration,
        fixed_refs,
        config,
        *,
        services,
        last_results,
        hypothesis="",
        experiment_desc="",
        analysis="",
        template_changes="",
        changed_section="",
        target_category="",
    ):
        from art_style_search.types import IterationResult

        call_counter["n"] += 1
        n_imgs = len(fixed_refs)

        # Create small fake generated images
        gen_dir = config.output_dir / f"iter_{iteration:03d}" / f"exp_{experiment_id}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        gen_paths = []
        captions = []
        scores = []
        for i in range(n_imgs):
            p = gen_dir / f"{i:02d}.png"
            Image.new("RGB", (64, 64), color="blue").save(p)
            gen_paths.append(p)
            captions.append(_fake_caption(fixed_refs[i]))
            scores.append(_make_metric_scores(seed=float(experiment_id + i * 0.1)))

        agg = aggregate(scores)

        return IterationResult(
            branch_id=experiment_id,
            iteration=iteration,
            template=template,
            rendered_prompt=template.render(),
            image_paths=gen_paths,
            per_image_scores=scores,
            aggregated=agg,
            claude_analysis=analysis or "fake analysis",
            template_changes=template_changes or "fake changes",
            kept=False,
            hypothesis=hypothesis or f"Hypothesis {experiment_id}",
            experiment=experiment_desc or f"Experiment {experiment_id}",
            vision_feedback="Style=PARTIAL, Subject=PARTIAL, Composition=PARTIAL",
            roundtrip_feedback="Image (ref_00.png): DS=0.65",
            iteration_captions=captions,
            n_images_attempted=n_imgs,
            n_images_succeeded=n_imgs,
            changed_section=changed_section,
            target_category=target_category,
        )

    monkeypatch.setattr("art_style_search.workflow.zero_step.run_experiment", fake_run_experiment)
    monkeypatch.setattr("art_style_search.workflow.iteration_execution.run_experiment", fake_run_experiment)

    # 5. Mock propose_experiments (per-iteration reasoning)
    async def fake_propose_experiments(
        style_profile,
        current_template,
        knowledge_base,
        best_metrics,
        last_results,
        *,
        client,
        model,
        num_experiments,
        vision_feedback="",
        roundtrip_feedback="",
        caption_diffs="",
    ):
        return [_build_refinement_result(_valid_template(), i) for i in range(num_experiments)]

    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.propose_experiments",
        fake_propose_experiments,
    )

    # 6. Mock enforce_hypothesis_diversity (pass-through)
    def fake_enforce_diversity(results, template):
        return results

    monkeypatch.setattr(
        "art_style_search.workflow.iteration_proposals.enforce_hypothesis_diversity",
        fake_enforce_diversity,
    )

    # 7. Mock validate_template (always valid)
    def fake_validate_template(template, changed_section=""):
        return []

    monkeypatch.setattr("art_style_search.workflow.zero_step.validate_template", fake_validate_template)
    monkeypatch.setattr("art_style_search.workflow.iteration_proposals.validate_template", fake_validate_template)

    # 8. Mock synthesize_templates (returns merged template)
    async def fake_synthesize_templates(experiments, style_profile, *, client, model):
        return _valid_template(), "Synthesis hypothesis: merge best aspects"

    monkeypatch.setattr(
        "art_style_search.workflow.iteration_execution.synthesize_templates",
        fake_synthesize_templates,
    )

    # 9. Mock review_iteration
    async def fake_review(experiments, proposals, baseline_metrics, knowledge_base, *, client, model):
        return ReviewResult(
            experiment_assessments=["SIGNAL"] * len(experiments),
            noise_vs_signal="All movements appear real.",
            strategic_guidance="Focus on color palette next.",
            recommended_categories=["color_palette"],
        )

    monkeypatch.setattr("art_style_search.workflow.iteration_execution.review_iteration", fake_review)

    # 10. Mock build_ref_gen_pairs (used by pairwise comparison in loop)
    def fake_build_pairs(result):
        pairs = []
        for i, gp in enumerate(result.image_paths):
            if i < len(result.iteration_captions):
                pairs.append((result.iteration_captions[i].image_path, gp))
        return pairs

    monkeypatch.setattr("art_style_search.workflow.iteration_execution.build_ref_gen_pairs", fake_build_pairs)

    return call_counter


# ---------------------------------------------------------------------------
# Test 1: converged run skips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_converged_run_skips(tmp_path: Path, monkeypatch):
    """A run whose state.json has converged=True should return early without experiments."""
    config = _make_config(tmp_path, max_iterations=5)
    ref_paths = sorted((config.reference_dir).iterdir())
    call_counter = _apply_all_patches(monkeypatch, tmp_path, ref_paths)

    # Pre-create a converged state
    template = _valid_template()
    state = LoopState(
        iteration=3,
        current_template=template,
        best_template=template,
        best_metrics=_make_aggregated(seed=1.0),
        knowledge_base=KnowledgeBase(),
        captions=[_fake_caption(p) for p in ref_paths],
        style_profile=_make_style_profile(),
        fixed_references=ref_paths,
        converged=True,
        convergence_reason=None,  # will be set via string
        global_best_prompt=template.render(),
        global_best_metrics=_make_aggregated(seed=1.0),
    )
    # Set convergence reason directly
    from art_style_search.types import ConvergenceReason

    state.convergence_reason = ConvergenceReason.PLATEAU
    save_state(state, config.state_file)

    # Also write a manifest so _verify_manifest doesn't fail
    from art_style_search.state import save_manifest
    from art_style_search.types import RunManifest

    manifest = RunManifest(
        protocol_version="classic",
        seed=42,
        cli_args={},
        model_names={},
        reasoning_provider="anthropic",
        git_sha=None,
        python_version="3.12",
        platform="test",
        timestamp_utc="2025-01-01T00:00:00",
        reference_image_hashes={p.name: "fakehash" for p in ref_paths},
        num_fixed_refs=len(ref_paths),
        uv_lock_hash=None,
    )
    save_manifest(manifest, config.run_dir / "run_manifest.json")

    from art_style_search.loop import run

    result = await run(config)

    # Should have returned early — no experiments run
    assert result.converged is True
    assert call_counter["n"] == 0


# ---------------------------------------------------------------------------
# Test 2: zero-step + one iteration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_step_and_one_iteration(tmp_path: Path, monkeypatch):
    """A fresh run with max_iterations=1 completes zero-step + 1 iteration."""
    config = _make_config(tmp_path, max_iterations=1, num_refs=4)
    ref_paths = sorted(config.reference_dir.iterdir())
    call_counter = _apply_all_patches(monkeypatch, tmp_path, ref_paths)

    from art_style_search.loop import run

    result = await run(config)

    # State should exist on disk
    assert config.state_file.exists()

    # Should have converged (max_iterations reached)
    assert result.converged is True

    # Experiments were actually run (zero-step evals + iteration 1 experiments)
    assert call_counter["n"] > 0

    # The state should have a best_template
    assert result.best_template is not None
    assert len(result.best_template.sections) >= 4

    # Global best prompt should be set
    assert result.global_best_prompt

    # Check that state.json is valid JSON
    state_data = json.loads(config.state_file.read_text())
    assert "iteration" in state_data
    assert "best_template" in state_data

    # Experiment history should be populated
    assert len(result.experiment_history) > 0


# ---------------------------------------------------------------------------
# Test 3: plateau triggers exploration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plateau_triggers_exploration(tmp_path: Path, monkeypatch):
    """Pre-built state with high plateau_counter triggers exploration (second-best adopted)."""
    config = _make_config(tmp_path, max_iterations=10, num_refs=4)
    ref_paths = sorted(config.reference_dir.iterdir())
    call_counter = _apply_all_patches(monkeypatch, tmp_path, ref_paths)

    # Pre-create state at iteration 5 with plateau_counter = 1
    # (After the next iteration fails to improve, counter becomes 2 → even → exploration)
    template = _valid_template()
    high_agg = _make_aggregated(seed=100.0)  # very high baseline so nothing beats it
    state = LoopState(
        iteration=5,
        current_template=template,
        best_template=template,
        best_metrics=high_agg,
        knowledge_base=KnowledgeBase(),
        captions=[_fake_caption(p) for p in ref_paths],
        style_profile=_make_style_profile(),
        fixed_references=ref_paths,
        plateau_counter=1,  # will become 2 after plateau
        global_best_prompt=template.render(),
        global_best_metrics=high_agg,
        experiment_history=[],
        last_iteration_results=[],
    )
    save_state(state, config.state_file)

    # Write manifest
    from art_style_search.state import save_manifest
    from art_style_search.types import RunManifest

    manifest = RunManifest(
        protocol_version="classic",
        seed=42,
        cli_args={},
        model_names={},
        reasoning_provider="anthropic",
        git_sha=None,
        python_version="3.12",
        platform="test",
        timestamp_utc="2025-01-01T00:00:00",
        reference_image_hashes={p.name: "fakehash" for p in ref_paths},
        num_fixed_refs=len(ref_paths),
        uv_lock_hash=None,
    )
    save_manifest(manifest, config.run_dir / "run_manifest.json")

    # Limit to a single iteration after the resumed state so we can inspect
    # what happened. max_iterations=6 means iteration index 5 runs and then
    # the for-loop range(5, 6) gives exactly one iteration pass.
    config_one = Config(
        reference_dir=config.reference_dir,
        output_dir=config.output_dir,
        log_dir=config.log_dir,
        state_file=config.state_file,
        run_dir=config.run_dir,
        max_iterations=6,  # range(5, 6) => 1 iteration
        plateau_window=config.plateau_window,
        num_branches=2,
        aspect_ratio=config.aspect_ratio,
        num_fixed_refs=config.num_fixed_refs,
        caption_model=config.caption_model,
        generator_model=config.generator_model,
        reasoning_model=config.reasoning_model,
        reasoning_provider=config.reasoning_provider,
        reasoning_base_url=config.reasoning_base_url,
        gemini_concurrency=config.gemini_concurrency,
        eval_concurrency=config.eval_concurrency,
        seed=config.seed,
        protocol=config.protocol,
        anthropic_api_key=config.anthropic_api_key,
        google_api_key=config.google_api_key,
        zai_api_key=config.zai_api_key,
        openai_api_key=config.openai_api_key,
    )

    from art_style_search.loop import run

    result = await run(config_one)

    # Experiments should have been run
    assert call_counter["n"] > 0

    # The plateau triggered exploration: plateau was 1, incremented to 2 (even),
    # with >= 2 experiments -> exploration resets plateau to 1
    # OR the plateau just incremented to 2 and exploration fired, resetting to 1.
    # Either way, the plateau_counter should be 1 (exploration reset) since
    # the baseline was set very high and no experiment can beat it.
    assert result.plateau_counter == 1, (
        f"Expected plateau_counter=1 after exploration reset, got {result.plateau_counter}"
    )

    # The template should have been updated (exploration adopts second-best)
    # Since exploration fires, the current_template is the second-best experiment's template
    assert result.current_template is not None

    # best_metrics should NOT have changed (exploration preserves it)
    assert result.best_metrics == high_agg
