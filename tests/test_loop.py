"""Unit tests for art_style_search.loop helpers."""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from art_style_search.loop import (
    IterationRanking,
    RunContext,
    _apply_best_result,
    _apply_exploration_result,
    _apply_iteration_result,
    _confirmatory_validation,
    _discover_images,
    _run_synthesis_experiment,
    _sample,
    _sanitize_initial_templates,
    _score_and_rank,
    _should_honor_stop,
    _split_information_barrier,
)
from art_style_search.scoring import composite_score, improvement_epsilon
from art_style_search.types import (
    AggregatedMetrics,
    CategoryProgress,
    IterationResult,
    KnowledgeBase,
    MetricScores,
    PromotionTestResult,
    PromptSection,
    PromptTemplate,
    ReplicatedEvaluation,
)
from tests.conftest import make_loop_state, make_prompt_template

# ---------------------------------------------------------------------------
# _discover_images
# ---------------------------------------------------------------------------


class TestDiscoverImages:
    """_discover_images should return only image files, sorted."""

    def test_filters_and_sorts(self, tmp_path: Path) -> None:
        # Create mixed files
        (tmp_path / "b.png").touch()
        (tmp_path / "a.jpg").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / "c.jpeg").touch()

        result = _discover_images(tmp_path)

        names = [p.name for p in result]
        assert "notes.txt" not in names
        assert names == sorted(names)
        assert set(names) == {"a.jpg", "b.png", "c.jpeg"}

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert _discover_images(tmp_path) == []

    def test_no_images(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.csv").touch()
        assert _discover_images(tmp_path) == []

    def test_all_supported_extensions(self, tmp_path: Path) -> None:
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
            (tmp_path / f"img{ext}").touch()

        result = _discover_images(tmp_path)
        assert len(result) == 6

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        (tmp_path / "photo.PNG").touch()
        (tmp_path / "photo.Jpg").touch()

        result = _discover_images(tmp_path)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _sample
# ---------------------------------------------------------------------------


class TestSample:
    """_sample should return all items when count <= max, else a random subset."""

    def _make_paths(self, n: int) -> list[Path]:
        return [Path(f"/fake/{i}.png") for i in range(n)]

    def test_smaller_than_max_returns_all(self) -> None:
        items = self._make_paths(3)
        result = _sample(items, max_count=5)
        assert result is items  # exact same list object, not a copy

    def test_exact_size_returns_all(self) -> None:
        items = self._make_paths(5)
        result = _sample(items, max_count=5)
        assert result is items

    def test_larger_than_max_returns_correct_count(self) -> None:
        items = self._make_paths(10)
        result = _sample(items, max_count=4)
        assert len(result) == 4
        # All returned items should come from the original list
        for p in result:
            assert p in items

    def test_larger_than_max_no_duplicates(self) -> None:
        items = self._make_paths(20)
        result = _sample(items, max_count=7)
        assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# Helpers for promotion / scoring test classes
# ---------------------------------------------------------------------------


def _make_agg(
    dreamsim: float = 0.70,
    color: float = 0.50,
    ssim: float = 0.50,
    hps: float = 0.25,
    aes: float = 5.0,
) -> AggregatedMetrics:
    """Build an AggregatedMetrics with controllable key values and low std."""
    return AggregatedMetrics(
        dreamsim_similarity_mean=dreamsim,
        dreamsim_similarity_std=0.02,
        hps_score_mean=hps,
        hps_score_std=0.01,
        aesthetics_score_mean=aes,
        aesthetics_score_std=0.3,
        color_histogram_mean=color,
        color_histogram_std=0.02,
        ssim_mean=ssim,
        ssim_std=0.02,
    )


def _make_result(branch_id: int, agg: AggregatedMetrics, iteration: int = 1) -> IterationResult:
    """Build an IterationResult with the given aggregated metrics."""
    return IterationResult(
        branch_id=branch_id,
        iteration=iteration,
        template=make_prompt_template(),
        rendered_prompt="test prompt",
        image_paths=[Path(f"/fake/img_{i}.png") for i in range(3)],
        per_image_scores=[MetricScores(dreamsim_similarity=0.7, hps_score=0.25, aesthetics_score=5.0)] * 3,
        aggregated=agg,
        claude_analysis="analysis",
        template_changes="changes",
        kept=False,
        hypothesis="test hypothesis",
    )


def _make_config(tmp_path: Path, *, max_iterations: int = 20, plateau_window: int = 5, protocol: str = "classic"):
    """Build a minimal Config for testing."""
    from art_style_search.config import Config

    return Config(
        reference_dir=tmp_path / "refs",
        output_dir=tmp_path / "outputs",
        log_dir=tmp_path / "logs",
        state_file=tmp_path / "state.json",
        run_dir=tmp_path,
        max_iterations=max_iterations,
        plateau_window=plateau_window,
        num_branches=3,
        aspect_ratio="1:1",
        num_fixed_refs=20,
        caption_model="gemini-2.5-pro-preview-06-05",
        generator_model="gemini-2.0-flash-preview-image-generation",
        reasoning_model="claude-sonnet-4-20250514",
        reasoning_provider="anthropic",
        reasoning_base_url="",
        gemini_concurrency=5,
        eval_concurrency=4,
        seed=42,
        protocol=protocol,
        anthropic_api_key="test",
        google_api_key="test",
        zai_api_key="",
        openai_api_key="",
    )


def _make_valid_template() -> PromptTemplate:
    sections = [
        PromptSection("style_foundation", "Core style", "Shared style rules. " * 4),
        PromptSection("color_palette", "Colors", "Color guidance. " * 4),
        PromptSection("composition", "Composition", "Composition guidance. " * 4),
        PromptSection("technique", "Technique", "Technique guidance. " * 4),
    ]
    return PromptTemplate(
        sections=sections,
        negative_prompt="avoid blur",
        caption_sections=["Art Style", "Color Palette", "Composition", "Technique"],
        caption_length_target=500,
    )


# ---------------------------------------------------------------------------
# TestScoreAndRank
# ---------------------------------------------------------------------------


class TestScoreAndRank:
    """Tests for _score_and_rank(exp_results, state)."""

    def test_picks_best_by_adaptive_score(self) -> None:
        """Create 2 IterationResults with different metrics; best_exp should be the higher one."""
        agg_low = _make_agg(dreamsim=0.50, color=0.30)
        agg_high = _make_agg(dreamsim=0.80, color=0.70)
        r_low = _make_result(branch_id=0, agg=agg_low)
        r_high = _make_result(branch_id=1, agg=agg_high)
        state = make_loop_state()

        ranking = _score_and_rank([r_low, r_high], state)

        assert ranking.best_exp is r_high

    def test_uses_composite_for_baseline(self) -> None:
        """Baseline score should match composite_score(state.best_metrics)."""
        state = make_loop_state()
        agg = _make_agg(dreamsim=0.70)
        result = _make_result(branch_id=0, agg=agg)

        ranking = _score_and_rank([result], state)

        assert state.best_metrics is not None
        expected_baseline = composite_score(state.best_metrics)
        assert ranking.baseline_score == expected_baseline


# ---------------------------------------------------------------------------
# TestApplyBestResult
# ---------------------------------------------------------------------------


class TestApplyBestResult:
    """Tests for _apply_best_result(state, result)."""

    def test_updates_global_best(self) -> None:
        """A result that beats global_best should update global_best_prompt and global_best_metrics."""
        state = make_loop_state()
        # Set global_best_metrics to something low
        state.global_best_metrics = _make_agg(dreamsim=0.30, color=0.20)
        state.global_best_prompt = "old prompt"

        high_agg = _make_agg(dreamsim=0.90, color=0.80)
        result = _make_result(branch_id=0, agg=high_agg)

        _apply_best_result(state, result)

        assert state.global_best_metrics is high_agg
        assert state.global_best_prompt == result.rendered_prompt

    def test_updates_current_and_best_template(self) -> None:
        """Both current_template and best_template should be updated to the result's template."""
        state = make_loop_state()
        state.global_best_metrics = None  # so any result beats it

        agg = _make_agg(dreamsim=0.75)
        result = _make_result(branch_id=0, agg=agg)

        _apply_best_result(state, result)

        assert state.current_template is result.template
        assert state.best_template is result.template
        assert state.best_metrics is result.aggregated


# ---------------------------------------------------------------------------
# TestApplyExplorationResult
# ---------------------------------------------------------------------------


class TestApplyExplorationResult:
    """Tests for _apply_exploration_result(state, result)."""

    def test_preserves_best_metrics(self) -> None:
        """Exploration must NOT change best_metrics or global_best_metrics."""
        state = make_loop_state()
        original_best_metrics = state.best_metrics
        original_global_metrics = state.global_best_metrics

        agg = _make_agg(dreamsim=0.99, color=0.99)
        result = _make_result(branch_id=0, agg=agg)

        _apply_exploration_result(state, result)

        assert state.best_metrics is original_best_metrics
        assert state.global_best_metrics is original_global_metrics
        # current_template should be updated, best_template must NOT change
        assert state.current_template is result.template
        assert state.best_template is not result.template


# ---------------------------------------------------------------------------
# TestApplyIterationResult
# ---------------------------------------------------------------------------


class TestApplyIterationResult:
    """Tests for _apply_iteration_result(state, ranking, config)."""

    def test_improvement_promotes(self, tmp_path: Path) -> None:
        """best_score > baseline + epsilon should promote the result."""
        state = make_loop_state()
        config = _make_config(tmp_path)

        # Create a low baseline so the candidate clearly beats it
        low_agg = _make_agg(dreamsim=0.30, color=0.20)
        state.best_metrics = low_agg
        state.global_best_metrics = low_agg

        high_agg = _make_agg(dreamsim=0.85, color=0.80)
        best_result = _make_result(branch_id=0, agg=high_agg)

        baseline_score = composite_score(low_agg)
        best_score = composite_score(high_agg)
        eps = improvement_epsilon(baseline_score)
        assert best_score > baseline_score + eps, "Test setup: candidate must beat baseline + epsilon"

        ranking = IterationRanking(
            exp_results=[best_result],
            adaptive_scores={id(best_result): best_score},
            best_exp=best_result,
            best_score=best_score,
            baseline_score=baseline_score,
            epsilon=eps,
        )

        _apply_iteration_result(state, ranking, config)

        assert state.plateau_counter == 0
        assert state.best_metrics is high_agg
        assert best_result.kept is True

    def test_plateau_increments_counter(self, tmp_path: Path) -> None:
        """best_score < baseline + epsilon should increment plateau_counter."""
        state = make_loop_state()
        config = _make_config(tmp_path)
        state.plateau_counter = 0

        # Make baseline and candidate nearly identical
        baseline_agg = _make_agg(dreamsim=0.70, color=0.50)
        state.best_metrics = baseline_agg

        candidate_agg = _make_agg(dreamsim=0.70, color=0.50)
        result = _make_result(branch_id=0, agg=candidate_agg)

        baseline_score = composite_score(baseline_agg)
        best_score = composite_score(candidate_agg)
        eps = improvement_epsilon(baseline_score)

        ranking = IterationRanking(
            exp_results=[result],
            adaptive_scores={id(result): best_score},
            best_exp=result,
            best_score=best_score,
            baseline_score=baseline_score,
            epsilon=eps,
        )

        _apply_iteration_result(state, ranking, config)

        assert state.plateau_counter == 1

    def test_exploration_on_even_plateau(self, tmp_path: Path) -> None:
        """At plateau_counter=2 with 2+ experiments, second-best should be adopted."""
        state = make_loop_state()
        config = _make_config(tmp_path)
        # Pre-set plateau to 1 so after increment it becomes 2 (even, triggers exploration)
        state.plateau_counter = 1

        baseline_agg = _make_agg(dreamsim=0.70, color=0.50)
        state.best_metrics = baseline_agg

        # Two candidates, neither beats baseline
        agg_a = _make_agg(dreamsim=0.70, color=0.50)
        agg_b = _make_agg(dreamsim=0.69, color=0.49)
        result_a = _make_result(branch_id=0, agg=agg_a)
        result_b = _make_result(branch_id=1, agg=agg_b)

        baseline_score = composite_score(baseline_agg)
        score_a = composite_score(agg_a)
        score_b = composite_score(agg_b)
        eps = improvement_epsilon(baseline_score)

        # result_a has higher adaptive score → it's best; result_b is second-best
        ranking = IterationRanking(
            exp_results=[result_a, result_b],
            adaptive_scores={id(result_a): score_a + 0.01, id(result_b): score_b},
            best_exp=result_a,
            best_score=score_a,
            baseline_score=baseline_score,
            epsilon=eps,
        )

        original_best_metrics = state.best_metrics

        _apply_iteration_result(state, ranking, config)

        # Exploration should adopt second-best template but preserve best_metrics
        assert state.plateau_counter == 1  # reset to _EXPLORATION_RESET_PLATEAU
        assert state.best_metrics is original_best_metrics  # NOT changed
        assert state.current_template is result_b.template


# ---------------------------------------------------------------------------
# Rigorous-mode validation + template sanitization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirmatory_validation_overwrites_selected_result_with_replicated_median(
    tmp_path: Path, monkeypatch
) -> None:
    state = make_loop_state()
    config = _make_config(tmp_path, protocol="rigorous")
    ctx = RunContext(
        config=config,
        gemini_client=MagicMock(),
        reasoning_client=MagicMock(),
        registry=MagicMock(),
        gemini_semaphore=MagicMock(),
        eval_semaphore=MagicMock(),
    )

    proposal_best = _make_result(branch_id=0, agg=_make_agg(dreamsim=0.82, color=0.78))
    proposal_runner_up = _make_result(branch_id=1, agg=_make_agg(dreamsim=0.80, color=0.75))
    synth = _make_result(branch_id=2, agg=_make_agg(dreamsim=0.79, color=0.76))

    ranking = _score_and_rank([proposal_best, proposal_runner_up, synth], state)
    ranking.synth_result = synth

    synth_median = _make_agg(dreamsim=0.93, color=0.90)
    incumbent_median = _make_agg(dreamsim=0.60, color=0.50)
    median_scores = [MetricScores(dreamsim_similarity=0.93, hps_score=0.25, aesthetics_score=5.0)] * 3
    incumbent_scores = [MetricScores(dreamsim_similarity=0.60, hps_score=0.25, aesthetics_score=5.0)] * 3

    async def fake_replicate_experiment(*, branch_id, **kwargs):
        if branch_id == 2:
            return ReplicatedEvaluation(
                template=synth.template,
                branch_id=branch_id,
                replicate_scores=[median_scores] * 3,
                replicate_aggregated=[synth_median] * 3,
                median_per_image=median_scores,
                median_aggregated=synth_median,
            )
        if branch_id == 900:
            return ReplicatedEvaluation(
                template=state.best_template,
                branch_id=branch_id,
                replicate_scores=[incumbent_scores] * 3,
                replicate_aggregated=[incumbent_median] * 3,
                median_per_image=incumbent_scores,
                median_aggregated=incumbent_median,
            )
        low_agg = _make_agg(dreamsim=0.70, color=0.60)
        low_scores = [MetricScores(dreamsim_similarity=0.70, hps_score=0.25, aesthetics_score=5.0)] * 3
        return ReplicatedEvaluation(
            template=_make_valid_template(),
            branch_id=branch_id,
            replicate_scores=[low_scores] * 3,
            replicate_aggregated=[low_agg] * 3,
            median_per_image=low_scores,
            median_aggregated=low_agg,
        )

    monkeypatch.setattr("art_style_search.workflow.iteration.replicate_experiment", fake_replicate_experiment)
    monkeypatch.setattr(
        "art_style_search.workflow.iteration.paired_promotion_test",
        lambda candidate, incumbent: PromotionTestResult(
            statistic=3.0,
            p_value=0.01,
            effect_size=0.05,
            ci_lower=0.01,
            ci_upper=0.09,
            passed=True,
        ),
    )

    await _confirmatory_validation(ranking, state, ctx, iteration=3)
    _apply_iteration_result(state, ranking, config)

    assert ranking.best_exp is synth
    assert synth.aggregated is synth_median
    assert state.best_metrics is synth_median
    assert ranking.best_replicate_scores == [composite_score(synth_median)] * 3


def test_sanitize_initial_templates_replaces_invalid_with_fallback() -> None:
    invalid = make_prompt_template()
    fallback = _make_valid_template()

    sanitized = _sanitize_initial_templates([invalid, fallback], fallback=fallback)

    assert sanitized[0] is fallback
    assert sanitized[1] is fallback


@pytest.mark.asyncio
async def test_run_synthesis_experiment_skips_invalid_template(tmp_path: Path, monkeypatch) -> None:
    state = make_loop_state()
    config = _make_config(tmp_path)
    ctx = RunContext(
        config=config,
        gemini_client=MagicMock(),
        reasoning_client=MagicMock(),
        registry=MagicMock(),
        gemini_semaphore=MagicMock(),
        eval_semaphore=MagicMock(),
    )
    result = _make_result(branch_id=0, agg=_make_agg())
    ranking = IterationRanking(
        exp_results=[result],
        adaptive_scores={id(result): composite_score(result.aggregated)},
        best_exp=result,
        best_score=composite_score(result.aggregated),
        baseline_score=0.1,
        epsilon=0.01,
    )

    async def should_not_run(*args, **kwargs):
        raise AssertionError("run_experiment should not be called for invalid synthesis templates")

    monkeypatch.setattr("art_style_search.workflow.iteration.run_experiment", should_not_run)

    await _run_synthesis_experiment((make_prompt_template(), "bad synthesis"), ranking, state, ctx, iteration=2)

    assert ranking.synth_result is None
    assert len(ranking.exp_results) == 1


# ---------------------------------------------------------------------------
# TestSplitInformationBarrier
# ---------------------------------------------------------------------------


class TestSplitInformationBarrier:
    """Tests for _split_information_barrier(fixed_refs, protocol, rng)."""

    def test_rigorous_20_refs(self) -> None:
        """20 refs in rigorous mode → 14 feedback + 6 silent."""
        refs = [Path(f"/fake/{i}.png") for i in range(20)]
        rng = random.Random(42)
        feedback, silent = _split_information_barrier(refs, "rigorous", rng)

        assert len(feedback) == 14
        assert len(silent) == 6
        assert set(feedback) | set(silent) == set(refs)
        assert not (set(feedback) & set(silent))

    def test_rigorous_small_n(self) -> None:
        """6 refs in rigorous mode → 4 feedback + 2 silent (minimum 2 silent)."""
        refs = [Path(f"/fake/{i}.png") for i in range(6)]
        rng = random.Random(42)
        feedback, silent = _split_information_barrier(refs, "rigorous", rng)

        assert len(feedback) == 4
        assert len(silent) == 2
        assert set(feedback) | set(silent) == set(refs)

    def test_classic_no_split(self) -> None:
        """Classic mode → all feedback, no silent."""
        refs = [Path(f"/fake/{i}.png") for i in range(20)]
        rng = random.Random(42)
        feedback, silent = _split_information_barrier(refs, "classic", rng)

        assert len(feedback) == 20
        assert len(silent) == 0


# ---------------------------------------------------------------------------
# TestShouldHonorStop
# ---------------------------------------------------------------------------


class TestShouldHonorStop:
    """Tests for _should_honor_stop(state, ctx, reason)."""

    @staticmethod
    def _build_ctx(tmp_path: Path, *, max_iterations: int = 20, plateau_window: int = 5) -> RunContext:
        """Build a minimal RunContext with a real Config."""
        from unittest.mock import MagicMock

        config = _make_config(tmp_path, max_iterations=max_iterations, plateau_window=plateau_window)
        return RunContext(
            config=config,
            gemini_client=MagicMock(),
            reasoning_client=MagicMock(),
            registry=MagicMock(),
            gemini_semaphore=MagicMock(),
            eval_semaphore=MagicMock(),
        )

    @staticmethod
    def _populate_all_categories(kb: KnowledgeBase) -> None:
        """Add at least one hypothesis per CATEGORY_SYNONYMS key."""
        from art_style_search.utils import CATEGORY_SYNONYMS

        for cat in CATEGORY_SYNONYMS:
            if cat not in kb.categories:
                kb.categories[cat] = CategoryProgress(category=cat, hypothesis_ids=["H_dummy"])
            elif not kb.categories[cat].hypothesis_ids:
                kb.categories[cat].hypothesis_ids.append("H_dummy")

    def test_all_conditions_met(self, tmp_path: Path) -> None:
        """All 3 conditions met → True."""
        ctx = self._build_ctx(tmp_path, max_iterations=20, plateau_window=5)
        state = make_loop_state(iteration=12)  # 13 >= 20 * 0.5 = 10
        state.plateau_counter = 4  # >= max(5-1, 2) = 4
        self._populate_all_categories(state.knowledge_base)

        assert _should_honor_stop(state, ctx, "test") is True

    def test_rejects_early_iteration(self, tmp_path: Path) -> None:
        """Iteration too low → False."""
        ctx = self._build_ctx(tmp_path, max_iterations=20, plateau_window=5)
        state = make_loop_state(iteration=3)  # 4 < 20 * 0.5 = 10
        state.plateau_counter = 4
        self._populate_all_categories(state.knowledge_base)

        assert _should_honor_stop(state, ctx, "test") is False

    def test_rejects_unexplored_categories(self, tmp_path: Path) -> None:
        """Untried categories → False."""
        ctx = self._build_ctx(tmp_path, max_iterations=20, plateau_window=5)
        state = make_loop_state(iteration=12)
        state.plateau_counter = 4
        # Do NOT populate all categories — leave gaps
        state.knowledge_base = KnowledgeBase()

        assert _should_honor_stop(state, ctx, "test") is False
