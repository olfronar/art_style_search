"""Unit tests for art_style_search.experiment — helper functions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from art_style_search.experiment import (
    _caption_and_generate,
    _median_metric_scores,
    best_kept_result,
    collect_experiment_results,
    replicate_experiment,
)
from art_style_search.types import Caption
from tests.conftest import make_iteration_result, make_metric_scores, make_prompt_template

# ---------------------------------------------------------------------------
# collect_experiment_results
# ---------------------------------------------------------------------------


class TestCollectExperimentResults:
    def test_filters_exceptions(self) -> None:
        good = make_iteration_result(branch_id=0, iteration=1)
        bad = RuntimeError("boom")
        results = collect_experiment_results([good, bad, good], label="test")
        assert len(results) == 2
        assert all(r is good for r in results)

    def test_keeps_all_successes(self) -> None:
        items = [make_iteration_result(branch_id=i, iteration=1) for i in range(3)]
        results = collect_experiment_results(items, label="test")
        assert results == items

    def test_empty_input(self) -> None:
        assert collect_experiment_results([], label="test") == []


# ---------------------------------------------------------------------------
# best_kept_result
# ---------------------------------------------------------------------------


class TestBestKeptResult:
    def test_prefers_kept(self) -> None:
        r0 = make_iteration_result(branch_id=0, iteration=1)
        r0 = r0.__class__(**{**r0.__dict__, "kept": False})
        r1 = make_iteration_result(branch_id=1, iteration=1)
        r1 = r1.__class__(**{**r1.__dict__, "kept": True})
        r2 = make_iteration_result(branch_id=2, iteration=1)
        r2 = r2.__class__(**{**r2.__dict__, "kept": False})

        result = best_kept_result([r0, r1, r2])
        assert result is not None
        assert result.kept is True
        assert result.branch_id == 1

    def test_falls_back_to_first(self) -> None:
        r0 = make_iteration_result(branch_id=0, iteration=1)
        r0 = r0.__class__(**{**r0.__dict__, "kept": False})
        r1 = make_iteration_result(branch_id=1, iteration=1)
        r1 = r1.__class__(**{**r1.__dict__, "kept": False})

        result = best_kept_result([r0, r1])
        assert result is not None
        assert result.branch_id == 0

    def test_empty_returns_none(self) -> None:
        assert best_kept_result([]) is None


# ---------------------------------------------------------------------------
# _median_metric_scores
# ---------------------------------------------------------------------------


class TestMedianMetricScores:
    def test_median_of_three_replicates(self) -> None:
        # 3 replicates, 2 images each — seeds chosen so the median is the middle value
        rep0 = [make_metric_scores(seed=0.0), make_metric_scores(seed=3.0)]
        rep1 = [make_metric_scores(seed=1.0), make_metric_scores(seed=5.0)]
        rep2 = [make_metric_scores(seed=2.0), make_metric_scores(seed=4.0)]

        medians = _median_metric_scores([rep0, rep1, rep2])
        assert len(medians) == 2

        # Image 0: seeds 0, 1, 2 — median is seed=1 values
        expected_img0 = make_metric_scores(seed=1.0)
        assert medians[0].dreamsim_similarity == expected_img0.dreamsim_similarity
        assert medians[0].hps_score == expected_img0.hps_score
        assert medians[0].aesthetics_score == expected_img0.aesthetics_score

        # Image 1: seeds 3, 5, 4 — median is seed=4 values
        expected_img1 = make_metric_scores(seed=4.0)
        assert medians[1].dreamsim_similarity == expected_img1.dreamsim_similarity
        assert medians[1].hps_score == expected_img1.hps_score
        assert medians[1].aesthetics_score == expected_img1.aesthetics_score

    def test_single_replicate_passthrough(self) -> None:
        scores = [make_metric_scores(seed=2.0), make_metric_scores(seed=7.0)]
        medians = _median_metric_scores([scores])
        assert len(medians) == 2

        for original, median in zip(scores, medians, strict=True):
            assert median.dreamsim_similarity == original.dreamsim_similarity
            assert median.hps_score == original.hps_score
            assert median.aesthetics_score == original.aesthetics_score
            assert median.color_histogram == original.color_histogram
            assert median.ssim == original.ssim
            assert median.vision_style == original.vision_style
            assert median.vision_subject == original.vision_subject
            assert median.vision_composition == original.vision_composition


class TestReplicateExperiment:
    @pytest.mark.asyncio
    async def test_seeded_existing_result_preserves_style_consistency_when_no_new_replicates(self, tmp_path) -> None:
        from art_style_search.config import Config

        config = Config(
            reference_dir=tmp_path / "refs",
            output_dir=tmp_path / "outputs",
            log_dir=tmp_path / "logs",
            state_file=tmp_path / "state.json",
            run_dir=tmp_path,
            max_iterations=1,
            plateau_window=5,
            num_branches=1,
            aspect_ratio="1:1",
            num_fixed_refs=1,
            caption_model="caption-model",
            generator_model="generator-model",
            reasoning_model="reasoning-model",
            reasoning_provider="anthropic",
            reasoning_base_url="",
            gemini_concurrency=1,
            eval_concurrency=1,
            seed=42,
            protocol="rigorous",
            anthropic_api_key="test",
            google_api_key="test",
            zai_api_key="",
            openai_api_key="",
        )
        existing_result = make_iteration_result(branch_id=7, iteration=3)
        existing_result.aggregated = replace(existing_result.aggregated, style_consistency=0.42)

        replicated = await replicate_experiment(
            template=existing_result.template,
            branch_id=existing_result.branch_id,
            iteration=existing_result.iteration,
            fixed_refs=[],
            config=config,
            services=object(),
            n_replicates=1,
            existing_result=existing_result,
        )

        assert replicated.median_aggregated.style_consistency == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_generates_fresh_replicate_forwards_negative_prompt(self, tmp_path) -> None:
        """Without existing_result, _run_one_replicate must run and forward the template's negative_prompt."""
        from art_style_search.config import Config
        from art_style_search.types import MetricScores, VisionDimensionScore, VisionScores

        ref_path = tmp_path / "ref.png"
        ref_path.touch()

        captured_negative: dict[str, str | None] = {}

        class FakeCaptioning:
            async def caption_single(self, image_path, *, prompt, cache_dir, cache_key=""):
                return Caption(
                    image_path=image_path,
                    text=(
                        "[Art Style] Watercolor with soft edges.\n"
                        "[Subject] A red fox with white socks and alert ears.\n"
                        "[Color Palette] Ochre, moss green."
                    ),
                )

        class FakeGeneration:
            async def generate_single(self, prompt, *, index, output_path, negative_prompt=None):
                captured_negative["negative_prompt"] = negative_prompt
                output_path.touch()
                return output_path

        class FakeEvaluation:
            async def evaluate_images(self, gen_paths, ref_paths, captions):
                scores = [
                    MetricScores(dreamsim_similarity=0.7, hps_score=0.25, aesthetics_score=5.5) for _ in gen_paths
                ]
                return scores, 0

            async def compare_vision_per_image(self, pairs, captions):
                vs = [
                    VisionScores(
                        style=VisionDimensionScore("style", 1.0, ""),
                        subject=VisionDimensionScore("subject", 1.0, ""),
                        composition=VisionDimensionScore("composition", 1.0, ""),
                        medium=VisionDimensionScore("medium", 1.0, ""),
                        proportions=VisionDimensionScore("proportions", 1.0, ""),
                    )
                    for _ in pairs
                ]
                return "", vs

        services = type(
            "FakeServices",
            (),
            {"captioning": FakeCaptioning(), "generation": FakeGeneration(), "evaluation": FakeEvaluation()},
        )()

        template = make_prompt_template()
        template = replace(template, negative_prompt="photorealism, harsh shadows")
        config = Config(
            reference_dir=tmp_path / "refs",
            output_dir=tmp_path / "outputs",
            log_dir=tmp_path / "logs",
            state_file=tmp_path / "state.json",
            run_dir=tmp_path,
            max_iterations=1,
            plateau_window=5,
            num_branches=1,
            aspect_ratio="1:1",
            num_fixed_refs=1,
            caption_model="caption-model",
            generator_model="generator-model",
            reasoning_model="reasoning-model",
            reasoning_provider="anthropic",
            reasoning_base_url="",
            gemini_concurrency=1,
            eval_concurrency=1,
            seed=42,
            protocol="classic",
            anthropic_api_key="test",
            google_api_key="test",
            zai_api_key="",
            openai_api_key="",
        )

        replicated = await replicate_experiment(
            template=template,
            branch_id=0,
            iteration=1,
            fixed_refs=[ref_path],
            config=config,
            services=services,
            n_replicates=1,
        )

        assert captured_negative["negative_prompt"] == "photorealism, harsh shadows"
        assert len(replicated.replicate_scores) == 1


class TestCaptionAndGenerate:
    @pytest.mark.asyncio
    async def test_generation_uses_subject_first_prompt(self, tmp_path) -> None:
        from art_style_search.config import Config

        ref_path = tmp_path / "ref.png"
        ref_path.touch()
        captured: dict[str, str] = {}
        captured_negative: dict[str, str | None] = {}

        class FakeCaptioning:
            async def caption_single(self, image_path, *, prompt, cache_dir, cache_key=""):
                return Caption(
                    image_path=image_path,
                    text=(
                        "[Art Style] Shared watercolor rules with soft edges.\n"
                        "[Subject] A red fox with white socks, a patched satchel, raised ears, "
                        "and a mid-step pose in marsh grass.\n"
                        "[Color Palette] Ochre, moss green, slate blue."
                    ),
                )

        class FakeGeneration:
            async def generate_single(self, prompt, *, index, output_path, negative_prompt=None):
                captured["prompt"] = prompt
                captured_negative["negative_prompt"] = negative_prompt
                output_path.touch()
                return output_path

        config = Config(
            reference_dir=tmp_path / "refs",
            output_dir=tmp_path / "outputs",
            log_dir=tmp_path / "logs",
            state_file=tmp_path / "state.json",
            run_dir=tmp_path,
            max_iterations=1,
            plateau_window=5,
            num_branches=1,
            aspect_ratio="1:1",
            num_fixed_refs=1,
            caption_model="caption-model",
            generator_model="generator-model",
            reasoning_model="reasoning-model",
            reasoning_provider="anthropic",
            reasoning_base_url="",
            gemini_concurrency=1,
            eval_concurrency=1,
            seed=42,
            protocol="classic",
            anthropic_api_key="test",
            google_api_key="test",
            zai_api_key="",
            openai_api_key="",
        )

        services = type(
            "FakeServices",
            (),
            {"captioning": FakeCaptioning(), "generation": FakeGeneration()},
        )()

        captions, generated, pairs = await _caption_and_generate(
            [ref_path],
            "meta prompt",
            negative_prompt="Avoid watermarks and signatures.",
            style_canon="",
            config=config,
            services=services,
            iteration=1,
            experiment_id=0,
        )

        assert captions[0].text.startswith("[Art Style]")
        assert generated[0].name == "00.png"
        assert pairs[0] == (ref_path, generated[0])
        assert captured["prompt"].startswith("[Subject]\nA red fox")
        assert "Render in this style:\n[Art Style]\nShared watercolor rules with soft edges." in captured["prompt"]
        assert captured_negative["negative_prompt"] == "Avoid watermarks and signatures."
