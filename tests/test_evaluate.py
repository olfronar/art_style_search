"""Unit tests for art_style_search.evaluate.aggregate."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from art_style_search.evaluate import aggregate, compare_vision_per_image, pairwise_compare_experiments
from art_style_search.types import MetricScores


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (16, 16), color=color).save(path)


class TestAggregateEmpty:
    """aggregate with an empty list should return all-zero metrics."""

    def test_returns_zeros(self) -> None:
        result = aggregate([])
        assert result.dreamsim_similarity_mean == 0.0
        assert result.dreamsim_similarity_std == 0.0
        assert result.hps_score_mean == 0.0
        assert result.hps_score_std == 0.0
        assert result.aesthetics_score_mean == 0.0
        assert result.aesthetics_score_std == 0.0


class TestAggregateSingle:
    """aggregate with a single score should return that score as mean, std=0."""

    def test_single_score(self) -> None:
        scores = [MetricScores(dreamsim_similarity=0.8, hps_score=0.25, aesthetics_score=6.5)]
        result = aggregate(scores)

        assert result.dreamsim_similarity_mean == 0.8
        assert result.dreamsim_similarity_std == 0.0
        assert result.hps_score_mean == 0.25
        assert result.hps_score_std == 0.0
        assert result.aesthetics_score_mean == 6.5
        assert result.aesthetics_score_std == 0.0


class TestAggregateMultiple:
    """aggregate with multiple scores should compute correct mean and population std."""

    def test_mean_and_std(self) -> None:
        scores = [
            MetricScores(dreamsim_similarity=0.6, hps_score=0.20, aesthetics_score=5.0),
            MetricScores(dreamsim_similarity=0.8, hps_score=0.30, aesthetics_score=7.0),
            MetricScores(dreamsim_similarity=1.0, hps_score=0.10, aesthetics_score=9.0),
        ]
        result = aggregate(scores)

        # Expected means
        assert math.isclose(result.dreamsim_similarity_mean, 0.8, abs_tol=1e-9)
        assert math.isclose(result.hps_score_mean, 0.2, abs_tol=1e-9)
        assert math.isclose(result.aesthetics_score_mean, 7.0, abs_tol=1e-9)

        # Expected population std: sqrt(mean((x - mean)^2))
        # dreamsim: values [0.6, 0.8, 1.0], mean=0.8, deviations [-0.2, 0, 0.2]
        expected_ds_std = ((0.2**2 + 0.0**2 + 0.2**2) / 3) ** 0.5
        assert math.isclose(result.dreamsim_similarity_std, expected_ds_std, abs_tol=1e-9)

        # hps: values [0.20, 0.30, 0.10], mean=0.2, deviations [0, 0.1, -0.1]
        expected_hps_std = ((0.0**2 + 0.1**2 + 0.1**2) / 3) ** 0.5
        assert math.isclose(result.hps_score_std, expected_hps_std, abs_tol=1e-9)

        # aesthetics: values [5.0, 7.0, 9.0], mean=7.0, deviations [-2, 0, 2]
        expected_aes_std = ((4.0 + 0.0 + 4.0) / 3) ** 0.5
        assert math.isclose(result.aesthetics_score_std, expected_aes_std, abs_tol=1e-9)


class TestXAIComparison:
    @pytest.mark.asyncio
    async def test_compare_vision_per_image_uses_xai_multimodal_payload(self, tmp_path: Path) -> None:
        ref_path = tmp_path / "ref.png"
        gen_path = tmp_path / "gen.png"
        _write_image(ref_path, (10, 20, 30))
        _write_image(gen_path, (30, 20, 10))

        captured: dict[str, object] = {}
        response_text = (
            '<style verdict="MATCH">Style matches closely.</style>\n'
            '<subject verdict="PARTIAL">Subject differs slightly.</subject>\n'
            '<composition verdict="MISS">Composition is off.</composition>\n'
            "<key_gap>Lighting drift</key_gap>"
        )

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text=response_text)

        xai_client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

        long_caption = "caption start " + ("detail " * 200) + "TAIL_MARKER"

        feedbacks, scores = await compare_vision_per_image(
            [(ref_path, gen_path)],
            [long_caption],
            provider="xai",
            client=None,
            xai_client=xai_client,
            model="grok-4.20-reasoning-latest",
            semaphore=asyncio.Semaphore(1),
        )

        assert feedbacks == [response_text]
        assert scores[0].style.score == 1.0
        assert scores[0].subject.score == 0.5
        assert scores[0].composition.score == 0.0
        assert captured["model"] == "grok-4.20-reasoning-latest"
        assert captured["store"] is False
        assert captured["input"][0]["role"] == "system"  # type: ignore[index]
        content = captured["input"][1]["content"]  # type: ignore[index]
        assert content[0]["type"] == "input_text"
        assert content[1]["type"] == "input_image"
        assert content[2]["type"] == "input_text"
        assert content[3]["type"] == "input_image"
        assert "TAIL_MARKER" in content[-1]["text"]

    @pytest.mark.asyncio
    async def test_compare_vision_per_image_can_swap_generated_before_original(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        ref_path = tmp_path / "ref.png"
        gen_path = tmp_path / "gen.png"
        _write_image(ref_path, (10, 20, 30))
        _write_image(gen_path, (30, 20, 10))

        monkeypatch.setattr("art_style_search.evaluate.random.random", lambda: 0.1)

        captured: dict[str, object] = {}
        response_text = (
            '<style verdict="MATCH">Style matches closely.</style>\n'
            '<subject verdict="PARTIAL">Subject differs slightly.</subject>\n'
            '<composition verdict="MISS">Composition is off.</composition>'
        )

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text=response_text)

        xai_client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

        await compare_vision_per_image(
            [(ref_path, gen_path)],
            ["caption text"],
            provider="xai",
            client=None,
            xai_client=xai_client,
            model="grok-4.20-reasoning-latest",
            semaphore=asyncio.Semaphore(1),
        )

        content = captured["input"][1]["content"]  # type: ignore[index]
        assert content[0]["text"].endswith("GENERATED reproduction:")  # type: ignore[index]
        assert content[2]["text"].endswith("ORIGINAL reference:")  # type: ignore[index]

    @pytest.mark.asyncio
    async def test_pairwise_compare_experiments_uses_xai_and_parses_winner(self, tmp_path: Path, monkeypatch) -> None:
        ref_path = tmp_path / "ref.png"
        gen_a = tmp_path / "a.png"
        gen_b = tmp_path / "b.png"
        _write_image(ref_path, (10, 20, 30))
        _write_image(gen_a, (30, 40, 50))
        _write_image(gen_b, (50, 40, 30))

        monkeypatch.setattr("art_style_search.evaluate.random.random", lambda: 0.9)

        captured: dict[str, object] = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text="<winner>A</winner><rationale>Set A is closer overall.</rationale>")

        xai_client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

        rationale, score = await pairwise_compare_experiments(
            [(ref_path, gen_a)],
            [(ref_path, gen_b)],
            provider="xai",
            client=None,
            xai_client=xai_client,
            model="grok-4.20-reasoning-latest",
            semaphore=asyncio.Semaphore(1),
            max_images=1,
        )

        assert rationale == "Set A is closer overall."
        assert score == 1.0
        assert captured["store"] is False
        assert captured["input"][0]["role"] == "system"  # type: ignore[index]
