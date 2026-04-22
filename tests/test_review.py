"""Unit tests for review prompt construction."""

from __future__ import annotations

from typing import Any

import pytest

from art_style_search.contracts import ExperimentProposal, Lessons
from art_style_search.prompt.review import review_iteration
from art_style_search.types import AggregatedMetrics, PromptSection, PromptTemplate, ReviewResult


def _template() -> PromptTemplate:
    return PromptTemplate(
        sections=[
            PromptSection("style_foundation", "rules", "Shared rules. " * 80),
            PromptSection("subject_anchor", "subject", "Subject rules. " * 80),
            PromptSection("color_palette", "colors", "Color rules. " * 80),
            PromptSection("composition", "layout", "Composition rules. " * 80),
            PromptSection("technique", "technique", "Technique rules. " * 80),
            PromptSection("lighting", "lighting", "Lighting rules. " * 80),
            PromptSection("environment", "environment", "Environment rules. " * 80),
            PromptSection("textures", "textures", "Texture rules. " * 80),
        ],
        caption_sections=["Art Style", "Subject", "Color Palette", "Composition", "Technique"],
        caption_length_target=500,
    )


def _metrics(seed: float) -> AggregatedMetrics:
    return AggregatedMetrics(
        dreamsim_similarity_mean=0.70 + seed,
        dreamsim_similarity_std=0.01,
        hps_score_mean=0.25 + seed / 10,
        hps_score_std=0.005,
        aesthetics_score_mean=6.0 + seed,
        aesthetics_score_std=0.2,
        color_histogram_mean=0.65 + seed,
        color_histogram_std=0.01,
        ssim_mean=0.55 + seed,
        ssim_std=0.01,
        style_consistency=0.75 + seed / 10,
        completion_rate=0.9,
        vision_style=0.5 + seed / 10,
        vision_style_std=0.01,
        vision_subject=0.6 + seed / 10,
        vision_subject_std=0.01,
        vision_composition=0.55 + seed / 10,
        vision_composition_std=0.01,
        compliance_topic_coverage=0.9,
        compliance_marker_coverage=0.9,
        section_ordering_rate=0.9,
        section_balance_rate=0.9,
        subject_specificity_rate=0.9,
        requested_ref_count=6,
        actual_ref_count=6,
    )


def _proposal() -> ExperimentProposal:
    return ExperimentProposal(
        template=_template(),
        hypothesis="Test palette precision",
        experiment_desc="Make the palette section more specific.",
        builds_on=None,
        open_problems=[],
        lessons=Lessons(),
        changed_section="color_palette",
    )


class _Result:
    def __init__(self, branch_id: int, metrics: AggregatedMetrics) -> None:
        self.branch_id = branch_id
        self.aggregated = metrics
        self.kept = branch_id == 0


class TestReviewIteration:
    @pytest.mark.asyncio
    async def test_injects_noise_floors_and_richer_deltas(self) -> None:
        captured: dict[str, Any] = {}

        class FakeClient:
            async def call_json(self, **kwargs):
                captured.update(kwargs)
                return ReviewResult(
                    experiment_assessments=["[EXP 0] SIGNAL - palette improved"],
                    noise_vs_signal="Noise floor applied.",
                    strategic_guidance="Keep palette precision.",
                    recommended_categories=["color_palette"],
                )

        await review_iteration(
            [_Result(0, _metrics(0.0)), _Result(1, _metrics(0.03))],  # type: ignore[list-item]
            [_proposal(), _proposal()],
            _metrics(0.0),
            knowledge_base=type("KB", (), {"hypotheses": [], "categories": {}, "open_problems": []})(),  # type: ignore[arg-type]
            client=FakeClient(),  # type: ignore[arg-type]
            model="fake-model",
        )

        user = captured["user"]  # type: ignore[assignment]
        system = captured["system"]  # type: ignore[assignment]
        assert "Calibration: +0.005" not in system
        assert "Noise floors for this run" in user
        assert "vision_style" in user
        assert "vision_subject" in user
        assert "vision_composition" in user
        assert "SSIM" in user
        assert "style_consistency" in user
