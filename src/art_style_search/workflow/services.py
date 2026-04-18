"""Internal service adapters for provider-facing workflow operations."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from google import genai  # type: ignore[attr-defined]

from art_style_search.caption import caption_references, caption_single
from art_style_search.evaluate import compare_vision_per_image, evaluate_images, pairwise_compare_experiments
from art_style_search.generate import generate_single
from art_style_search.models import ModelRegistry
from art_style_search.reasoning_client import ReasoningClient
from art_style_search.types import Caption, MetricScores, VisionScores

T = TypeVar("T")


@dataclass(frozen=True)
class CaptioningService:
    """Captions reference images through the configured provider."""

    client: genai.Client
    model: str
    semaphore: asyncio.Semaphore
    thinking_level: str = "MINIMAL"

    async def caption_single(
        self,
        image_path: Path,
        *,
        prompt: str,
        cache_dir: Path | None,
        cache_key: str = "",
    ) -> Caption:
        return await caption_single(
            image_path,
            prompt=prompt,
            model=self.model,
            client=self.client,
            cache_dir=cache_dir,
            semaphore=self.semaphore,
            cache_key=cache_key,
            thinking_level=self.thinking_level,
        )

    async def caption_references(
        self,
        reference_paths: list[Path],
        *,
        cache_dir: Path,
        prompt: str | None = None,
        cache_key: str = "",
    ) -> list[Caption]:
        return await caption_references(
            reference_paths,
            model=self.model,
            client=self.client,
            cache_dir=cache_dir,
            semaphore=self.semaphore,
            prompt=prompt,
            cache_key=cache_key,
            thinking_level=self.thinking_level,
        )


@dataclass(frozen=True)
class GenerationService:
    """Generates images from captions through the configured provider."""

    client: genai.Client
    model: str
    semaphore: asyncio.Semaphore
    aspect_ratio: str
    thinking_level: str = "MINIMAL"

    async def generate_single(
        self,
        prompt: str,
        *,
        index: int,
        output_path: Path,
        negative_prompt: str | None = None,
        style_invariants: str = "",
    ) -> Path:
        return await generate_single(
            prompt,
            index=index,
            aspect_ratio=self.aspect_ratio,
            output_path=output_path,
            client=self.client,
            model=self.model,
            semaphore=self.semaphore,
            negative_prompt=negative_prompt,
            style_invariants=style_invariants,
            thinking_level=self.thinking_level,
        )


@dataclass(frozen=True)
class EvaluationService:
    """Runs local metrics and Gemini-based vision comparisons."""

    gemini_client: genai.Client
    registry: ModelRegistry
    comparison_provider: str
    comparison_model: str
    gemini_semaphore: asyncio.Semaphore
    eval_semaphore: asyncio.Semaphore
    xai_client: object | None = None

    async def evaluate_images(
        self,
        generated_paths: list[Path],
        reference_paths: list[Path],
        captions: list[str],
    ) -> tuple[list[MetricScores], int]:
        return await evaluate_images(
            generated_paths,
            reference_paths,
            captions,
            registry=self.registry,
            semaphore=self.eval_semaphore,
        )

    async def compare_vision_per_image(
        self,
        pairs: list[tuple[Path, Path]],
        captions: list[str],
    ) -> tuple[list[str], list[VisionScores]]:
        return await compare_vision_per_image(
            pairs,
            captions,
            provider=self.comparison_provider,
            client=self.gemini_client,
            xai_client=self.xai_client,
            model=self.comparison_model,
            semaphore=self.gemini_semaphore,
        )

    async def pairwise_compare(
        self,
        pairs_a: list[tuple[Path, Path]],
        pairs_b: list[tuple[Path, Path]],
        *,
        max_images: int = 3,
    ) -> tuple[str, float]:
        return await pairwise_compare_experiments(
            pairs_a,
            pairs_b,
            provider=self.comparison_provider,
            client=self.gemini_client,
            xai_client=self.xai_client,
            model=self.comparison_model,
            semaphore=self.gemini_semaphore,
            max_images=max_images,
        )


@dataclass(frozen=True)
class ReasoningService:
    """Runs text or JSON reasoning requests against the configured provider."""

    client: ReasoningClient
    model: str
    effort: str = "medium"

    async def call_text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 16000,
        stage: str = "unknown",
    ) -> str:
        return await self.client.call(model=self.model, system=system, user=user, max_tokens=max_tokens, stage=stage)

    async def call_json(
        self,
        *,
        system: str,
        user: str,
        validator,
        response_name: str,
        schema_hint: str = "",
        response_schema: dict[str, object] | None = None,
        max_tokens: int = 16000,
        repair_retries: int = 1,
        final_failure_log_level: int = logging.WARNING,
        stage: str = "unknown",
    ):
        return await self.client.call_json(
            model=self.model,
            system=system,
            user=user,
            validator=validator,
            response_name=response_name,
            schema_hint=schema_hint,
            response_schema=response_schema,
            max_tokens=max_tokens,
            repair_retries=repair_retries,
            final_failure_log_level=final_failure_log_level,
            stage=stage,
        )


@dataclass(frozen=True)
class RunServices:
    """Container grouping all workflow-facing service adapters."""

    captioning: CaptioningService
    generation: GenerationService
    evaluation: EvaluationService
    reasoning: ReasoningService
