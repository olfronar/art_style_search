"""Dispatch metric computations via asyncio.to_thread through ModelRegistry."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from PIL import Image

from art_style_search.models import ModelRegistry
from art_style_search.types import AggregatedMetrics, MetricScores

logger = logging.getLogger(__name__)


def _aggregate(scores: list[MetricScores]) -> AggregatedMetrics:
    """Compute mean and std for each metric across a list of per-image scores."""
    n = len(scores)
    if n == 0:
        return AggregatedMetrics(
            dino_similarity_mean=0.0,
            dino_similarity_std=0.0,
            lpips_distance_mean=0.0,
            lpips_distance_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
        )

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    def _std(vals: list[float]) -> float:
        m = _mean(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    dino = [s.dino_similarity for s in scores]
    lpips_vals = [s.lpips_distance for s in scores]
    hps = [s.hps_score for s in scores]
    aes = [s.aesthetics_score for s in scores]

    return AggregatedMetrics(
        dino_similarity_mean=_mean(dino),
        dino_similarity_std=_std(dino),
        lpips_distance_mean=_mean(lpips_vals),
        lpips_distance_std=_std(lpips_vals),
        hps_score_mean=_mean(hps),
        hps_score_std=_std(hps),
        aesthetics_score_mean=_mean(aes),
        aesthetics_score_std=_std(aes),
    )


def _compute_single_sync(
    registry: ModelRegistry,
    generated: Image.Image,
    references: list[Image.Image],
    prompt: str,
) -> MetricScores:
    """Compute all 4 metrics for one generated image (synchronous)."""
    return MetricScores(
        dino_similarity=registry.compute_dino(generated, references),
        lpips_distance=registry.compute_lpips(generated, references),
        hps_score=registry.compute_hps(generated, prompt),
        aesthetics_score=registry.compute_aesthetics(generated),
    )


async def evaluate_images(
    generated_paths: list[Path],
    reference_paths: list[Path],
    prompt: str,
    *,
    registry: ModelRegistry,
    semaphore: asyncio.Semaphore,
) -> tuple[list[MetricScores], AggregatedMetrics]:
    """Evaluate all generated images against references.

    Each image's 4 metrics are computed in a thread via ``asyncio.to_thread``,
    throttled by *semaphore* to prevent GPU/CPU oversubscription.

    Returns per-image scores and aggregated statistics.
    """
    ref_images = [Image.open(p).convert("RGB") for p in reference_paths]

    async def _eval_one(gen_path: Path) -> MetricScores | None:
        async with semaphore:
            gen_image: Image.Image | None = None
            try:
                gen_image = Image.open(gen_path).convert("RGB")
                scores = await asyncio.to_thread(_compute_single_sync, registry, gen_image, ref_images, prompt)
                return scores
            except Exception as exc:
                logger.warning("Evaluation failed for %s: %s", gen_path.name, exc)
                return None
            finally:
                if gen_image is not None:
                    gen_image.close()

    results = await asyncio.gather(*[_eval_one(p) for p in generated_paths])

    scores = [r for r in results if r is not None]
    aggregated = _aggregate(scores)

    for img in ref_images:
        img.close()

    return scores, aggregated
