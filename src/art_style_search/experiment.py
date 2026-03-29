"""Single-experiment execution: caption, generate, evaluate."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from google import genai

from art_style_search.caption import caption_references
from art_style_search.config import Config
from art_style_search.evaluate import check_caption_compliance, compare_vision, evaluate_images
from art_style_search.generate import _generate_single
from art_style_search.models import ModelRegistry
from art_style_search.prompt import Lessons
from art_style_search.types import Caption, IterationResult, PromptTemplate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment proposal dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExperimentProposal:
    """Holds Claude's proposed experiment before it's executed."""

    template: PromptTemplate
    hypothesis: str
    experiment_desc: str
    builds_on: str | None
    open_problems: list[str]
    lessons: Lessons


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_experiment_results(raw: list[IterationResult | BaseException], label: str) -> list[IterationResult]:
    """Filter successful results from asyncio.gather output, logging failures."""
    results: list[IterationResult] = []
    for r in raw:
        if isinstance(r, BaseException):
            logger.error("%s failed: %s: %s", label, type(r).__name__, r, exc_info=r)
        else:
            results.append(r)
    return results


def best_kept_result(results: list[IterationResult]) -> IterationResult | None:
    """Return the kept result from a list, or the first result, or None."""
    if not results:
        return None
    return next((r for r in results if r.kept), results[0])


# ---------------------------------------------------------------------------
# Captioning + generation + evaluation
# ---------------------------------------------------------------------------


async def _caption_and_generate(
    ref_paths: list[Path],
    meta_prompt: str,
    *,
    config: Config,
    gemini_client: genai.Client,
    gemini_semaphore: asyncio.Semaphore,
    iteration: int,
    experiment_id: int,
) -> tuple[list[Caption], list[Path], list[tuple[Path, Path]]]:
    """Caption reference images with the meta-prompt, then generate images from captions.

    Returns (captions, generated_paths, pairs) where pairs maps (original, generated).
    """
    captions = await caption_references(
        ref_paths,
        model=config.caption_model,
        client=gemini_client,
        cache_dir=config.log_dir / f"iter_{iteration:03d}" / f"exp_{experiment_id}" / "captions",
        semaphore=gemini_semaphore,
        prompt=meta_prompt,
        cache_key=f"iter{iteration}_e{experiment_id}",
    )

    gen_dir = config.output_dir / f"iter_{iteration:03d}" / f"exp_{experiment_id}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    gen_tasks = [
        _generate_single(
            caption.text,
            index=i,
            aspect_ratio=config.aspect_ratio,
            output_path=gen_dir / f"{i:02d}.png",
            client=gemini_client,
            model=config.generator_model,
            semaphore=gemini_semaphore,
        )
        for i, caption in enumerate(captions)
    ]

    gen_results = await asyncio.gather(*gen_tasks, return_exceptions=True)

    generated_paths: list[Path] = []
    pairs: list[tuple[Path, Path]] = []
    for i, (caption, gen_result) in enumerate(zip(captions, gen_results, strict=True)):
        if isinstance(gen_result, BaseException):
            logger.warning("Exp %d: generation from caption %d failed: %s", experiment_id, i, gen_result)
        else:
            generated_paths.append(gen_result)
            pairs.append((caption.image_path, gen_result))

    return captions, generated_paths, pairs


async def run_experiment(
    experiment_id: int,
    template: PromptTemplate,
    iteration: int,
    fixed_refs: list[Path],
    config: Config,
    *,
    gemini_client: genai.Client,
    registry: ModelRegistry,
    gemini_semaphore: asyncio.Semaphore,
    eval_semaphore: asyncio.Semaphore,
    last_results: list[IterationResult],
    hypothesis: str = "",
    experiment_desc: str = "",
) -> IterationResult:
    """Execute one experiment: caption -> generate -> evaluate (no Claude call here)."""
    meta_prompt = template.render()
    logger.info("Exp %d iter %d — meta-prompt: %.100s...", experiment_id, iteration, meta_prompt)

    captions, generated_paths, pairs = await _caption_and_generate(
        fixed_refs,
        meta_prompt,
        config=config,
        gemini_client=gemini_client,
        gemini_semaphore=gemini_semaphore,
        iteration=iteration,
        experiment_id=experiment_id,
    )

    if not generated_paths:
        raise RuntimeError(f"Experiment {experiment_id}: no images generated")

    logger.info(
        "Exp %d iter %d — %d/%d images generated", experiment_id, iteration, len(generated_paths), len(fixed_refs)
    )

    # Evaluate — per-image paired comparison
    gen_paths_for_eval = [gen for _, gen in pairs]
    ref_paths_for_eval = [orig for orig, _ in pairs]
    caption_by_path = {c.image_path: c.text for c in captions}
    eval_captions = [caption_by_path[orig] for orig, _ in pairs]

    eval_task = evaluate_images(
        gen_paths_for_eval,
        ref_paths_for_eval,
        eval_captions,
        registry=registry,
        semaphore=eval_semaphore,
    )

    section_names = [s.name for s in template.sections]
    compliance = check_caption_compliance(section_names, captions)

    scores, aggregated = await eval_task

    # Sort by DINO worst first
    scored_items = list(zip(pairs, scores, captions, strict=False))
    scored_items.sort(key=lambda x: x[1].dino_similarity)

    sorted_pairs = [item[0] for item in scored_items]
    sorted_captions = [item[2] for item in scored_items]

    # Vision comparison with per-pair captions — returns text + structured scores
    sorted_caption_texts = [c.text for c in sorted_captions]
    vision_feedback, vision_scores = await compare_vision(
        sorted_pairs,
        sorted_caption_texts,
        meta_prompt,
        client=gemini_client,
        model=config.caption_model,
        semaphore=gemini_semaphore,
        max_pairs=5,
    )

    # Merge vision scores into aggregated metrics
    aggregated = replace(
        aggregated,
        vision_style=vision_scores.style.score,
        vision_subject=vision_scores.subject.score,
        vision_composition=vision_scores.composition.score,
    )

    # Build roundtrip feedback — full caption for worst image, truncated for rest
    roundtrip_details: list[str] = []
    prev = best_kept_result(last_results)
    prev_scores: dict[Path, float] = {}
    if prev:
        for cap, sc in zip(prev.iteration_captions, prev.per_image_scores, strict=False):
            prev_scores[cap.image_path] = sc.dino_similarity

    for idx, ((_orig, _gen), sc, cap) in enumerate(scored_items):
        prev_dino = prev_scores.get(cap.image_path)
        trend = ""
        if prev_dino is not None:
            arrow = "↑" if sc.dino_similarity > prev_dino else "↓" if sc.dino_similarity < prev_dino else "="
            trend = f" [prev DINO={prev_dino:.3f} → {sc.dino_similarity:.3f} {arrow}]"
        caption_text = cap.text if idx < 3 else f"{cap.text[:300]}..."
        roundtrip_details.append(
            f"Image ({_orig.name}): DINO={sc.dino_similarity:.3f} LPIPS={sc.lpips_distance:.3f} "
            f"Color={sc.color_histogram:.3f} Tex={sc.texture:.3f} SSIM={sc.ssim:.3f} "
            f"HPS={sc.hps_score:.3f} Aes={sc.aesthetics_score:.1f}{trend}\n"
            f"  Caption: {caption_text}"
        )
    roundtrip_feedback = "\n".join(roundtrip_details)
    if compliance:
        roundtrip_feedback = compliance + "\n\n" + roundtrip_feedback

    return IterationResult(
        branch_id=experiment_id,
        iteration=iteration,
        template=template,
        rendered_prompt=meta_prompt,
        image_paths=generated_paths,
        per_image_scores=scores,
        aggregated=aggregated,
        claude_analysis="",
        template_changes="",
        kept=False,
        hypothesis=hypothesis,
        experiment=experiment_desc,
        vision_feedback=vision_feedback,
        roundtrip_feedback=roundtrip_feedback,
        iteration_captions=captions,
    )
