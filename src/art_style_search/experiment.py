"""Single-experiment execution: caption, generate, evaluate."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from google import genai

from art_style_search.caption import caption_single
from art_style_search.config import Config
from art_style_search.evaluate import (
    aggregate,
    check_caption_compliance,
    compare_vision_per_image,
    compute_style_consistency,
    evaluate_images,
)
from art_style_search.generate import generate_single
from art_style_search.models import ModelRegistry
from art_style_search.prompt import Lessons
from art_style_search.types import Caption, IterationResult, MetricScores, PromptTemplate, verdict_label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment proposal dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExperimentProposal:
    """Holds the reasoning model's proposed experiment before it's executed."""

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
    """Caption and generate per-image in a pipeline (no serial boundary).

    Each image's caption→generate runs as a single chained task so generation
    starts as soon as each individual caption completes. Returns (captions,
    generated_paths, pairs) where pairs maps (original, generated).
    """
    cache_dir = config.log_dir / f"iter_{iteration:03d}" / f"exp_{experiment_id}" / "captions"
    gen_dir = config.output_dir / f"iter_{iteration:03d}" / f"exp_{experiment_id}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"iter{iteration}_e{experiment_id}"

    async def _caption_then_generate(ref_path: Path, i: int) -> tuple[Caption, Path]:
        caption = await caption_single(
            ref_path,
            prompt=meta_prompt,
            model=config.caption_model,
            client=gemini_client,
            cache_dir=cache_dir,
            semaphore=gemini_semaphore,
            cache_key=cache_key,
        )
        gen_path = await generate_single(
            caption.text,
            index=i,
            aspect_ratio=config.aspect_ratio,
            output_path=gen_dir / f"{i:02d}.png",
            client=gemini_client,
            model=config.generator_model,
            semaphore=gemini_semaphore,
        )
        return caption, gen_path

    results = await asyncio.gather(
        *[_caption_then_generate(p, i) for i, p in enumerate(ref_paths)],
        return_exceptions=True,
    )

    captions: list[Caption] = []
    generated_paths: list[Path] = []
    pairs: list[tuple[Path, Path]] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning("Exp %d: image %d (%s) failed: %s", experiment_id, i, ref_paths[i].name, result)
        else:
            caption, gen_path = result
            captions.append(caption)
            generated_paths.append(gen_path)
            pairs.append((caption.image_path, gen_path))

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
    """Execute one experiment: caption -> generate -> evaluate (no reasoning-model call here)."""
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

    gen_paths_for_eval = [gen for _, gen in pairs]
    ref_paths_for_eval = [orig for orig, _ in pairs]
    caption_by_path = {c.image_path: c.text for c in captions}
    eval_captions = [caption_by_path[orig] for orig, _ in pairs]

    # Run metric evaluation and vision comparison in parallel
    (metric_scores, _), (vision_feedbacks, vision_scores_list) = await asyncio.gather(
        evaluate_images(
            gen_paths_for_eval, ref_paths_for_eval, eval_captions, registry=registry, semaphore=eval_semaphore
        ),
        compare_vision_per_image(
            pairs, eval_captions, client=gemini_client, model=config.caption_model, semaphore=gemini_semaphore
        ),
    )

    section_names = [s.name for s in template.sections]
    compliance = check_caption_compliance(section_names, captions, caption_sections=template.caption_sections)

    # Sort by DreamSim worst-first, merge vision scores into MetricScores, then aggregate
    order = sorted(range(len(metric_scores)), key=lambda i: metric_scores[i].dreamsim_similarity)
    scores: list[MetricScores] = []
    vision_parts: list[str] = []
    for i in order:
        sc, vs, fb = metric_scores[i], vision_scores_list[i], vision_feedbacks[i]
        scores.append(
            replace(
                sc,
                vision_style=vs.style.score,
                vision_subject=vs.subject.score,
                vision_composition=vs.composition.score,
            )
        )
        ref_path = pairs[i][0]
        vl = f"S={verdict_label(vs.style.score)} Su={verdict_label(vs.subject.score)} Co={verdict_label(vs.composition.score)}"
        vision_parts.append(f"**{ref_path.name}** [{vl}]: {fb[:300]}")
    aggregated = aggregate(scores)
    # Measure how consistent the [Art Style] blocks are across captions
    style_con = compute_style_consistency(captions)
    aggregated = replace(aggregated, style_consistency=style_con)
    vision_feedback = "\n".join(vision_parts)

    sorted_pairs = [pairs[i] for i in order]
    sorted_captions_list = [captions[order[j]] for j in range(len(order))]

    # Build roundtrip feedback — full caption for worst images, truncated for rest
    prev = best_kept_result(last_results)
    prev_scores: dict[Path, float] = {}
    if prev:
        for cap, sc in zip(prev.iteration_captions, prev.per_image_scores, strict=False):
            prev_scores[cap.image_path] = sc.dreamsim_similarity

    roundtrip_details: list[str] = []
    for idx, ((ref_p, _), sc, cap) in enumerate(zip(sorted_pairs, scores, sorted_captions_list, strict=True)):
        prev_ds = prev_scores.get(cap.image_path)
        trend = ""
        if prev_ds is not None:
            arrow = "↑" if sc.dreamsim_similarity > prev_ds else "↓" if sc.dreamsim_similarity < prev_ds else "="
            trend = f" [prev DS={prev_ds:.3f} → {sc.dreamsim_similarity:.3f} {arrow}]"
        vl = f"V[S={verdict_label(sc.vision_style)} Su={verdict_label(sc.vision_subject)} Co={verdict_label(sc.vision_composition)}]"
        caption_text = cap.text if idx < 3 else f"{cap.text[:300]}..."
        roundtrip_details.append(
            f"Image ({ref_p.name}): DS={sc.dreamsim_similarity:.3f} "
            f"Color={sc.color_histogram:.3f} SSIM={sc.ssim:.3f} "
            f"HPS={sc.hps_score:.3f} Aes={sc.aesthetics_score:.1f} {vl}{trend}\n"
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
