"""Dispatch metric computations via asyncio.to_thread through ModelRegistry."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from pathlib import Path

from google import genai
from PIL import Image

from art_style_search.models import ModelRegistry
from art_style_search.types import AggregatedMetrics, MetricScores, VisionDimensionScore, VisionScores
from art_style_search.utils import image_to_gemini_part

logger = logging.getLogger(__name__)

_VISION_COMPARE_PROMPT = (
    "You are an expert art analyst. You are shown reference images and generated images "
    "(attempts to reproduce each reference from a text caption).\n\n"
    "The META-PROMPT (captioner instruction) was:\n{rendered_prompt}\n\n"
    "Each pair shows the caption used for generation. Be BRUTALLY HONEST — do not be generous. "
    "Focus on what is WRONG, not what is right.\n\n"
    "Respond in this structured format:\n\n"
    "<matched>Aspects captured correctly (be brief)</matched>\n"
    "<gaps>\n"
    '<gap priority="high">Most critical difference — be specific about WHAT pixels/regions are wrong</gap>\n'
    '<gap priority="high">Second most critical difference</gap>\n'
    '<gap priority="medium">Third difference</gap>\n'
    "</gaps>\n"
    "<characters>For EACH pair that contains characters/figures: compare the reference character "
    "against the generated one. List SPECIFIC differences in: face shape, eye size/position, "
    "nose shape, mouth, skin tone, hair style/color, body proportions, clothing details, pose accuracy. "
    "If a character is unrecognizable compared to the reference, say so explicitly.</characters>\n"
    "<caption_diagnosis>For each pair: is the caption missing critical information, or is the "
    "generator failing to follow the caption? Quote the specific caption phrases that were "
    "ignored or misinterpreted by the generator.</caption_diagnosis>\n"
    "<prompt_issues>What specific meta-prompt wording changes would fix the character and "
    "subject fidelity gaps above</prompt_issues>\n\n"
    "Rate reproduction quality per dimension using STRICT criteria "
    "(5=mediocre match with obvious differences, 7=good but noticeable gaps, 9+=near-identical):\n"
    "<dimensions>\n"
    '  <style score="N">Art technique reproduction — be strict</style>\n'
    '  <subject score="N">Character/subject identity and proportions — be very strict, '
    "any wrong proportions or missing features = low score</subject>\n"
    '  <composition score="N">Spatial layout, object positions, framing — be strict</composition>\n'
    "</dimensions>"
)

_DIMENSION_RE = re.compile(
    r'<(\w+)\s+score="(\d+(?:\.\d+)?)">(.*?)</\1>',
    re.DOTALL,
)


def _parse_vision_scores(text: str) -> VisionScores:
    """Parse <dimensions> scores from Gemini's vision comparison response."""
    scores: dict[str, VisionDimensionScore] = {}
    for match in _DIMENSION_RE.finditer(text):
        dim_name = match.group(1)
        score = float(match.group(2))
        assessment = match.group(3).strip()
        if dim_name in ("style", "subject", "color", "composition"):
            scores[dim_name] = VisionDimensionScore(
                dimension=dim_name,
                score=min(max(score, 1.0), 10.0),  # clamp to [1, 10]
                assessment=assessment,
            )

    return VisionScores(
        style=scores.get("style", VisionDimensionScore("style", 5.0, "")),
        subject=scores.get("subject", VisionDimensionScore("subject", 5.0, "")),
        color=scores.get("color", VisionDimensionScore("color", 5.0, "")),
        composition=scores.get("composition", VisionDimensionScore("composition", 5.0, "")),
    )


async def compare_vision(
    pairs: list[tuple[Path, Path]],
    captions: list[str],
    rendered_prompt: str,
    *,
    client: genai.Client,
    model: str,
    semaphore: asyncio.Semaphore,
    max_pairs: int = 5,
) -> tuple[str, VisionScores]:
    """Compare (original, generated) pairs via Gemini vision.

    Shows up to *max_pairs* pairs, deterministically picking the first N.
    Pairs should be pre-sorted worst-first for best feedback quality.
    Returns (qualitative_text, structured_scores).
    """
    show_pairs = pairs[:max_pairs]
    show_captions = captions[:max_pairs]

    contents: list[object] = []
    for i, ((ref_path, gen_path), caption) in enumerate(zip(show_pairs, show_captions, strict=True)):
        contents.append(f"\n## Pair {i + 1} — {ref_path.name}:")
        contents.append(f"### Caption used for generation:\n{caption[:600]}")
        contents.append("### ORIGINAL:")
        contents.append(image_to_gemini_part(ref_path))
        contents.append("### GENERATED (from caption above):")
        contents.append(image_to_gemini_part(gen_path))

    contents.append(f"\n{_VISION_COMPARE_PROMPT.format(rendered_prompt=rendered_prompt)}")

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            async with semaphore:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                )
            text = response.text
            scores = _parse_vision_scores(text)
            return text, scores
        except Exception as exc:
            last_exc = exc
            delay = 3.0 * (2**attempt)
            logger.warning(
                "Vision comparison attempt %d/3 failed: %s: %s — retrying in %.0fs",
                attempt + 1,
                type(exc).__name__,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    msg = "Vision comparison failed after 3 retries"
    raise RuntimeError(msg) from last_exc


def check_caption_compliance(
    section_names: list[str],
    captions: object,
) -> str:
    """Check whether captions address the topics from the meta-prompt sections.

    Returns a summary of which sections are well-covered vs missed.
    """
    from art_style_search.types import Caption

    typed = [c for c in captions if isinstance(c, Caption)]
    if not typed or not section_names:
        return ""

    # Simple keyword presence check per section
    section_hits: dict[str, int] = {name: 0 for name in section_names}
    for caption in typed:
        text_lower = caption.text.lower()
        for name in section_names:
            # Check if section topic keywords appear in caption
            keywords = name.replace("_", " ").split()
            if any(kw in text_lower for kw in keywords):
                section_hits[name] += 1

    total = len(typed)
    lines: list[str] = []
    for name, hits in section_hits.items():
        pct = hits / total * 100
        status = "OK" if pct >= 70 else "WEAK" if pct >= 30 else "MISSING"
        lines.append(f"  {name}: {status} ({hits}/{total} captions address this)")

    return "Caption compliance with meta-prompt sections:\n" + "\n".join(lines)


_ROUNDTRIP_COMPARE_PROMPT = (
    "You are an expert art analyst. For each pair below, the ORIGINAL is a reference artwork "
    "and the GENERATED image was created from a text caption of that original.\n\n"
    "For each pair, describe:\n"
    "<pair>\n"
    "<captured>What aspects of the original the generated image reproduces well</captured>\n"
    "<lost>What aspects were lost or changed — focus on style, color, technique, mood, details</lost>\n"
    "<caption_gap>What the caption likely failed to describe that would be needed to reproduce the original</caption_gap>\n"
    "</pair>\n\n"
    "After all pairs, provide:\n"
    "<summary>Overall patterns: what do captions consistently miss? "
    "What style elements are hardest to capture in text? "
    "What specific descriptors should be added to prompts to close these gaps?</summary>"
)


async def caption_roundtrip_test(
    reference_paths: list[Path],
    captions: list[object],
    *,
    gemini_client: object,
    caption_model: str,
    generator_model: str,
    gen_semaphore: asyncio.Semaphore,
    output_dir: Path,
    iteration: int,
    aspect_ratio: str,
    num_test_images: int = 4,
) -> str:
    """Generate images from reference captions and compare with originals via Gemini vision.

    Picks *num_test_images* reference images, generates an image from each caption,
    then asks Gemini to compare original vs generated to identify what captions miss.
    """
    from art_style_search.generate import _generate_single
    from art_style_search.types import Caption

    typed_captions: list[Caption] = [c for c in captions if isinstance(c, Caption)]

    # Match captions to paths
    caption_by_path = {c.image_path: c for c in typed_captions}
    available = [p for p in reference_paths if p in caption_by_path]
    if not available:
        return ""

    sample = random.sample(available, min(num_test_images, len(available)))

    # Generate images from captions
    roundtrip_dir = output_dir / f"iter_{iteration:03d}" / "roundtrip"
    roundtrip_dir.mkdir(parents=True, exist_ok=True)

    gen_tasks = []
    for i, ref_path in enumerate(sample):
        caption_text = caption_by_path[ref_path].text
        gen_tasks.append(
            _generate_single(
                caption_text,
                index=i,
                aspect_ratio=aspect_ratio,
                output_path=roundtrip_dir / f"rt_{i:02d}.png",
                client=gemini_client,
                model=generator_model,
                semaphore=gen_semaphore,
            )
        )

    gen_results = await asyncio.gather(*gen_tasks, return_exceptions=True)

    # Build comparison contents for Gemini vision
    pairs: list[tuple[Path, Path]] = []
    for i, (ref_path, gen_result) in enumerate(zip(sample, gen_results, strict=True)):
        if isinstance(gen_result, BaseException):
            logger.warning("Roundtrip generation %d failed: %s", i, gen_result)
            continue
        pairs.append((ref_path, gen_result))

    if not pairs:
        return ""

    contents: list[object] = []
    for i, (ref_path, gen_path) in enumerate(pairs):
        contents.append(f"\n## Pair {i + 1}:\n### ORIGINAL:")
        contents.append(image_to_gemini_part(ref_path))
        contents.append("### GENERATED (from caption):")
        contents.append(image_to_gemini_part(gen_path))

    contents.append(f"\n{_ROUNDTRIP_COMPARE_PROMPT}")

    async with gen_semaphore:
        response = await gemini_client.aio.models.generate_content(
            model=caption_model,
            contents=contents,
        )

    return response.text


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
    ssim_vals = [s.ssim for s in scores]
    color_vals = [s.color_histogram for s in scores]

    return AggregatedMetrics(
        dino_similarity_mean=_mean(dino),
        dino_similarity_std=_std(dino),
        lpips_distance_mean=_mean(lpips_vals),
        lpips_distance_std=_std(lpips_vals),
        hps_score_mean=_mean(hps),
        hps_score_std=_std(hps),
        aesthetics_score_mean=_mean(aes),
        aesthetics_score_std=_std(aes),
        ssim_mean=_mean(ssim_vals),
        ssim_std=_std(ssim_vals),
        color_histogram_mean=_mean(color_vals),
        color_histogram_std=_std(color_vals),
    )


def _compute_single_sync(
    registry: ModelRegistry,
    generated: Image.Image,
    references: list[Image.Image],
    prompt: str,
) -> MetricScores:
    """Compute all 6 per-image metrics for one generated image (synchronous)."""
    ref = references[0]  # always a single paired reference
    return MetricScores(
        dino_similarity=registry.compute_dino(generated, references),
        lpips_distance=registry.compute_lpips(generated, references),
        hps_score=registry.compute_hps(generated, prompt),
        aesthetics_score=registry.compute_aesthetics(generated),
        ssim=registry.compute_ssim(generated, ref),
        color_histogram=registry.compute_color_histogram(generated, ref),
    )


async def evaluate_images(
    generated_paths: list[Path],
    reference_paths: list[Path],
    captions: list[str],
    *,
    registry: ModelRegistry,
    semaphore: asyncio.Semaphore,
) -> tuple[list[MetricScores], AggregatedMetrics]:
    """Evaluate each generated image against its paired reference.

    Each generated image is compared against ONLY its corresponding reference
    (not the mean of all references).  HPS is scored against the per-image
    caption (the actual generation prompt), not the meta-prompt.

    Returns per-image scores and aggregated statistics.
    """

    async def _eval_one(gen_path: Path, ref_path: Path, caption: str) -> MetricScores | None:
        async with semaphore:
            gen_image: Image.Image | None = None
            ref_image: Image.Image | None = None
            try:
                gen_image = Image.open(gen_path).convert("RGB")
                ref_image = Image.open(ref_path).convert("RGB")
                scores = await asyncio.to_thread(_compute_single_sync, registry, gen_image, [ref_image], caption)
                return scores
            except Exception as exc:
                logger.warning("Evaluation failed for %s: %s", gen_path.name, exc)
                return None
            finally:
                if gen_image is not None:
                    gen_image.close()
                if ref_image is not None:
                    ref_image.close()

    results = await asyncio.gather(
        *[_eval_one(gp, rp, cap) for gp, rp, cap in zip(generated_paths, reference_paths, captions, strict=True)]
    )

    scores = [r for r in results if r is not None]
    aggregated = _aggregate(scores)

    return scores, aggregated
