"""Dispatch metric computations via asyncio.to_thread through ModelRegistry."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from google import genai
from PIL import Image

from art_style_search.models import ModelRegistry
from art_style_search.types import (
    VISION_VERDICT_DEFAULT,
    VISION_VERDICT_MAP,
    AggregatedMetrics,
    MetricScores,
    VisionDimensionScore,
    VisionScores,
)
from art_style_search.utils import image_to_gemini_part

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-image vision comparison (ternary: MATCH / PARTIAL / MISS)
# ---------------------------------------------------------------------------

_VISION_SINGLE_PROMPT = (
    "Compare the ORIGINAL reference image with the GENERATED reproduction.\n"
    "The caption used to generate was:\n{caption}\n\n"
    "Be BRUTALLY HONEST. Focus on what is WRONG.\n\n"
    "For each dimension, judge as:\n"
    "- MATCH: reproduction captures this aspect well\n"
    "- PARTIAL: some aspects captured, but notable differences\n"
    "- MISS: significant failure to reproduce this aspect\n\n"
    "Respond in EXACTLY this format:\n"
    '<style verdict="MATCH|PARTIAL|MISS">1-sentence explanation</style>\n'
    '<subject verdict="MATCH|PARTIAL|MISS">1-sentence explanation about character/subject fidelity</subject>\n'
    '<composition verdict="MATCH|PARTIAL|MISS">1-sentence explanation about spatial layout</composition>\n'
    "<key_gap>The single most critical thing the caption missed or the generator failed on</key_gap>"
)

_VERDICT_RE = re.compile(
    r'<(\w+)\s+verdict="(\w+)">(.*?)</\1>',
    re.DOTALL,
)


def _parse_vision_verdicts(text: str) -> VisionScores:
    """Parse ternary verdicts from per-image Gemini vision comparison."""
    scores: dict[str, VisionDimensionScore] = {}
    for match in _VERDICT_RE.finditer(text):
        dim_name = match.group(1)
        verdict = match.group(2).upper()
        assessment = match.group(3).strip()
        if dim_name in ("style", "subject", "composition"):
            score = VISION_VERDICT_MAP.get(verdict, VISION_VERDICT_DEFAULT)
            scores[dim_name] = VisionDimensionScore(dimension=dim_name, score=score, assessment=assessment)

    return VisionScores(
        style=scores.get("style", VisionDimensionScore("style", VISION_VERDICT_DEFAULT, "")),
        subject=scores.get("subject", VisionDimensionScore("subject", VISION_VERDICT_DEFAULT, "")),
        composition=scores.get("composition", VisionDimensionScore("composition", VISION_VERDICT_DEFAULT, "")),
    )


async def _compare_vision_single(
    ref_path: Path,
    gen_path: Path,
    caption: str,
    *,
    client: genai.Client,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, VisionScores]:
    """Compare a single (original, generated) pair via Gemini vision.

    Returns (qualitative_feedback, vision_scores) for this one image.
    """
    contents: list[object] = [
        "### ORIGINAL:",
        image_to_gemini_part(ref_path),
        "### GENERATED (from caption):",
        image_to_gemini_part(gen_path),
        _VISION_SINGLE_PROMPT.format(caption=caption[:600]),
    ]

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            async with semaphore:
                response = await client.aio.models.generate_content(model=model, contents=contents)
            text = response.text or ""
            scores = _parse_vision_verdicts(text)
            return text, scores
        except Exception as exc:
            last_exc = exc
            delay = 3.0 * (2**attempt)
            logger.warning(
                "Vision %s attempt %d/3 failed: %s — retrying in %.0fs",
                ref_path.name,
                attempt + 1,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    logger.error("Vision %s failed after 3 retries: %s — using neutral defaults", ref_path.name, last_exc)
    return "", VisionScores.default()


async def compare_vision_per_image(
    pairs: list[tuple[Path, Path]],
    captions: list[str],
    *,
    client: genai.Client,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[list[str], list[VisionScores]]:
    """Compare each (original, generated) pair individually via Gemini vision.

    Returns (list_of_feedback_texts, list_of_vision_scores), one per pair.
    """
    tasks = [
        _compare_vision_single(ref, gen, cap, client=client, model=model, semaphore=semaphore)
        for (ref, gen), cap in zip(pairs, captions, strict=True)
    ]
    results = await asyncio.gather(*tasks)
    feedbacks = [text for text, _ in results]
    scores = [vs for _, vs in results]
    return feedbacks, scores


def check_caption_compliance(
    section_names: list[str],
    captions: object,
    caption_sections: list[str] | None = None,
) -> str:
    """Check whether captions address the topics from the meta-prompt sections.

    When *caption_sections* is provided, also checks for the presence of
    labeled ``[Section Name]`` markers in the caption text.

    Returns a summary of which sections are well-covered vs missed.
    """
    from art_style_search.types import Caption

    typed = [c for c in captions if isinstance(c, Caption)]
    if not typed or not section_names:
        return ""

    total = len(typed)
    lines: list[str] = []
    lowered = [c.text.lower() for c in typed]

    # Keyword presence check per meta-prompt section
    section_hits: dict[str, int] = {name: 0 for name in section_names}
    for text_lower in lowered:
        for name in section_names:
            keywords = name.replace("_", " ").split()
            if any(kw in text_lower for kw in keywords):
                section_hits[name] += 1

    for name, hits in section_hits.items():
        pct = hits / total * 100
        status = "OK" if pct >= 70 else "WEAK" if pct >= 30 else "MISSING"
        lines.append(f"  {name}: {status} ({hits}/{total} captions address this)")

    # Labeled section marker check (e.g. "[Art Style]" in caption text)
    if caption_sections:
        marker_lines: list[str] = []
        for sec_name in caption_sections:
            marker = f"[{sec_name}]".lower()
            hits = sum(1 for tl in lowered if marker in tl)
            pct = hits / total * 100
            status = "OK" if pct >= 70 else "WEAK" if pct >= 30 else "MISSING"
            marker_lines.append(f"  [{sec_name}]: {status} ({hits}/{total} captions contain this label)")
        lines.append("Labeled section markers in captions:")
        lines.extend(marker_lines)

    return "Caption compliance with meta-prompt sections:\n" + "\n".join(lines)


def compute_style_consistency(captions: object, section_label: str = "Art Style") -> float:
    """Measure how consistent the [Art Style] blocks are across captions.

    Extracts the text between ``[Art Style]`` and the next ``[`` marker from
    each caption, then computes the mean pairwise word-overlap (Jaccard)
    similarity.  Returns a float in [0, 1]; 1.0 means all style blocks are
    identical.  Returns 0.0 if fewer than 2 captions contain the label.
    """
    from art_style_search.types import Caption

    typed = [c for c in captions if isinstance(c, Caption)]
    if len(typed) < 2:
        return 0.0

    import re

    marker = re.escape(f"[{section_label}]")
    # Extract text from [Art Style] up to the next [SectionName] or end
    pattern = re.compile(marker + r"\s*(.*?)(?=\n\[|\Z)", re.DOTALL | re.IGNORECASE)

    blocks: list[set[str]] = []
    for cap in typed:
        m = pattern.search(cap.text)
        if m:
            words = set(m.group(1).lower().split())
            if words:
                blocks.append(words)

    if len(blocks) < 2:
        return 0.0

    # Mean pairwise Jaccard similarity
    total = 0.0
    count = 0
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            intersection = len(blocks[i] & blocks[j])
            union = len(blocks[i] | blocks[j])
            total += intersection / union if union else 0.0
            count += 1

    return total / count if count else 0.0


def aggregate(scores: list[MetricScores]) -> AggregatedMetrics:
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
    color_vals = [s.color_histogram for s in scores]
    texture_vals = [s.texture for s in scores]
    ssim_vals = [s.ssim for s in scores]
    v_style = [s.vision_style for s in scores]
    v_subject = [s.vision_subject for s in scores]
    v_composition = [s.vision_composition for s in scores]

    return AggregatedMetrics(
        dino_similarity_mean=_mean(dino),
        dino_similarity_std=_std(dino),
        lpips_distance_mean=_mean(lpips_vals),
        lpips_distance_std=_std(lpips_vals),
        hps_score_mean=_mean(hps),
        hps_score_std=_std(hps),
        aesthetics_score_mean=_mean(aes),
        aesthetics_score_std=_std(aes),
        color_histogram_mean=_mean(color_vals),
        color_histogram_std=_std(color_vals),
        texture_mean=_mean(texture_vals),
        texture_std=_std(texture_vals),
        ssim_mean=_mean(ssim_vals),
        ssim_std=_std(ssim_vals),
        vision_style=_mean(v_style),
        vision_style_std=_std(v_style),
        vision_subject=_mean(v_subject),
        vision_subject_std=_std(v_subject),
        vision_composition=_mean(v_composition),
        vision_composition_std=_std(v_composition),
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
        color_histogram=registry.compute_color_histogram(generated, ref),
        texture=registry.compute_texture(generated, ref),
        ssim=registry.compute_ssim(generated, ref),
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
    aggregated = aggregate(scores)

    return scores, aggregated
