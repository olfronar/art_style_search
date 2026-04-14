"""Dispatch metric computations via asyncio.to_thread through ModelRegistry."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from pathlib import Path
from typing import Any

from google import genai  # type: ignore[attr-defined]
from PIL import Image

from art_style_search.caption_sections import parse_labeled_sections
from art_style_search.media import image_to_xai_data_url
from art_style_search.models import ModelRegistry
from art_style_search.types import (
    VISION_VERDICT_DEFAULT,
    VISION_VERDICT_MAP,
    AggregatedMetrics,
    Caption,
    CaptionComplianceStats,
    MetricScores,
    VisionDimensionScore,
    VisionScores,
)
from art_style_search.utils import async_retry, extract_xml_tag, image_to_gemini_part, vision_circuit_breaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-image vision comparison (ternary: MATCH / PARTIAL / MISS)
# ---------------------------------------------------------------------------

_VISION_SINGLE_PROMPT = (
    "Compare the ORIGINAL reference image with the GENERATED reproduction.\n"
    "The caption used to generate was:\n{caption}\n\n"
    "Assess each dimension independently and neutrally.\n\n"
    "For each dimension, judge as:\n"
    "- MATCH: reproduction captures this aspect well. Concrete standard: a viewer would "
    "recognize it as the same image in this dimension without hesitation.\n"
    "- PARTIAL: correct general direction but notable errors. Examples: right subject but "
    "wrong expression; correct palette but shifted proportions; recognizable layout with "
    "missing or added elements.\n"
    "- MISS: significant failure. Examples: different subject identity; wrong dominant "
    "palette; unrecognizable spatial arrangement.\n\n"
    "Respond in EXACTLY this format:\n"
    '<style verdict="MATCH|PARTIAL|MISS">1-sentence explanation</style>\n'
    '<subject verdict="MATCH|PARTIAL|MISS">1-sentence explanation about character/subject fidelity</subject>\n'
    '<composition verdict="MATCH|PARTIAL|MISS">1-sentence explanation about spatial layout</composition>\n'
)
_VISION_SYSTEM = (
    "You are a careful visual judge comparing a reference image to a generated reproduction. "
    "Use the rubric exactly as given and return only the requested pseudo-XML tags."
)
_VISION_CAPTION_CHAR_LIMIT = 3500

_VERDICT_RE = re.compile(
    r'<(\w+)\s+verdict="(\w+)">(.*?)</\1>',
    re.DOTALL,
)

_SUBJECT_MIN_WORDS = 80
_SUBJECT_MIN_FACETS = 4
_SUBJECT_MIN_SPECIFIC_WORDS = 8
_SUBJECT_WORD_RE = re.compile(r"[a-z0-9'-]+")
_SUBJECT_GENERIC_WORDS = {
    "figure",
    "person",
    "people",
    "character",
    "subject",
    "creature",
    "animal",
    "object",
    "thing",
}
_SUBJECT_FILLER_WORDS = {
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
    "into",
    "near",
    "over",
    "under",
    "while",
    "main",
    "scene",
}
_SUBJECT_FACET_KEYWORDS = {
    "identity_species": {
        "man",
        "woman",
        "girl",
        "boy",
        "child",
        "fox",
        "wolf",
        "cat",
        "dog",
        "bird",
        "horse",
        "deer",
        "rabbit",
        "robot",
        "knight",
        "soldier",
        "merchant",
        "traveler",
        "animal",
        "character",
    },
    "distinguishing_features": {
        "eyes",
        "eye",
        "ear",
        "ears",
        "scar",
        "stripes",
        "spots",
        "tail",
        "fur",
        "hair",
        "muzzle",
        "markings",
        "beak",
        "horn",
        "horns",
        "face",
        "jaw",
        "paws",
        "hands",
    },
    "clothing_equipment": {
        "wears",
        "wearing",
        "coat",
        "cloak",
        "jacket",
        "armor",
        "dress",
        "shirt",
        "hat",
        "boots",
        "satchel",
        "bag",
        "sword",
        "lantern",
        "helmet",
        "harness",
        "scarf",
        "gloves",
        "belt",
    },
    "pose_action": {
        "standing",
        "sitting",
        "running",
        "walking",
        "mid-step",
        "leaning",
        "turning",
        "reaching",
        "holding",
        "lifting",
        "crouching",
        "jumping",
        "trotting",
        "twisting",
        "posed",
        "stride",
        "step",
    },
    "expression": {
        "expression",
        "smile",
        "frown",
        "grim",
        "alert",
        "wary",
        "calm",
        "angry",
        "joyful",
        "focused",
        "glance",
        "stare",
        "mouth",
        "brow",
        "gaze",
    },
    "props_context": {
        "nearby",
        "beside",
        "surrounded",
        "props",
        "map",
        "lantern",
        "reeds",
        "chair",
        "table",
        "basket",
        "rope",
        "field",
        "background",
        "context",
        "marsh",
        "grass",
    },
}


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


async def _compare_vision_single_gemini(
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
    swapped = random.random() < 0.5
    first_label = "### GENERATED reproduction:" if swapped else "### ORIGINAL reference:"
    second_label = "### ORIGINAL reference:" if swapped else "### GENERATED reproduction:"
    first_image = gen_path if swapped else ref_path
    second_image = ref_path if swapped else gen_path
    contents: list[object] = [
        first_label,
        image_to_gemini_part(first_image),
        second_label,
        image_to_gemini_part(second_image),
        _VISION_SINGLE_PROMPT.format(caption=caption[:_VISION_CAPTION_CHAR_LIMIT]),
    ]

    async def _call() -> tuple[str, VisionScores]:
        async with semaphore:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(system_instruction=_VISION_SYSTEM),
                ),
                timeout=90,
            )
        text = response.text or ""
        return text, _parse_vision_verdicts(text)

    try:
        return await async_retry(_call, label=f"Vision {ref_path.name}", circuit_breaker=vision_circuit_breaker)
    except RuntimeError:
        logger.error("Vision %s failed after retries — using neutral defaults", ref_path.name)
        return "", VisionScores.default()


async def _compare_vision_single_xai(
    ref_path: Path,
    gen_path: Path,
    caption: str,
    *,
    client: Any,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, VisionScores]:
    """Compare a single (original, generated) pair via xAI multimodal responses."""
    swapped = random.random() < 0.5
    first_label = "### GENERATED reproduction:" if swapped else "### ORIGINAL reference:"
    second_label = "### ORIGINAL reference:" if swapped else "### GENERATED reproduction:"
    first_image = gen_path if swapped else ref_path
    second_image = ref_path if swapped else gen_path
    contents = [
        {"type": "input_text", "text": first_label},
        {"type": "input_image", "image_url": image_to_xai_data_url(first_image), "detail": "high"},
        {"type": "input_text", "text": second_label},
        {"type": "input_image", "image_url": image_to_xai_data_url(second_image), "detail": "high"},
        {"type": "input_text", "text": _VISION_SINGLE_PROMPT.format(caption=caption[:_VISION_CAPTION_CHAR_LIMIT])},
    ]

    async def _call() -> tuple[str, VisionScores]:
        async with semaphore:
            response = await asyncio.wait_for(
                client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": _VISION_SYSTEM},
                        {"role": "user", "content": contents},
                    ],
                    max_output_tokens=1000,
                    store=False,
                ),
                timeout=90,
            )
        text = response.output_text or ""
        return text, _parse_vision_verdicts(text)

    try:
        return await async_retry(_call, label=f"xAI vision {ref_path.name}", circuit_breaker=vision_circuit_breaker)
    except RuntimeError:
        logger.error("xAI vision %s failed after retries — using neutral defaults", ref_path.name)
        return "", VisionScores.default()


async def compare_vision_per_image(
    pairs: list[tuple[Path, Path]],
    captions: list[str],
    *,
    provider: str,
    model: str,
    semaphore: asyncio.Semaphore,
    client: genai.Client | None = None,
    xai_client: Any | None = None,
) -> tuple[list[str], list[VisionScores]]:
    """Compare each (original, generated) pair individually via the configured provider.

    Returns (list_of_feedback_texts, list_of_vision_scores), one per pair.
    """
    if provider == "gemini":
        if client is None:
            msg = "Gemini comparison requires a Gemini client"
            raise ValueError(msg)
        tasks = [
            _compare_vision_single_gemini(ref, gen, cap, client=client, model=model, semaphore=semaphore)
            for (ref, gen), cap in zip(pairs, captions, strict=True)
        ]
    elif provider == "xai":
        if xai_client is None:
            msg = "xAI comparison requires an xAI client"
            raise ValueError(msg)
        tasks = [
            _compare_vision_single_xai(ref, gen, cap, client=xai_client, model=model, semaphore=semaphore)
            for (ref, gen), cap in zip(pairs, captions, strict=True)
        ]
    else:
        msg = f"Unknown comparison provider: {provider}"
        raise ValueError(msg)

    logger.info("Vision comparison: scoring %d image pairs", len(tasks))
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    feedbacks: list[str] = []
    scores: list[VisionScores] = []
    for i, result in enumerate(raw_results):
        if isinstance(result, BaseException):
            if isinstance(result, ValueError):
                raise result
            logger.warning("Vision pair %d failed: %s: %s", i, type(result).__name__, result)
            feedbacks.append("")
            scores.append(VisionScores.default())
        else:
            feedbacks.append(result[0])
            scores.append(result[1])
    logger.info("Vision comparison: done (%d pairs scored)", len(scores))
    return feedbacks, scores


# ---------------------------------------------------------------------------
# Pairwise experiment comparison (SPO-inspired)
# ---------------------------------------------------------------------------

_PAIRWISE_COMPARE_PROMPT = (
    "You are comparing two sets of art reproductions against the same original reference images.\n\n"
    "SET A and SET B each attempted to reproduce the same originals using different meta-prompt strategies.\n\n"
    "## Evaluation method\n"
    "1. First, evaluate each set INDEPENDENTLY — how well does it reproduce style, color, subject, "
    "and composition relative to the original?\n"
    "2. Then compare: which set is consistently closer to the original across the image trios?\n"
    "3. Do NOT let the order (A-first or B-first) influence your judgment.\n"
    "4. Focus on concrete visual differences, not overall impressions.\n\n"
    "## Aspects to compare\n"
    "- Art style and technique (brushwork, rendering, visual treatment)\n"
    "- Color palette and mood (dominant colors, temperature, atmosphere)\n"
    "- Subject matter and composition (identity, pose, spatial layout)\n"
    "- Overall fidelity to the original\n\n"
    "Respond with:\n"
    "<winner>A</winner> or <winner>B</winner> or <winner>TIE</winner>\n"
    "<rationale>1-3 sentences explaining your overall judgment across all image trios</rationale>"
)
_PAIRWISE_SYSTEM = (
    "You are a careful visual judge comparing two reproduction sets against the same references. "
    "Evaluate the trios independently, ignore ordering bias, and return only the requested tags."
)


async def pairwise_compare_experiments(
    pairs_a: list[tuple[Path, Path]],
    pairs_b: list[tuple[Path, Path]],
    *,
    provider: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_images: int = 3,
    client: genai.Client | None = None,
    xai_client: Any | None = None,
) -> tuple[str, float]:
    """Compare two experiments' outputs via the configured provider.

    Samples up to *max_images* representative pairs (evenly spaced) from each
    experiment.  Returns (rationale, score_for_a) where score_for_a is 1.0 if
    A wins, 0.0 if B wins, 0.5 for tie.
    """
    n = min(len(pairs_a), len(pairs_b))
    if n == 0:
        return ("No images to compare", 0.5)
    step = max(1, n // max_images)
    indices = list(range(0, n, step))[:max_images]

    # Randomize A/B ordering to eliminate position bias in LLM comparisons
    swapped = random.random() < 0.5
    if swapped:
        pairs_a, pairs_b = pairs_b, pairs_a

    if provider == "gemini":
        if client is None:
            msg = "Gemini comparison requires a Gemini client"
            raise ValueError(msg)
        contents: list[object] = []
        for idx in indices:
            ref_a, gen_a = pairs_a[idx]
            _, gen_b = pairs_b[idx]
            contents.extend(
                [
                    f"### Image {idx + 1} — ORIGINAL:",
                    image_to_gemini_part(ref_a),
                    f"### Image {idx + 1} — SET A reproduction:",
                    image_to_gemini_part(gen_a),
                    f"### Image {idx + 1} — SET B reproduction:",
                    image_to_gemini_part(gen_b),
                ]
            )
        contents.append(_PAIRWISE_COMPARE_PROMPT)

        async def _call() -> tuple[str, float]:
            async with semaphore:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=genai.types.GenerateContentConfig(system_instruction=_PAIRWISE_SYSTEM),
                    ),
                    timeout=120,
                )
            text = response.text or ""
            winner = (extract_xml_tag(text, "winner") or "TIE").upper()
            rationale = extract_xml_tag(text, "rationale") or text[:300]
            score = {"A": 1.0, "B": 0.0, "TIE": 0.5}.get(winner, 0.5)
            return (rationale, score)
    elif provider == "xai":
        if xai_client is None:
            msg = "xAI comparison requires an xAI client"
            raise ValueError(msg)

        contents = []
        for idx in indices:
            ref_a, gen_a = pairs_a[idx]
            _, gen_b = pairs_b[idx]
            contents.extend(
                [
                    {"type": "input_text", "text": f"### Image {idx + 1} — ORIGINAL:"},
                    {"type": "input_image", "image_url": image_to_xai_data_url(ref_a), "detail": "high"},
                    {"type": "input_text", "text": f"### Image {idx + 1} — SET A reproduction:"},
                    {"type": "input_image", "image_url": image_to_xai_data_url(gen_a), "detail": "high"},
                    {"type": "input_text", "text": f"### Image {idx + 1} — SET B reproduction:"},
                    {"type": "input_image", "image_url": image_to_xai_data_url(gen_b), "detail": "high"},
                ]
            )
        contents.append({"type": "input_text", "text": _PAIRWISE_COMPARE_PROMPT})

        async def _call() -> tuple[str, float]:
            async with semaphore:
                response = await asyncio.wait_for(
                    xai_client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": _PAIRWISE_SYSTEM},
                            {"role": "user", "content": contents},
                        ],
                        max_output_tokens=1000,
                        store=False,
                    ),
                    timeout=120,
                )
            text = response.output_text or ""
            winner = (extract_xml_tag(text, "winner") or "TIE").upper()
            rationale = extract_xml_tag(text, "rationale") or text[:300]
            score = {"A": 1.0, "B": 0.0, "TIE": 0.5}.get(winner, 0.5)
            return (rationale, score)
    else:
        msg = f"Unknown comparison provider: {provider}"
        raise ValueError(msg)

    try:
        rationale, score = await async_retry(_call, label="Pairwise comparison", circuit_breaker=vision_circuit_breaker)
    except RuntimeError:
        logger.error("Pairwise comparison failed after retries")
        return ("Comparison failed", 0.5)

    # Flip the score back if we swapped A/B for position-bias mitigation
    if swapped:
        score = 1.0 - score
    return (rationale, score)


def _ordering_from_parsed(parsed: dict[str, str], expected_sections: list[str]) -> str:
    """Check if the parsed section keys appear in the expected order."""
    present = [name for name in parsed if name in set(expected_sections)]
    if len(present) < 2:
        return "SKIP"

    expected_present = [s for s in expected_sections if s in present]
    if present == expected_present:
        return "OK"
    return f"MISORDERED (expected {expected_present[:3]}..., got {present[:3]}...)"


def _lengths_from_parsed(parsed: dict[str, str], expected_sections: list[str]) -> str:
    """Check if parsed sections have roughly proportional word counts."""
    section_texts = {name: body for name, body in parsed.items() if name in set(expected_sections)}
    if not section_texts:
        return "SKIP"

    word_counts = {name: len(body.split()) for name, body in section_texts.items()}
    total_words = sum(word_counts.values())
    if total_words == 0:
        return "EMPTY"

    issues: list[str] = []
    for name, count in word_counts.items():
        ratio = count / total_words
        if ratio > 0.50:
            issues.append(f"{name} too long ({ratio:.0%})")
        elif ratio < 0.05 and len(expected_sections) <= 6:
            issues.append(f"{name} too short ({ratio:.0%})")

    return "OK" if not issues else f"IMBALANCED: {'; '.join(issues)}"


def _subject_specificity_from_parsed(parsed: dict[str, str]) -> str:
    """Check whether the ``[Subject]`` block is detailed and specific."""
    subject_text = parsed.get("Subject", "").strip()
    if not subject_text:
        return "MISSING"

    lowered = subject_text.lower()
    words = _SUBJECT_WORD_RE.findall(lowered)
    if len(words) < _SUBJECT_MIN_WORDS:
        return f"WEAK (too short: {len(words)} words)"

    n_facets = len(_SUBJECT_FACET_KEYWORDS)
    facet_count = sum(1 for keywords in _SUBJECT_FACET_KEYWORDS.values() if any(kw in lowered for kw in keywords))
    if facet_count < _SUBJECT_MIN_FACETS:
        return f"WEAK (facet coverage {facet_count}/{n_facets})"

    meaningful = [w for w in words if len(w) > 2 and w not in _SUBJECT_FILLER_WORDS]
    generic_count = sum(1 for w in meaningful if w in _SUBJECT_GENERIC_WORDS)
    specific_count = len(meaningful) - generic_count
    if generic_count > 0 and specific_count < max(_SUBJECT_MIN_SPECIFIC_WORDS, generic_count * 2):
        return "WEAK (generic subject terms without enough modifiers)"

    return "OK"


def _check_section_ordering(caption_text: str, expected_sections: list[str]) -> str:
    """Check if labeled sections appear in the expected order."""
    return _ordering_from_parsed(parse_labeled_sections(caption_text), expected_sections)


def _check_section_lengths(caption_text: str, expected_sections: list[str]) -> str:
    """Check if sections have roughly proportional word counts."""
    return _lengths_from_parsed(parse_labeled_sections(caption_text), expected_sections)


def _check_subject_specificity(caption_text: str) -> str:
    """Check whether the ``[Subject]`` block is detailed and specific."""
    return _subject_specificity_from_parsed(parse_labeled_sections(caption_text))


def compute_caption_compliance(
    section_names: list[str],
    captions: list[Caption],
    caption_sections: list[str] | None = None,
) -> tuple[CaptionComplianceStats, str]:
    """Parse every caption once and produce both structured stats and prose.

    Captions are parsed a single time via ``parse_labeled_sections`` and the
    ordering / length / subject checks all share that result.  Returns
    ``(stats, prose)`` — callers wanting only one piece discard the other.
    """
    has_subject = bool(caption_sections and "Subject" in caption_sections)
    if not captions or not section_names:
        empty_stats = CaptionComplianceStats(
            section_topic_coverage=0.0,
            section_marker_coverage=0.0 if caption_sections else 1.0,
            section_ordering_rate=0.0 if caption_sections else 1.0,
            section_balance_rate=0.0 if caption_sections else 1.0,
            subject_specificity_rate=0.0 if has_subject else 1.0,
        )
        return empty_stats, ""

    total = len(captions)
    lowered = [c.text.lower() for c in captions]
    parsed = [parse_labeled_sections(c.text) for c in captions]

    # Topic coverage: keywords derived from each meta-prompt section name
    section_hits: dict[str, int] = {}
    for name in section_names:
        keywords = name.replace("_", " ").split()
        section_hits[name] = sum(1 for tl in lowered if any(kw in tl for kw in keywords))
    topic_coverage = sum(section_hits.values()) / (len(section_names) * total)

    # Marker presence: count captions containing each labeled `[Section]` marker
    marker_hits: dict[str, int] = {}
    if caption_sections:
        for sec_name in caption_sections:
            marker = f"[{sec_name}]".lower()
            marker_hits[sec_name] = sum(1 for tl in lowered if marker in tl)
        marker_coverage = sum(marker_hits.values()) / (len(caption_sections) * total)
    else:
        marker_coverage = 1.0

    # Per-caption structural checks, computed once from parsed sections
    ordering_results: list[str] = []
    length_results: list[str] = []
    subject_results: list[str] = []
    if caption_sections:
        ordering_results = [_ordering_from_parsed(p, caption_sections) for p in parsed]
        length_results = [_lengths_from_parsed(p, caption_sections) for p in parsed]
    if has_subject:
        subject_results = [_subject_specificity_from_parsed(p) for p in parsed]

    if caption_sections:
        checked_ordering = [r for r in ordering_results if r != "SKIP"]
        ordering_rate = (
            sum(1 for r in checked_ordering if r == "OK") / len(checked_ordering) if checked_ordering else 0.0
        )
        checked_lengths = [r for r in length_results if r not in {"SKIP", "EMPTY"}]
        balance_rate = sum(1 for r in checked_lengths if r == "OK") / len(checked_lengths) if checked_lengths else 0.0
    else:
        ordering_rate = 1.0
        balance_rate = 1.0

    subject_specificity_rate = sum(1 for r in subject_results if r == "OK") / total if has_subject else 1.0

    stats = CaptionComplianceStats(
        section_topic_coverage=topic_coverage,
        section_marker_coverage=marker_coverage,
        section_ordering_rate=ordering_rate,
        section_balance_rate=balance_rate,
        subject_specificity_rate=subject_specificity_rate,
    )

    # Prose summary — same per-caption results drive the human-readable report
    lines: list[str] = []
    for name, hits in section_hits.items():
        pct = hits / total * 100
        status = "OK" if pct >= 70 else "WEAK" if pct >= 30 else "MISSING"
        lines.append(f"  {name}: {status} ({hits}/{total} captions address this)")

    if caption_sections:
        lines.append("Labeled section markers in captions:")
        for sec_name, hits in marker_hits.items():
            pct = hits / total * 100
            status = "OK" if pct >= 70 else "WEAK" if pct >= 30 else "MISSING"
            lines.append(f"  [{sec_name}]: {status} ({hits}/{total} captions contain this label)")

        ok_ordering = sum(1 for r in ordering_results if r == "OK")
        checked = total - sum(1 for r in ordering_results if r == "SKIP")
        if checked > 0:
            pct = ok_ordering / checked * 100
            status = "OK" if pct >= 70 else "WEAK" if pct >= 30 else "MISORDERED"
            lines.append(f"Section ordering: {status} ({ok_ordering}/{checked} captions in correct order)")

        ok_lengths = sum(1 for r in length_results if r == "OK")
        imbalanced = [r for r in length_results if r.startswith("IMBALANCED")]
        if imbalanced:
            lines.append(f"Section balance: IMBALANCED ({len(imbalanced)}/{total} captions) — {imbalanced[0]}")
        elif ok_lengths > 0:
            lines.append(f"Section balance: OK ({ok_lengths}/{total} captions well-balanced)")

    if has_subject:
        ok_subject = sum(1 for r in subject_results if r == "OK")
        if ok_subject == total:
            lines.append(f"Subject specificity: OK ({ok_subject}/{total} captions)")
        else:
            first_issue = next((r for r in subject_results if r != "OK"), "MISSING")
            status = "WEAK" if ok_subject > 0 else "MISSING"
            lines.append(f"Subject specificity: {status} ({ok_subject}/{total} captions) — {first_issue}")

    prose = "Caption compliance with meta-prompt sections:\n" + "\n".join(lines)
    return stats, prose


def compute_caption_compliance_stats(
    section_names: list[str],
    captions: list[Caption],
    caption_sections: list[str] | None = None,
) -> CaptionComplianceStats:
    """Structured caption-compliance rates — thin wrapper over :func:`compute_caption_compliance`."""
    return compute_caption_compliance(section_names, captions, caption_sections)[0]


def check_caption_compliance(
    section_names: list[str],
    captions: list[Caption],
    caption_sections: list[str] | None = None,
) -> str:
    """Human-readable compliance summary — thin wrapper over :func:`compute_caption_compliance`."""
    return compute_caption_compliance(section_names, captions, caption_sections)[1]


def compute_style_consistency(captions: list[Caption]) -> float:
    """Measure how consistent the [Art Style] blocks are across captions.

    Extracts the text between ``[Art Style]`` and the next ``[`` marker from
    each caption, then computes the mean pairwise word-overlap (Jaccard)
    similarity.  Returns a float in [0, 1]; 1.0 means all style blocks are
    identical.  Returns 0.0 if fewer than 2 captions contain the label.
    """
    if len(captions) < 2:
        return 0.0

    blocks: list[set[str]] = []
    for cap in captions:
        art_style_text = parse_labeled_sections(cap.text).get("Art Style", "")
        words = set(art_style_text.lower().split())
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


def aggregate(scores: list[MetricScores], *, completion_rate: float = 1.0) -> AggregatedMetrics:
    """Compute mean and std for each metric across a list of per-image scores.

    Fallback scores (``is_fallback=True``) are excluded from aggregation to
    prevent zero-score sentinels from contaminating means and inflating std.
    """
    # Filter out fallback sentinels for aggregation (they exist only for index alignment)
    genuine = [s for s in scores if not s.is_fallback]
    n = len(genuine)
    if n == 0:
        return AggregatedMetrics(
            dreamsim_similarity_mean=0.0,
            dreamsim_similarity_std=0.0,
            hps_score_mean=0.0,
            hps_score_std=0.0,
            aesthetics_score_mean=0.0,
            aesthetics_score_std=0.0,
            completion_rate=completion_rate,
        )

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    def _std(vals: list[float]) -> float:
        m = _mean(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    dreamsim_vals = [s.dreamsim_similarity for s in genuine]
    hps = [s.hps_score for s in genuine]
    aes = [s.aesthetics_score for s in genuine]
    color_vals = [s.color_histogram for s in genuine]
    ssim_vals = [s.ssim for s in genuine]
    v_style = [s.vision_style for s in genuine]
    v_subject = [s.vision_subject for s in genuine]
    v_composition = [s.vision_composition for s in genuine]

    return AggregatedMetrics(
        dreamsim_similarity_mean=_mean(dreamsim_vals),
        dreamsim_similarity_std=_std(dreamsim_vals),
        hps_score_mean=_mean(hps),
        hps_score_std=_std(hps),
        aesthetics_score_mean=_mean(aes),
        aesthetics_score_std=_std(aes),
        color_histogram_mean=_mean(color_vals),
        color_histogram_std=_std(color_vals),
        ssim_mean=_mean(ssim_vals),
        ssim_std=_std(ssim_vals),
        vision_style=_mean(v_style),
        vision_style_std=_std(v_style),
        vision_subject=_mean(v_subject),
        vision_subject_std=_std(v_subject),
        vision_composition=_mean(v_composition),
        vision_composition_std=_std(v_composition),
        completion_rate=completion_rate,
    )


_ZERO_SCORES = MetricScores(
    dreamsim_similarity=0.0,
    hps_score=0.0,
    aesthetics_score=0.0,
    color_histogram=0.0,
    ssim=0.0,
    vision_style=0.0,
    vision_subject=0.0,
    vision_composition=0.0,
    is_fallback=True,
)


def _compute_single_sync(
    registry: ModelRegistry,
    generated: Image.Image,
    reference: Image.Image,
    prompt: str,
) -> MetricScores:
    """Compute all per-image metrics for one generated image (synchronous)."""
    return MetricScores(
        dreamsim_similarity=registry.compute_dreamsim(generated, reference),
        hps_score=registry.compute_hps(generated, prompt),
        aesthetics_score=registry.compute_aesthetics(generated),
        color_histogram=registry.compute_color_histogram(generated, reference),
        ssim=registry.compute_ssim(generated, reference),
    )


async def evaluate_images(
    generated_paths: list[Path],
    reference_paths: list[Path],
    captions: list[str],
    *,
    registry: ModelRegistry,
    semaphore: asyncio.Semaphore,
) -> tuple[list[MetricScores], int]:
    """Evaluate each generated image against its paired reference.

    Each generated image is compared against ONLY its corresponding reference
    (not the mean of all references).  HPS is scored against the per-image
    caption (the actual generation prompt), not the meta-prompt.

    Returns (per-image scores, n_eval_failed).
    Failed evaluations get zero-score sentinels to preserve index alignment.
    """

    async def _eval_one(gen_path: Path, ref_path: Path, caption: str) -> MetricScores | None:
        async with semaphore:
            gen_image: Image.Image | None = None
            ref_image: Image.Image | None = None
            try:
                gen_image = Image.open(gen_path).convert("RGB")
                ref_image = Image.open(ref_path).convert("RGB")
                scores = await asyncio.to_thread(_compute_single_sync, registry, gen_image, ref_image, caption)
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

    n_eval_failed = sum(1 for r in results if r is None)
    if n_eval_failed:
        logger.warning("evaluate_images: %d/%d evaluations failed, using zero scores", n_eval_failed, len(results))
    scores = [r if r is not None else _ZERO_SCORES for r in results]

    return scores, n_eval_failed
