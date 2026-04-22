"""Caption reference images via Gemini Pro (per-iteration) or Anthropic (bootstrap) with disk caching."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, cast

from google import genai  # type: ignore[attr-defined]
from google.genai import types as genai_types  # type: ignore[attr-defined]

from art_style_search.caption_sections import parse_labeled_sections
from art_style_search.evaluate import compute_canon_fidelity
from art_style_search.reasoning_client import ANTHROPIC_EFFORT_FROM_THINKING, ReasoningClient, TruncationError
from art_style_search.types import Caption
from art_style_search.utils import (
    async_retry,
    caption_circuit_breaker,
    gemini_timeout_s,
    image_to_gemini_part,
    log_api_call,
)

_ANTHROPIC_BOOTSTRAP_CONCURRENCY = 5
# Sized to cover the caption length ceiling (~4000 words * ~1.5 tokens/word for `[Subject]` + headroom);
# `_gemini_analyze` uses 8k tokens because it produces a 300-600 word summary, not a caption.
_ANTHROPIC_BOOTSTRAP_MAX_TOKENS = 32000

logger = logging.getLogger(__name__)

_ANCHOR_SECTION_MIN_WORDS: dict[str, int] = {
    "Art Style": 100,
    "Subject": 80,
}

# Upper bounds on the per-section word count of the captioner output. Instruction
# targets are `[Art Style]` 400-800 and `[Subject]` 800-2000; the ceilings here
# are generous (2x the upper target) to reject only pathological runaway blocks
# that dilute downstream scoring signal.
_ANCHOR_SECTION_MAX_WORDS: dict[str, int] = {
    "Art Style": 1600,
    "Subject": 4000,
}

CAPTION_SYSTEM = (
    "You are an expert art analyst producing captions that function as both faithful descriptions and "
    "ready-to-use text-to-image prompts. Your captions will be fed directly to an image generator, so every "
    "detail you write must be specific enough to reproduce the original image visually.\n\n"
    "Quality standards:\n"
    "- Lead with concrete subject identity and scene-defining information before softer stylistic interpretation.\n"
    "- Prefer direct positive phrasing and generation-ready wording over hedged narration.\n"
    "- When dense specifics help, use compact comma-delimited visual tokens; one observation per sentence; "
    "tie-break shorter when the signal is the same. Prefer signal over filler — epistemic hedges are signal.\n"
    "- Use precise color names (e.g. 'burnt sienna', 'cerulean blue'), never vague terms like 'warm colors' or 'dark tones'. "
    "If you cannot name a specific color, describe it by hue, saturation, and value.\n"
    "- Quantify when possible: 'occupies roughly the left third' not 'on the left side'; '3-4 visible characters' not 'several characters'.\n"
    "- Never speculate about artist intent or historical context. Describe only what is visible.\n\n"
    "Medium identification — describe the medium in plain, observable vocabulary that matches what you actually "
    "see on the surface. Pick words that fit the specific grain, edge behavior, shading response, and texture "
    "visible in this image. Use no menu, no checklist, and no letter bucket. The observation drives the vocabulary, "
    "not the other way around. Don't mix vocabulary from incompatible media in the same image — technique words "
    "must be self-consistent with the surface you just named.\n\n"
    "[Art Style] is the STYLE CANON — shared style DNA, copied verbatim into every caption.\n"
    "- The meta-prompt's `style_foundation` section already contains the canon: concrete assertive rules about this "
    "specific art style (medium, shading stack, color principle, surfaces, invariants). Your job is to **reproduce "
    "that canon content verbatim (or near-verbatim) as the body of [Art Style]** — do not paraphrase it, do not "
    "rewrite it in your own voice, do not invent new rules, do not drop slots. Preserve the `How to Draw:` / "
    "`Shading & Light:` / `Color Principle:` / `Surface & Texture:` / `Style Invariants:` markers if the canon has them.\n"
    "- Your per-image writing freedom lives entirely in the observation blocks ([Subject], [Color Palette], "
    "[Composition], [Lighting & Atmosphere], and any other per-image sections the meta-prompt defines).\n"
    "- Observations-vs-rules: specific body parts, named objects, proper nouns, actual colors, pose details NEVER "
    "appear in [Art Style]. A sentence inside [Art Style] is well-formed only if it would still be true of a "
    "DIFFERENT image in the same style. Character proportions (heads-tall, archetype, silhouette shape) "
    "belong in the [Subject] Proportions sub-block, not here.\n"
    "- Anti-name: phrases like '3D CGI of X', 'cel-shaded anime', '{Artist}-style', 'watercolor illustration' are "
    "forbidden inside [Art Style]. Describe the technique from observable surface cues, not from genre labels.\n"
    "- If the meta-prompt's `style_foundation` does not yet cover a slot, leave that slot's canonical marker in "
    "place and keep the section compact — do not fabricate rules that are not in the canon.\n"
    "- [Art Style] length target: the same 400-800 words that the canon itself targets.\n\n"
    "Output-format discipline:\n"
    "- Write section labels as PLAIN TEXT: `[Art Style]`, `[Subject]`. Never wrap in markdown bolding, backticks, or angle brackets.\n"
    "- Emit labeled blocks in the exact order the meta-prompt's ## Caption Sections block specifies. "
    "Target the word count in ## Caption Length Target. Let [Subject] carry the per-image weight; "
    "keep [Art Style] as the verbatim style canon from `style_foundation`.\n\n"
    "Forbidden terms (never use — no exceptions):\n"
    "  cartoon, cartoonish, stylised, stylized (as bare adjective), beautiful, epic, whimsical, charming, cinematic."
)

CAPTION_PROMPT = (
    "Describe this image in comprehensive detail for someone who cannot see it. "
    "These descriptions will be used to understand and reproduce the art style.\n\n"
    "## Output format\n"
    "Use these labeled sections in this exact order. Keep the labels exactly as written.\n\n"
    "[Art Style]: Shared style DNA — RULES that apply to every image in this style, never per-image observations. "
    "Describe HOW the style is rendered, never NAME the genre. Cover 5 slots in 2-4 sentences each in your own voice: "
    "(1) How to Draw (medium identification in plain observable vocabulary + construction + line policy), "
    "(2) Shading & Light (shading-layer stack + edge softness + key-fill-rim direction), "
    "(3) Color Principle (palette family in generic terms + value structure + saturation policy), "
    "(4) Surface & Texture (grain + class-appropriate material vocabulary), "
    "(5) Style Invariants (3-5 MUST/NEVER rules that every image in this style obeys). "
    "No specific body parts, named objects, or actual colors here — those belong in the sections below. "
    "No style labels like '3D CGI of X', 'cel-shaded anime', or '{Artist}-style'.\n"
    "[Subject] (MOST IMPORTANT): What is depicted — identity, species, poses, expressions, "
    "clothing or equipment, relationships between figures, props, distinguishing features, character proportions "
    "(heads-tall + archetype).\n"
    "[Color Palette]: Dominant palette (name specific colors like 'burnt sienna', not just 'brown'), "
    "color relationships, saturation, temperature, gradients.\n"
    "[Composition]: Layout, focal points, balance, use of space, perspective, framing.\n"
    "[Lighting & Atmosphere]: Light direction, shadow treatment, emotional tone, sense of time or place.\n\n"
    "## Constraints\n"
    "- Be precise. Use art terminology and specific color names where they belong.\n"
    "- Target 1500-4000 words total.\n"
    "- Do not speculate about the artist's intent; describe only what is visible.\n"
    "- [Subject] is the longest section, 800-2000 words when the image supports it.\n"
    "- [Art Style] holds generic rules only, 400-800 words.\n"
    "- Ancillary sections usually land in the 150-400 word range.\n"
    "- Keep the labels exactly as written above."
)

_CAPTION_TARGET_RE = re.compile(r"target length:\s*approximately\s*(\d+)\s*words", re.IGNORECASE)
_CAPTION_TARGET_RANGE_RE = re.compile(r"target\s+(\d+)\s*-\s*(\d+)\s*words", re.IGNORECASE)
_AVG_WORD_CHARS = 4
_FALLBACK_MIN_CAPTION_CHARS = 600
_CAPTIONER_MAX_OUTPUT_TOKENS = 32000
_CANON_FIDELITY_RETRY_THRESHOLD = 0.7


def _caption_length_target_from_prompt(prompt: str) -> int:
    match = _CAPTION_TARGET_RE.search(prompt)
    if match:
        return int(match.group(1))
    range_match = _CAPTION_TARGET_RANGE_RE.search(prompt)
    if range_match:
        return int(range_match.group(1))
    return 0


def _minimum_caption_chars(prompt: str) -> int:
    target = _caption_length_target_from_prompt(prompt)
    if target > 0:
        return max(_FALLBACK_MIN_CAPTION_CHARS, int(target * 0.5 * _AVG_WORD_CHARS))
    return _FALLBACK_MIN_CAPTION_CHARS


async def caption_single(
    image_path: Path,
    *,
    prompt: str,
    model: str,
    client: genai.Client,
    cache_dir: Path | None,
    semaphore: asyncio.Semaphore,
    cache_key: str = "",
    thinking_level: str = "MINIMAL",
    style_canon: str = "",
) -> Caption:
    """Caption a single image, optionally using disk cache.

    When *cache_dir* is provided and *cache_key* matches the stored key,
    the cached result is returned.  The cache_key should change whenever
    the prompt changes (e.g. hash or iteration number) to invalidate stale
    entries.

    When *style_canon* is non-empty and the first attempt scores below
    ``_CANON_FIDELITY_RETRY_THRESHOLD`` against it, reissue once with a
    verbatim-paste suffix. Caller pre-computes the canon once per experiment
    to avoid a regex over the meta-prompt per image.
    """
    current_mtime: float | None = None
    if cache_dir is not None:
        current_mtime = image_path.stat().st_mtime
        cached = _read_caption_cache(cache_dir, image_path, cache_key=cache_key, mtime=current_mtime)
        if cached is not None:
            return cached

    logger.info("Captioning %s via %s", image_path.name, model)

    # Gemini 3.1 Pro rejects thinking_level="MINIMAL" (only LOW/MEDIUM/HIGH are valid), so
    # for MINIMAL we leave thinking_config unset and let the model pick its own default depth.
    config_kwargs: dict[str, Any] = {
        "system_instruction": CAPTION_SYSTEM,
        "max_output_tokens": _CAPTIONER_MAX_OUTPUT_TOKENS,
    }
    if thinking_level != "MINIMAL":
        config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_level=cast("Any", thinking_level))

    min_caption_length = _minimum_caption_chars(prompt)

    async def _call_captioner(effective_prompt: str, *, validate_sections: bool) -> str:
        async with semaphore:
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model,
                    contents=[image_to_gemini_part(image_path), effective_prompt],
                    config=genai_types.GenerateContentConfig(**config_kwargs),
                ),
                timeout=gemini_timeout_s(_CAPTIONER_MAX_OUTPUT_TOKENS),
            )
        candidates = getattr(resp, "candidates", None) or []
        if candidates and str(getattr(candidates[0], "finish_reason", "")).endswith("MAX_TOKENS"):
            raise TruncationError(
                provider="gemini",
                stage="caption",
                max_tokens=_CAPTIONER_MAX_OUTPUT_TOKENS,
            )
        text = resp.text or ""
        # Raising inside the retry loop lets async_retry re-issue on transient captioner misbehavior
        # (truncated output, dropped sections). Bootstrap retries skip section validation — the
        # retry's stricter prompt sometimes trims observation blocks that would otherwise pass.
        if validate_sections:
            _validate_caption_text(text, image_name=image_path.name, min_length=min_caption_length)
        elif not text or len(text.strip()) < min_caption_length:
            msg = (
                f"Captioning {image_path.name} produced empty or too-short caption "
                f"({len(text.strip())} chars, min {min_caption_length})"
            )
            raise RuntimeError(msg)
        return text

    started = time.monotonic()
    try:
        caption_text: str = await async_retry(
            lambda: _call_captioner(prompt, validate_sections=True),
            label=f"Caption {image_path.name}",
            circuit_breaker=caption_circuit_breaker,
        )
    except Exception:
        log_api_call(
            provider="gemini",
            model=model,
            stage="caption",
            duration_s=time.monotonic() - started,
            max_tokens=_CAPTIONER_MAX_OUTPUT_TOKENS,
            thinking_level=thinking_level,
            status="error",
        )
        raise
    log_api_call(
        provider="gemini",
        model=model,
        stage="caption",
        duration_s=time.monotonic() - started,
        max_tokens=_CAPTIONER_MAX_OUTPUT_TOKENS,
        thinking_level=thinking_level,
    )

    if style_canon:
        fidelity = compute_canon_fidelity(caption_text, style_canon)
        if fidelity < _CANON_FIDELITY_RETRY_THRESHOLD:
            logger.info(
                "Caption %s paraphrased canon (fidelity %.2f < %.2f) — reissuing with verbatim-paste suffix",
                image_path.name,
                fidelity,
                _CANON_FIDELITY_RETRY_THRESHOLD,
            )
            retried_prompt = prompt + (
                "\n\n## Canon-copy enforcement (your previous attempt paraphrased the canon)\n"
                "Your [Art Style] block must paste the following canon text verbatim (or nearly so — "
                "whitespace and minor edits only). Do NOT paraphrase it. Write original content only in "
                "the observation blocks ([Subject], [Color Palette], [Composition], [Lighting & Atmosphere]).\n\n"
                "Canon to paste into [Art Style]:\n"
                f"{style_canon}\n"
            )
            retry_started = time.monotonic()
            try:
                caption_text = await async_retry(
                    lambda: _call_captioner(retried_prompt, validate_sections=False),
                    label=f"Caption {image_path.name} (canon-retry)",
                    circuit_breaker=caption_circuit_breaker,
                )
                log_api_call(
                    provider="gemini",
                    model=model,
                    stage="caption_canon_retry",
                    duration_s=time.monotonic() - retry_started,
                    max_tokens=_CAPTIONER_MAX_OUTPUT_TOKENS,
                    thinking_level=thinking_level,
                )
            except Exception as exc:
                logger.warning(
                    "Canon-retry failed for %s (keeping original paraphrased caption): %s",
                    image_path.name,
                    exc,
                )

    if cache_dir is not None and current_mtime is not None:
        _write_caption_cache(cache_dir, image_path, text=caption_text, cache_key=cache_key, mtime=current_mtime)

    return Caption(image_path=image_path, text=caption_text)


async def caption_references(
    reference_paths: list[Path],
    *,
    model: str,
    client: genai.Client,
    cache_dir: Path,
    semaphore: asyncio.Semaphore,
    prompt: str | None = None,
    cache_key: str = "",
    thinking_level: str = "MINIMAL",
    style_canon: str = "",
) -> list[Caption]:
    """Caption all reference images concurrently with disk caching.

    When *prompt* is None, uses the default CAPTION_PROMPT.
    The *cache_key* invalidates cached entries when the prompt changes.
    """
    effective_prompt = prompt or CAPTION_PROMPT
    tasks = [
        caption_single(
            path,
            prompt=effective_prompt,
            model=model,
            client=client,
            cache_dir=cache_dir,
            semaphore=semaphore,
            cache_key=cache_key,
            thinking_level=thinking_level,
            style_canon=style_canon,
        )
        for path in reference_paths
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    captions: list[Caption] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning("Caption %d (%s) failed: %s", i, reference_paths[i].name, result)
        else:
            captions.append(result)
    return captions


def _validate_caption_text(text: str, *, image_name: str, min_length: int) -> None:
    """Raise RuntimeError when a caption response is empty, too short, or has out-of-bounds anchor sections."""
    if not text or len(text.strip()) < min_length:
        msg = (
            f"Captioning {image_name} produced empty or too-short caption ({len(text.strip())} chars, min {min_length})"
        )
        raise RuntimeError(msg)
    parsed = parse_labeled_sections(text)
    if not parsed:
        return
    violations: list[str] = []
    for section_name, floor in _ANCHOR_SECTION_MIN_WORDS.items():
        if section_name in parsed and len(parsed[section_name].split()) < floor:
            violations.append(f"[{section_name}]={len(parsed[section_name].split())}w (min {floor})")
    for section_name, ceiling in _ANCHOR_SECTION_MAX_WORDS.items():
        if section_name in parsed and len(parsed[section_name].split()) > ceiling:
            violations.append(f"[{section_name}]={len(parsed[section_name].split())}w (max {ceiling})")
    if violations:
        msg = f"Captioning {image_name} produced out-of-bounds anchor sections: " + "; ".join(violations)
        raise RuntimeError(msg)


def _read_caption_cache(cache_dir: Path, image_path: Path, *, cache_key: str, mtime: float) -> Caption | None:
    """Return a cached caption if it matches ``(mtime, cache_key)``, else None."""
    try:
        cached = json.loads((cache_dir / f"{image_path.stem}.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None
    if cached.get("mtime") != mtime or cached.get("cache_key", "") != cache_key:
        return None
    logger.debug("Cache hit for %s", image_path.name)
    return Caption(image_path=Path(cached["image_path"]), text=cached["text"])


def _write_caption_cache(
    cache_dir: Path,
    image_path: Path,
    *,
    text: str,
    cache_key: str,
    mtime: float,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"image_path": str(image_path), "text": text, "mtime": mtime, "cache_key": cache_key}
    (cache_dir / f"{image_path.stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.debug("Cached caption for %s", image_path.name)


async def caption_bootstrap(
    reference_paths: list[Path],
    *,
    client: ReasoningClient,
    model: str,
    cache_dir: Path,
    cache_key: str = "",
    prompt: str | None = None,
    system: str | None = None,
    thinking_level: str = "MINIMAL",
    concurrency: int = _ANTHROPIC_BOOTSTRAP_CONCURRENCY,
) -> list[Caption]:
    """Caption references via a multimodal reasoning client (zero-step only).

    Uses the same disk-cache layout as :func:`caption_references` — per-image JSON keyed
    by (mtime, cache_key). Cache_key should include the provider tag (e.g. ``"initial-claude"``)
    so cached entries from a different provider aren't mistakenly reused.
    """
    effective_prompt = prompt or CAPTION_PROMPT
    effective_system = system or CAPTION_SYSTEM
    reasoning_effort = ANTHROPIC_EFFORT_FROM_THINKING.get(thinking_level.upper(), "low")
    min_caption_length = _minimum_caption_chars(effective_prompt)
    semaphore = asyncio.Semaphore(concurrency)

    async def _caption_one(image_path: Path) -> Caption:
        current_mtime = image_path.stat().st_mtime
        cached = _read_caption_cache(cache_dir, image_path, cache_key=cache_key, mtime=current_mtime)
        if cached is not None:
            return cached

        logger.info("Bootstrap-captioning %s via %s (%s)", image_path.name, client.provider, model)

        async def _call() -> str:
            async with semaphore:
                text = await client.call_with_images(
                    model=model,
                    system=effective_system,
                    user=effective_prompt,
                    image_paths=[image_path],
                    max_tokens=_ANTHROPIC_BOOTSTRAP_MAX_TOKENS,
                    reasoning_effort=reasoning_effort,
                    stage="caption_bootstrap",
                )
            _validate_caption_text(text, image_name=image_path.name, min_length=min_caption_length)
            return text

        caption_text = await async_retry(
            _call,
            label=f"Bootstrap caption {image_path.name}",
            circuit_breaker=caption_circuit_breaker,
        )
        _write_caption_cache(cache_dir, image_path, text=caption_text, cache_key=cache_key, mtime=current_mtime)
        return Caption(image_path=image_path, text=caption_text)

    results = await asyncio.gather(*[_caption_one(p) for p in reference_paths], return_exceptions=True)
    captions: list[Caption] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning("Bootstrap caption %d (%s) failed: %s", i, reference_paths[i].name, result)
        else:
            captions.append(result)
    return captions
