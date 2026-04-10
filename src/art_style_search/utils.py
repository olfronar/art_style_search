"""Shared utilities for Anthropic, Z.AI, OpenAI, and Gemini API interactions."""

from __future__ import annotations

import asyncio
import logging
import random as _rng
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import anthropic
import httpcore
import httpx
from anthropic.types import Message
from google.genai import types as genai_types

if TYPE_CHECKING:
    from art_style_search.types import IterationResult

logger = logging.getLogger(__name__)

_STREAM_MAX_RETRIES = 3
_STREAM_BASE_DELAY = 5.0

MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

IMAGE_EXTENSIONS: frozenset[str] = frozenset(MIME_MAP)

# Canonical hypothesis categories used for classification, diversity enforcement,
# and target-category ranking. Used by scoring.classify_hypothesis,
# types.get_category_names, and loop._should_honor_stop.
CATEGORY_SYNONYMS: dict[str, list[str]] = {
    "color_palette": ["color", "hue", "palette", "saturation", "tone", "gradient", "shade"],
    "composition": ["layout", "framing", "spatial", "arrangement", "perspective", "depth"],
    "technique": ["medium", "brushwork", "brushstroke", "rendering", "stroke", "paint", "watercolor"],
    "mood_atmosphere": ["mood", "atmosphere", "emotion", "feeling", "ambiance", "tone"],
    "lighting": ["light", "shadow", "illumination", "glow", "highlight", "contrast"],
    "texture": ["texture", "surface", "grain", "detail", "pattern"],
    "subject_matter": ["subject", "character", "figure", "object", "scene"],
    "background": ["background", "environment", "setting", "landscape", "sky"],
    "caption_structure": ["section", "label", "order", "ordering", "structure", "format", "length"],
}


def extract_xml_tag(text: str, tag: str) -> str:
    """Extract text content between <tag> and </tag>, stripped. Returns '' if absent."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


T = TypeVar("T")


_RATE_LIMIT_DELAY = 30.0


def _is_rate_limit(exc: Exception) -> bool:
    """Detect Gemini 429 / ResourceExhausted errors."""
    return "resourceexhausted" in type(exc).__name__.lower() or "429" in str(exc)


class CircuitBreaker:
    """Pauses all calls after consecutive failures, then auto-resets after cooldown.

    Use ``record_success``/``record_failure`` around API calls.
    Call ``await wait_if_open()`` before each attempt — it sleeps
    if the breaker is tripped, otherwise returns immediately.
    """

    def __init__(self, failure_threshold: int = 15, cooldown: float = 60.0) -> None:
        self._threshold = failure_threshold
        self._cooldown = cooldown
        self._consecutive_failures = 0
        self._open_until: float = 0.0

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> None:

        self._consecutive_failures += 1
        if self._consecutive_failures >= self._threshold:
            self._open_until = time.monotonic() + self._cooldown
            logger.warning(
                "Circuit breaker tripped after %d consecutive failures — pausing %.0fs",
                self._consecutive_failures,
                self._cooldown,
            )

    async def wait_if_open(self) -> None:

        remaining = self._open_until - time.monotonic()
        if remaining > 0:
            logger.info("Circuit breaker open — waiting %.0fs before retry", remaining)
            await asyncio.sleep(remaining)
            self._consecutive_failures = 0


# Shared circuit breaker for all Gemini API calls
gemini_circuit_breaker = CircuitBreaker(failure_threshold=15, cooldown=60.0)


async def async_retry(
    coro_fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 5,
    base_delay: float = 5.0,
    label: str = "",
    circuit_breaker: CircuitBreaker | None = None,
) -> T:
    """Generic async retry with jittered exponential backoff and circuit breaker.

    Calls `coro_fn()` up to `max_retries` times. Uses longer backoff for
    rate-limit (429) errors. Raises RuntimeError after exhaustion.
    When *circuit_breaker* is provided, waits if breaker is open and records
    successes/failures to trip or reset it.
    """

    cb = circuit_breaker
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        if cb:
            await cb.wait_if_open()
        try:
            result = await coro_fn()
            if cb:
                cb.record_success()
            return result
        except Exception as exc:
            if cb:
                cb.record_failure()
            last_exc = exc
            rate_limited = _is_rate_limit(exc)
            base = _RATE_LIMIT_DELAY if rate_limited else base_delay
            delay = base * (2**attempt) * (0.5 + _rng.random())
            logger.warning(
                "%s %s %d/%d: %s: %s — retrying in %.0fs",
                label or "Retry",
                "rate-limited" if rate_limited else "attempt",
                attempt + 1,
                max_retries,
                type(exc).__name__,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    msg = f"{label or 'Operation'} failed after {max_retries} retries"
    raise RuntimeError(msg) from last_exc


def image_to_gemini_part(path: Path) -> genai_types.Part:
    """Read an image file and return a Gemini Part with correct MIME type."""
    mime_type = MIME_MAP.get(path.suffix.lower(), "image/png")
    return genai_types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def build_ref_gen_pairs(result: IterationResult) -> list[tuple[Path, Path]]:
    """Reconstruct (reference, generated) pairs from an IterationResult.

    Generated image filenames encode the caption index (e.g. ``05.png``
    corresponds to ``iteration_captions[5]``).  We parse the stem to recover
    the mapping.
    """
    caption_by_idx = {i: c.image_path for i, c in enumerate(result.iteration_captions)}
    pairs: list[tuple[Path, Path]] = []
    for gen_path in result.image_paths:
        try:
            idx = int(gen_path.stem)
        except ValueError:
            continue
        ref = caption_by_idx.get(idx)
        if ref is not None:
            pairs.append((ref, gen_path))
    return pairs


def extract_text(response: Message) -> str:
    """Extract text content from a response that may contain thinking blocks."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


async def stream_message(client: anthropic.AsyncAnthropic, **kwargs: object) -> Message:
    """Call messages.create with streaming and return the final Message.

    Retries on transient connection errors (dropped connections, incomplete reads).
    """

    last_exc: Exception | None = None
    for attempt in range(_STREAM_MAX_RETRIES):
        try:
            async with client.messages.stream(**kwargs) as stream:
                return await stream.get_final_message()
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            last_exc = exc
            delay = _STREAM_BASE_DELAY * (2**attempt) * (0.5 + _rng.random())
            logger.warning(
                "Anthropic stream attempt %d/%d failed: %s — retrying in %.0fs",
                attempt + 1,
                _STREAM_MAX_RETRIES,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
        except (httpx.RemoteProtocolError, httpx.ReadError, httpcore.RemoteProtocolError, httpcore.ReadError) as exc:
            last_exc = exc
            delay = _STREAM_BASE_DELAY * (2**attempt) * (0.5 + _rng.random())
            logger.warning(
                "Anthropic stream attempt %d/%d failed: %s: %s — retrying in %.0fs",
                attempt + 1,
                _STREAM_MAX_RETRIES,
                type(exc).__name__,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
        except Exception as exc:
            exc_str = str(exc).lower()
            is_transient = "incomplete chunked read" in exc_str or "peer closed connection" in exc_str
            if is_transient:
                last_exc = exc
                delay = _STREAM_BASE_DELAY * (2**attempt) * (0.5 + _rng.random())
                logger.warning(
                    "Anthropic stream attempt %d/%d failed: %s: %s — retrying in %.0fs",
                    attempt + 1,
                    _STREAM_MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise

    msg = f"Anthropic stream failed after {_STREAM_MAX_RETRIES} retries"
    raise RuntimeError(msg) from last_exc


# ---------------------------------------------------------------------------
# Provider-agnostic reasoning client
# ---------------------------------------------------------------------------


class ReasoningClient:
    """Wraps Anthropic, Z.AI, OpenAI, or a local OpenAI-compatible server behind a unified async interface."""

    def __init__(
        self,
        provider: str,
        *,
        anthropic_api_key: str = "",
        zai_api_key: str = "",
        openai_api_key: str = "",
        base_url: str = "",
    ) -> None:
        self.provider = provider
        if provider == "anthropic":
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=anthropic_api_key,
                timeout=anthropic.Timeout(600.0, connect=30.0),
            )
        elif provider == "zai":
            import httpx
            from zai import ZaiClient

            self._zai = ZaiClient(
                api_key=zai_api_key,
                timeout=httpx.Timeout(300.0, connect=15.0),
            )
        elif provider == "openai":
            import httpx
            from openai import AsyncOpenAI

            self._openai = AsyncOpenAI(
                api_key=openai_api_key,
                timeout=httpx.Timeout(600.0, connect=30.0),
            )
        elif provider == "local":
            import httpx
            from openai import AsyncOpenAI

            self._local = AsyncOpenAI(
                api_key="not-needed",
                base_url=base_url,
                timeout=httpx.Timeout(600.0, connect=30.0),
            )
        else:
            msg = f"Unknown reasoning provider: {provider}"
            raise ValueError(msg)

    async def call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 16000,
    ) -> str:
        """Send a reasoning request and return the text response."""
        if self.provider == "anthropic":
            return await self._call_anthropic(model=model, system=system, user=user, max_tokens=max_tokens)
        if self.provider == "openai":
            return await self._call_openai(model=model, system=system, user=user, max_tokens=max_tokens)
        if self.provider == "local":
            return await self._call_local(model=model, system=system, user=user, max_tokens=max_tokens)
        return await self._call_zai(model=model, system=system, user=user, max_tokens=max_tokens)

    async def _call_anthropic(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        response = await stream_message(
            self._anthropic,
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return extract_text(response)

    async def _call_zai(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        """Call Z.AI synchronous SDK via asyncio.to_thread with retry."""

        def _sync_call() -> str:
            response = self._zai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        async def _call() -> str:
            return await asyncio.to_thread(_sync_call)

        return await async_retry(_call, label="Z.AI call", base_delay=_STREAM_BASE_DELAY)

    async def _call_openai(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        """Call OpenAI Responses API with medium reasoning effort."""

        async def _call() -> str:
            response = await self._openai.responses.create(
                model=model,
                instructions=system,
                input=user,
                reasoning={"effort": "medium"},
                max_output_tokens=max_tokens,
            )
            return response.output_text

        return await async_retry(_call, label="OpenAI call", base_delay=_STREAM_BASE_DELAY)

    async def _call_local(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        """Call a local OpenAI-compatible server (vLLM, SGLang, Ollama) via chat completions."""

        async def _call() -> str:
            response = await self._local.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        return await async_retry(_call, label="Local model call", base_delay=2.0)
