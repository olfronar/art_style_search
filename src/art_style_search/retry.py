"""Retry helpers and circuit breaker used for external API calls."""

from __future__ import annotations

import asyncio
import logging
import random as _rng
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RATE_LIMIT_DELAY = 30.0

# Per-request Gemini timeout is scaled with the output budget: small vision verdicts
# finish in seconds while 32k-token captions regularly need minutes. Linear model
# derived from observed ~50 tokens/sec throughput with a 60s floor for short calls.
_GEMINI_TIMEOUT_FLOOR_S = 60.0
_GEMINI_TIMEOUT_BASE_S = 30.0
_GEMINI_TIMEOUT_PER_TOKEN_S = 0.02


def gemini_timeout_s(max_output_tokens: int) -> float:
    """Scale a per-request Gemini timeout to the output-token budget."""
    return max(_GEMINI_TIMEOUT_FLOOR_S, _GEMINI_TIMEOUT_BASE_S + max_output_tokens * _GEMINI_TIMEOUT_PER_TOKEN_S)


def log_api_call(
    *,
    provider: str,
    model: str,
    stage: str,
    duration_s: float,
    max_tokens: int | None = None,
    effort: str | None = None,
    thinking_level: str | None = None,
    usage: dict[str, int] | None = None,
    status: str = "ok",
) -> None:
    """Emit one structured line summarizing a single provider API call.

    Format: ``API_CALL provider=<p> model=<m> stage=<s> duration_s=<d> [max_tokens=<n>]
    [effort=<e>] [thinking_level=<t>] [usage=...] status=<st>``

    Log grep-friendly: ``grep API_CALL runs/*/logs/*.log``.
    """
    parts: list[str] = [
        f"provider={provider}",
        f"model={model}",
        f"stage={stage}",
        f"duration_s={duration_s:.2f}",
    ]
    if max_tokens is not None:
        parts.append(f"max_tokens={max_tokens}")
    if effort is not None:
        parts.append(f"effort={effort}")
    if thinking_level is not None:
        parts.append(f"thinking_level={thinking_level}")
    if usage:
        parts.append("usage=" + ",".join(f"{k}:{v}" for k, v in sorted(usage.items())))
    parts.append(f"status={status}")
    logger.info("API_CALL " + " ".join(parts))


def _is_rate_limit(exc: Exception) -> bool:
    """Detect Gemini 429 / ResourceExhausted errors."""
    # Check by class name (works without importing google.api_core)
    if "resourceexhausted" in type(exc).__name__.lower():
        return True
    # Check explicit class if google.api_core is available
    try:
        from google.api_core.exceptions import ResourceExhausted  # type: ignore[attr-defined]

        if isinstance(exc, ResourceExhausted):
            return True
    except ImportError:
        pass
    return "429" in str(exc)


class CircuitBreaker:
    """Pauses all calls after consecutive failures, then auto-resets after cooldown."""

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


# Per-surface circuit breakers so a failure storm in one Gemini surface
# (e.g. generation) doesn't block unrelated surfaces (e.g. captioning).
caption_circuit_breaker = CircuitBreaker(failure_threshold=15, cooldown=60.0)
generation_circuit_breaker = CircuitBreaker(failure_threshold=15, cooldown=60.0)
vision_circuit_breaker = CircuitBreaker(failure_threshold=15, cooldown=60.0)


def _is_truncation(exc: Exception) -> bool:
    """Detect output-truncation errors raised by the reasoning_client.

    Named rather than imported to avoid a retry→reasoning_client circular dependency.
    """
    return type(exc).__name__ == "TruncationError"


async def async_retry(
    coro_fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 5,
    base_delay: float = 5.0,
    label: str = "",
    circuit_breaker: CircuitBreaker | None = None,
) -> T:
    """Generic async retry with jittered exponential backoff and circuit breaker.

    ``TruncationError`` is re-raised immediately — retrying with the same budget
    would deterministically re-truncate, wasting retries.
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
            if _is_truncation(exc):
                logger.warning(
                    "%s truncated (%s); not retrying",
                    label or "Retry",
                    exc,
                )
                raise
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
