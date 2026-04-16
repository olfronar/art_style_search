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


async def async_retry(
    coro_fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 5,
    base_delay: float = 5.0,
    label: str = "",
    circuit_breaker: CircuitBreaker | None = None,
) -> T:
    """Generic async retry with jittered exponential backoff and circuit breaker."""
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
