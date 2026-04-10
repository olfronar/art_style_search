"""Unit tests for CircuitBreaker and async_retry."""

from __future__ import annotations

import asyncio
import time

import pytest

from art_style_search.utils import CircuitBreaker, async_retry

# -- CircuitBreaker -----------------------------------------------------------


class TestCircuitBreaker:
    def test_initially_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, cooldown=10.0)
        # Should not block
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cb.wait_if_open())

    def test_trips_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, cooldown=0.1)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()  # Should trip
        assert cb._open_until > time.monotonic()

    def test_success_resets_counter(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, cooldown=10.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Should reset
        cb.record_failure()
        # Only 1 failure after reset, should not trip
        assert cb._open_until == 0.0

    def test_does_not_trip_below_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=5, cooldown=10.0)
        for _ in range(4):
            cb.record_failure()
        assert cb._open_until == 0.0

    @pytest.mark.asyncio
    async def test_wait_if_open_sleeps_then_resets(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown=0.05)
        cb.record_failure()
        start = time.monotonic()
        await cb.wait_if_open()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04  # Allow small tolerance
        assert cb._consecutive_failures == 0  # Reset after cooldown


# -- async_retry --------------------------------------------------------------


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self) -> None:
        call_count = 0

        async def _ok() -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        result = await async_retry(_ok, max_retries=3, base_delay=0.01)
        assert result == "done"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self) -> None:
        call_count = 0

        async def _fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "ok"

        result = await async_retry(_fail_twice, max_retries=5, base_delay=0.01)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhaustion_raises_runtime_error(self) -> None:
        async def _always_fail() -> str:
            raise ValueError("permanent")

        with pytest.raises(RuntimeError, match="failed after 2 retries"):
            await async_retry(_always_fail, max_retries=2, base_delay=0.01, label="Test op")

    @pytest.mark.asyncio
    async def test_with_circuit_breaker(self) -> None:
        cb = CircuitBreaker(failure_threshold=100, cooldown=0.01)
        call_count = 0

        async def _fail_then_ok() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient")
            return "done"

        result = await async_retry(_fail_then_ok, max_retries=3, base_delay=0.01, circuit_breaker=cb)
        assert result == "done"
        assert call_count == 2
