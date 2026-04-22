"""Unit tests for bootstrap caption prompt and validation behavior."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import pytest

from art_style_search.caption import (
    CAPTION_PROMPT,
    CAPTION_SYSTEM,
    CAPTION_SYSTEM_BOOTSTRAP,
    caption_bootstrap,
    caption_references,
    caption_single,
)

_T = TypeVar("_T")


@pytest.fixture(autouse=True)
def _bypass_retry_for_deterministic_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Captioning validation now runs inside ``async_retry`` so real runs can recover from
    transient truncation. Tests feed a mock that returns the same bad caption every time,
    so we bypass the retry loop here — otherwise each test would stall for ~155s on backoff.
    """

    async def _passthrough(fn: Callable[[], Awaitable[_T]], **_kwargs: object) -> _T:
        return await fn()

    monkeypatch.setattr("art_style_search.caption.async_retry", _passthrough)


class TestBootstrapCaptionPrompt:
    def test_uses_bracketed_art_style_and_subject_sections(self) -> None:
        assert "[Art Style]" in CAPTION_PROMPT
        assert "[Subject]" in CAPTION_PROMPT
        assert "**Subjects**" not in CAPTION_PROMPT
        assert "Target 1500-4000 words total" in CAPTION_PROMPT
        assert "[Subject] is the longest section, 800-2000 words" in CAPTION_PROMPT
        assert "[Art Style] holds generic rules only, 400-800 words" in CAPTION_PROMPT
        # Post-distillation: Technique + Textures dropped from default caption_sections
        assert "[Technique]:" not in CAPTION_PROMPT
        assert "[Textures]:" not in CAPTION_PROMPT


class TestCaptionSingle:
    @pytest.mark.asyncio
    async def test_rejects_caption_shorter_than_bootstrap_floor(self, tmp_path: Path) -> None:
        image_path = tmp_path / "ref.png"
        image_path.write_bytes(b"fake-image")

        class FakeModels:
            async def generate_content(self, **kwargs):
                return SimpleNamespace(text="[Art Style] short [Subject] too short")

        class FakeClient:
            aio = SimpleNamespace(models=FakeModels())

        with pytest.raises(RuntimeError, match="too-short caption"):
            await caption_single(
                image_path,
                prompt=CAPTION_PROMPT,
                model="fake-model",
                client=FakeClient(),  # type: ignore[arg-type]
                cache_dir=None,
                semaphore=asyncio.Semaphore(1),
            )

    @pytest.mark.asyncio
    async def test_bootstrap_prompt_enforces_scaled_floor(self, tmp_path: Path) -> None:
        """CAPTION_PROMPT targets 2000-6000 words, so a ~700-char caption must be rejected."""
        image_path = tmp_path / "ref.png"
        image_path.write_bytes(b"fake-image")

        class FakeModels:
            async def generate_content(self, **kwargs):
                return SimpleNamespace(text="x" * 700)

        class FakeClient:
            aio = SimpleNamespace(models=FakeModels())

        with pytest.raises(RuntimeError, match="too-short caption"):
            await caption_single(
                image_path,
                prompt=CAPTION_PROMPT,
                model="fake-model",
                client=FakeClient(),  # type: ignore[arg-type]
                cache_dir=None,
                semaphore=asyncio.Semaphore(1),
            )

    @pytest.mark.asyncio
    async def test_rejects_caption_shorter_than_scaled_target_floor(self, tmp_path: Path) -> None:
        image_path = tmp_path / "ref.png"
        image_path.write_bytes(b"fake-image")

        class FakeModels:
            async def generate_content(self, **kwargs):
                return SimpleNamespace(text="x" * 700)

        class FakeClient:
            aio = SimpleNamespace(models=FakeModels())

        with pytest.raises(RuntimeError, match="too-short caption"):
            await caption_single(
                image_path,
                prompt="Target length: approximately 4000 words.",
                model="fake-model",
                client=FakeClient(),  # type: ignore[arg-type]
                cache_dir=None,
                semaphore=asyncio.Semaphore(1),
            )

    @pytest.mark.asyncio
    async def test_sets_explicit_max_output_tokens_for_long_captions(self, tmp_path: Path) -> None:
        image_path = tmp_path / "ref.png"
        image_path.write_bytes(b"fake-image")
        captured: dict[str, Any] = {}

        class FakeModels:
            async def generate_content(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(text="[Art Style] " + ("detail " * 1200))

        class FakeClient:
            aio = SimpleNamespace(models=FakeModels())

        await caption_single(
            image_path,
            prompt=CAPTION_PROMPT,
            model="fake-model",
            client=FakeClient(),  # type: ignore[arg-type]
            cache_dir=None,
            semaphore=asyncio.Semaphore(1),
        )

        assert captured["config"].max_output_tokens == 32000

    @pytest.mark.asyncio
    async def test_rejects_runaway_art_style_section(self, tmp_path: Path) -> None:
        """[Art Style] exceeding 1600 words is rejected as runaway content."""
        image_path = tmp_path / "ref.png"
        image_path.write_bytes(b"fake-image")

        long_art_style = "word " * 1700  # 1700 words, exceeds 1600 ceiling
        long_subject = "detail " * 1500  # 1500 words, within bounds
        caption = f"[Art Style] {long_art_style}\n[Subject] {long_subject}"

        class FakeModels:
            async def generate_content(self, **kwargs):
                return SimpleNamespace(text=caption)

        class FakeClient:
            aio = SimpleNamespace(models=FakeModels())

        with pytest.raises(RuntimeError, match=r"\[Art Style\]=1700w \(max 1600\)"):
            await caption_single(
                image_path,
                prompt=CAPTION_PROMPT,
                model="fake-model",
                client=FakeClient(),  # type: ignore[arg-type]
                cache_dir=None,
                semaphore=asyncio.Semaphore(1),
            )


class TestCaptionBootstrap:
    @pytest.mark.asyncio
    async def test_routes_through_reasoning_client_and_persists_cache(self, tmp_path: Path) -> None:
        image_path = tmp_path / "ref.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00fake")
        cache_dir = tmp_path / "captions"

        long_art = "word " * 500
        long_subject = "subject " * 1000
        caption_text = f"[Art Style] {long_art}\n[Subject] {long_subject}"

        captured_calls: list[dict[str, object]] = []

        class FakeClient:
            provider = "anthropic"

            async def call_with_images(self, **kwargs):
                captured_calls.append(kwargs)
                return caption_text

        captions = await caption_bootstrap(
            [image_path],
            client=FakeClient(),  # type: ignore[arg-type]
            model="claude-opus-4-7",
            cache_dir=cache_dir,
            cache_key="initial-claude",
            thinking_level="MINIMAL",
        )

        assert len(captions) == 1
        assert captions[0].text == caption_text
        assert captured_calls[0]["reasoning_effort"] == "low"
        assert captured_calls[0]["stage"] == "caption_bootstrap"
        assert captured_calls[0]["image_paths"] == [image_path]

        cache_file = cache_dir / "ref.json"
        assert cache_file.exists()

        captured_calls.clear()
        cached = await caption_bootstrap(
            [image_path],
            client=FakeClient(),  # type: ignore[arg-type]
            model="claude-opus-4-7",
            cache_dir=cache_dir,
            cache_key="initial-claude",
            thinking_level="MINIMAL",
        )
        assert cached[0].text == caption_text
        assert captured_calls == []  # served from disk cache


class TestCaptionSystemBootstrap:
    """The zero-step system prompt must drop canon-reproduction language but keep every other
    discipline signal (quality, medium, observations-vs-rules, anti-name, output-format,
    forbidden-terms). CAPTION_SYSTEM stays verbatim for per-iteration callers.
    """

    def test_drops_canon_reproduction_clause(self) -> None:
        bootstrap = CAPTION_SYSTEM_BOOTSTRAP.lower()
        assert "reproduce that canon content verbatim" not in bootstrap
        assert "copied verbatim into every caption" not in bootstrap
        assert "pastes verbatim" not in bootstrap
        assert "pasted verbatim" not in bootstrap
        assert "style_foundation" not in bootstrap
        assert "if the canon has them" not in bootstrap
        assert "the meta-prompt's" not in bootstrap
        assert "## caption sections" not in bootstrap
        assert "## caption length target" not in bootstrap

    def test_keeps_quality_and_medium_discipline(self) -> None:
        assert "Quality standards:" in CAPTION_SYSTEM_BOOTSTRAP
        assert "Medium identification" in CAPTION_SYSTEM_BOOTSTRAP
        assert "no menu, no checklist, and no letter bucket" in CAPTION_SYSTEM_BOOTSTRAP
        assert "one observation per sentence" in CAPTION_SYSTEM_BOOTSTRAP

    def test_keeps_observations_vs_rules_and_anti_name(self) -> None:
        assert "Observations-vs-rules" in CAPTION_SYSTEM_BOOTSTRAP
        assert "would still be true of a DIFFERENT image" in CAPTION_SYSTEM_BOOTSTRAP
        assert "'3D CGI of X'" in CAPTION_SYSTEM_BOOTSTRAP
        assert "'cel-shaded anime'" in CAPTION_SYSTEM_BOOTSTRAP

    def test_keeps_output_format_and_forbidden_terms(self) -> None:
        assert "PLAIN TEXT: `[Art Style]`, `[Subject]`" in CAPTION_SYSTEM_BOOTSTRAP
        assert "Forbidden terms" in CAPTION_SYSTEM_BOOTSTRAP
        for term in ("cartoon", "stylised", "beautiful", "epic", "cinematic"):
            assert term in CAPTION_SYSTEM_BOOTSTRAP

    def test_per_iteration_caption_system_still_references_canon(self) -> None:
        # Normalize whitespace so the assertion survives line-wrap reflows.
        system_flat = " ".join(CAPTION_SYSTEM.split())
        assert "reproduce that canon content verbatim" in system_flat
        assert "`How to Draw:`" in CAPTION_SYSTEM
        assert "style_foundation" in CAPTION_SYSTEM
        assert "## Caption Sections" in CAPTION_SYSTEM


class TestCaptionBootstrapDefaultsToBootstrapSystem:
    @pytest.mark.asyncio
    async def test_caption_bootstrap_defaults_to_bootstrap_system(self, tmp_path: Path) -> None:
        """caption_bootstrap() must send CAPTION_SYSTEM_BOOTSTRAP (not CAPTION_SYSTEM) by default."""
        image_path = tmp_path / "ref.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00fake")

        long_art = "word " * 500
        long_subject = "subject " * 1000
        caption_text = f"[Art Style] {long_art}\n[Subject] {long_subject}"

        captured: list[dict[str, object]] = []

        class FakeClient:
            provider = "anthropic"

            async def call_with_images(self, **kwargs):
                captured.append(kwargs)
                return caption_text

        await caption_bootstrap(
            [image_path],
            client=FakeClient(),  # type: ignore[arg-type]
            model="claude-opus-4-7",
            cache_dir=tmp_path / "cache",
            cache_key="initial-claude",
            thinking_level="MINIMAL",
        )

        system_sent = captured[0]["system"]
        assert system_sent == CAPTION_SYSTEM_BOOTSTRAP
        assert system_sent != CAPTION_SYSTEM


def _make_valid_caption_text() -> str:
    long_art = "word " * 500
    long_subject = "subject " * 1000
    return f"[Art Style] {long_art}\n[Subject] {long_subject}"


def _make_gemini_fake_client(captured: dict[str, Any], caption_text: str) -> object:
    class FakeModels:
        async def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text=caption_text)

    class FakeClient:
        aio = SimpleNamespace(models=FakeModels())

    return FakeClient()


class TestCaptionSingleSystemKwarg:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            (None, CAPTION_SYSTEM),
            (CAPTION_SYSTEM_BOOTSTRAP, CAPTION_SYSTEM_BOOTSTRAP),
        ],
        ids=["default", "override"],
    )
    async def test_caption_single_system_kwarg(self, tmp_path: Path, override: str | None, expected: str) -> None:
        image_path = tmp_path / "ref.png"
        image_path.write_bytes(b"fake-image")
        captured: dict[str, Any] = {}
        client = _make_gemini_fake_client(captured, _make_valid_caption_text())

        kwargs: dict[str, Any] = {
            "prompt": CAPTION_PROMPT,
            "model": "fake-model",
            "client": client,
            "cache_dir": None,
            "semaphore": asyncio.Semaphore(1),
        }
        if override is not None:
            kwargs["system"] = override

        await caption_single(image_path, **kwargs)

        assert captured["config"].system_instruction == expected

    @pytest.mark.asyncio
    async def test_caption_references_threads_system_kwarg(self, tmp_path: Path) -> None:
        image_path = tmp_path / "ref.png"
        image_path.write_bytes(b"fake-image")
        captured: dict[str, Any] = {}
        client = _make_gemini_fake_client(captured, _make_valid_caption_text())

        await caption_references(
            [image_path],
            model="fake-model",
            client=client,  # type: ignore[arg-type]
            cache_dir=tmp_path / "cache",
            semaphore=asyncio.Semaphore(1),
            prompt=CAPTION_PROMPT,
            system=CAPTION_SYSTEM_BOOTSTRAP,
        )

        assert captured["config"].system_instruction == CAPTION_SYSTEM_BOOTSTRAP


class TestZeroStepUsesBootstrapSystem:
    def test_zero_step_gemini_branch_imports_bootstrap_system(self) -> None:
        """workflow/zero_step.py must import CAPTION_SYSTEM_BOOTSTRAP and pass it to the
        Gemini captioning service at the bootstrap call site.
        """
        zero_step_src = Path(__file__).resolve().parents[1] / "src/art_style_search/workflow/zero_step.py"
        text = zero_step_src.read_text(encoding="utf-8")
        assert "CAPTION_SYSTEM_BOOTSTRAP" in text, "zero_step.py must reference CAPTION_SYSTEM_BOOTSTRAP"
        assert "system=CAPTION_SYSTEM_BOOTSTRAP" in text, (
            "zero_step.py Gemini branch must pass system=CAPTION_SYSTEM_BOOTSTRAP to caption_references"
        )
