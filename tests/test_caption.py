"""Unit tests for bootstrap caption prompt and validation behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from art_style_search.caption import CAPTION_PROMPT, caption_single


class TestBootstrapCaptionPrompt:
    def test_uses_bracketed_art_style_and_subject_sections(self) -> None:
        assert "[Art Style]" in CAPTION_PROMPT
        assert "[Subject]" in CAPTION_PROMPT
        assert "**Subjects**" not in CAPTION_PROMPT
        assert "Target 2000-6000 words" in CAPTION_PROMPT
        assert "[Subject] and [Art Style] each 800-2000 words" in CAPTION_PROMPT


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
        captured: dict[str, object] = {}

        class FakeModels:
            async def generate_content(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(text="[Art Style] " + ("detail " * 120))

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
