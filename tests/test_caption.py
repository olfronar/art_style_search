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
        assert "Target 400-600 words total" in CAPTION_PROMPT
        assert "80-140 words" in CAPTION_PROMPT


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
