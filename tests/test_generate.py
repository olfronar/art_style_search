"""Unit tests for generation prompt wiring."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from art_style_search.generate import generate_single


class TestGenerateSingle:
    @pytest.mark.asyncio
    async def test_passes_system_instruction_with_negative_prompt(self, tmp_path: Path) -> None:
        output_path = tmp_path / "gen.png"
        captured: dict[str, Any] = {}

        async def fake_generate_content(**kwargs):
            captured.update(kwargs)
            inline = SimpleNamespace(data=b"png-bytes")
            part = SimpleNamespace(inline_data=inline)
            content = SimpleNamespace(parts=[part])
            candidate = SimpleNamespace(content=content)
            return SimpleNamespace(candidates=[candidate])

        client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content)))

        result = await generate_single(
            "Render a red fox.",
            index=0,
            aspect_ratio="1:1",
            output_path=output_path,
            client=client,  # type: ignore[arg-type]
            model="fake-model",
            semaphore=asyncio.Semaphore(1),
            negative_prompt="Avoid watermarks and signatures.",
        )

        assert result == output_path
        config = captured["config"]
        assert "Avoid watermarks and signatures." in config.system_instruction
