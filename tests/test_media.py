"""Unit tests for media helpers used by multimodal providers."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from art_style_search.media import image_to_xai_data_url


def _write_image(path: Path, *, fmt: str) -> None:
    Image.new("RGB", (16, 16), color=(20, 40, 60)).save(path, format=fmt)


class TestImageToXAIDataURL:
    def test_preserves_png_mime_type(self, tmp_path: Path) -> None:
        image_path = tmp_path / "sample.png"
        _write_image(image_path, fmt="PNG")

        result = image_to_xai_data_url(image_path)

        assert result.startswith("data:image/png;base64,")

    def test_transcodes_unsupported_bmp_to_png(self, tmp_path: Path) -> None:
        image_path = tmp_path / "sample.bmp"
        _write_image(image_path, fmt="BMP")

        result = image_to_xai_data_url(image_path)

        assert result.startswith("data:image/png;base64,")

    def test_rejects_payloads_larger_than_twenty_mib(self, tmp_path: Path) -> None:
        image_path = tmp_path / "large.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * (20 * 1024 * 1024 + 1))

        with pytest.raises(ValueError, match="20 MiB"):
            image_to_xai_data_url(image_path)
