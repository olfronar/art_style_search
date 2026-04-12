"""Image/media helpers shared across captioning, evaluation, and reporting."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from google.genai import types as genai_types  # type: ignore[attr-defined]
from PIL import Image

if TYPE_CHECKING:
    from art_style_search.types import IterationResult


MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

IMAGE_EXTENSIONS: frozenset[str] = frozenset(MIME_MAP)
_XAI_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})
_XAI_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def image_to_gemini_part(path: Path) -> genai_types.Part:
    """Read an image file and return a Gemini Part with correct MIME type."""
    mime_type = MIME_MAP.get(path.suffix.lower(), "image/png")
    return genai_types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def image_to_xai_data_url(path: Path) -> str:
    """Return a data URL suitable for xAI image inputs.

    xAI accepts only PNG/JPEG image uploads. Unsupported repo-local formats are
    transcoded to PNG in memory before base64 encoding.
    """
    suffix = path.suffix.lower()
    if suffix in _XAI_IMAGE_EXTENSIONS:
        mime_type = MIME_MAP.get(suffix, "image/png")
        image_bytes = path.read_bytes()
    else:
        with Image.open(path) as image:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        mime_type = "image/png"

    if len(image_bytes) > _XAI_MAX_IMAGE_BYTES:
        msg = f"xAI image input exceeds 20 MiB after preparation: {path}"
        raise ValueError(msg)

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_ref_gen_pairs(result: IterationResult) -> list[tuple[Path, Path]]:
    """Reconstruct (reference, generated) pairs from an IterationResult."""
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
