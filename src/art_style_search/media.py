"""Image/media helpers shared across captioning, evaluation, and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from google.genai import types as genai_types  # type: ignore[attr-defined]

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


def image_to_gemini_part(path: Path) -> genai_types.Part:
    """Read an image file and return a Gemini Part with correct MIME type."""
    mime_type = MIME_MAP.get(path.suffix.lower(), "image/png")
    return genai_types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


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
