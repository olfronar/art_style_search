"""Unit tests for knowledge.build_caption_diffs."""

from __future__ import annotations

from pathlib import Path

from art_style_search.knowledge import build_caption_diffs
from art_style_search.types import Caption


class TestBuildCaptionDiffs:
    def test_empty_inputs(self) -> None:
        assert build_caption_diffs([], []) == ""
        assert build_caption_diffs([Caption(image_path=Path("a.png"), text="x")], []) == ""
        assert build_caption_diffs([], [Caption(image_path=Path("a.png"), text="x")]) == ""

    def test_unchanged_caption(self) -> None:
        path = Path("img.png")
        prev = [Caption(image_path=path, text="Same caption text")]
        worst = [Caption(image_path=path, text="Same caption text")]
        result = build_caption_diffs(prev, worst)
        assert "UNCHANGED" in result
        assert "img.png" in result

    def test_changed_caption(self) -> None:
        path = Path("img.png")
        prev = [Caption(image_path=path, text="Old description of the image style")]
        worst = [Caption(image_path=path, text="New description of the image style")]
        result = build_caption_diffs(prev, worst)
        assert "PREV:" in result
        assert "NOW:" in result
        assert "img.png" in result

    def test_no_matching_paths(self) -> None:
        prev = [Caption(image_path=Path("a.png"), text="text a")]
        worst = [Caption(image_path=Path("b.png"), text="text b")]
        assert build_caption_diffs(prev, worst) == ""

    def test_mixed_changed_and_unchanged(self) -> None:
        p1, p2 = Path("a.png"), Path("b.png")
        prev = [
            Caption(image_path=p1, text="same"),
            Caption(image_path=p2, text="old text"),
        ]
        worst = [
            Caption(image_path=p1, text="same"),
            Caption(image_path=p2, text="new text"),
        ]
        result = build_caption_diffs(prev, worst)
        assert "UNCHANGED" in result
        assert "PREV:" in result
