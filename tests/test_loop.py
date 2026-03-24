"""Unit tests for art_style_search.loop helpers: _discover_images, _sample."""

from __future__ import annotations

from pathlib import Path

from art_style_search.loop import _discover_images, _sample

# ---------------------------------------------------------------------------
# _discover_images
# ---------------------------------------------------------------------------


class TestDiscoverImages:
    """_discover_images should return only image files, sorted."""

    def test_filters_and_sorts(self, tmp_path: Path) -> None:
        # Create mixed files
        (tmp_path / "b.png").touch()
        (tmp_path / "a.jpg").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / "c.jpeg").touch()

        result = _discover_images(tmp_path)

        names = [p.name for p in result]
        assert "notes.txt" not in names
        assert names == sorted(names)
        assert set(names) == {"a.jpg", "b.png", "c.jpeg"}

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert _discover_images(tmp_path) == []

    def test_no_images(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.csv").touch()
        assert _discover_images(tmp_path) == []

    def test_all_supported_extensions(self, tmp_path: Path) -> None:
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
            (tmp_path / f"img{ext}").touch()

        result = _discover_images(tmp_path)
        assert len(result) == 6

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        (tmp_path / "photo.PNG").touch()
        (tmp_path / "photo.Jpg").touch()

        result = _discover_images(tmp_path)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _sample
# ---------------------------------------------------------------------------


class TestSample:
    """_sample should return all items when count <= max, else a random subset."""

    def _make_paths(self, n: int) -> list[Path]:
        return [Path(f"/fake/{i}.png") for i in range(n)]

    def test_smaller_than_max_returns_all(self) -> None:
        items = self._make_paths(3)
        result = _sample(items, max_count=5)
        assert result is items  # exact same list object, not a copy

    def test_exact_size_returns_all(self) -> None:
        items = self._make_paths(5)
        result = _sample(items, max_count=5)
        assert result is items

    def test_larger_than_max_returns_correct_count(self) -> None:
        items = self._make_paths(10)
        result = _sample(items, max_count=4)
        assert len(result) == 4
        # All returned items should come from the original list
        for p in result:
            assert p in items

    def test_larger_than_max_no_duplicates(self) -> None:
        items = self._make_paths(20)
        result = _sample(items, max_count=7)
        assert len(result) == len(set(result))
