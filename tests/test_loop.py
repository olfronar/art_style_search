"""Unit tests for art_style_search.loop helpers: _discover_images, _sample, _find_global_best."""

from __future__ import annotations

from pathlib import Path

from art_style_search.loop import _discover_images, _find_global_best, _sample
from art_style_search.types import AggregatedMetrics, BranchState, PromptTemplate

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


# ---------------------------------------------------------------------------
# _find_global_best
# ---------------------------------------------------------------------------


def _make_metrics(
    dino: float = 0.5,
    lpips: float = 0.5,
    hps: float = 0.2,
    aes: float = 5.0,
) -> AggregatedMetrics:
    """Helper to build AggregatedMetrics with convenient defaults."""
    return AggregatedMetrics(
        dino_similarity_mean=dino,
        dino_similarity_std=0.0,
        lpips_distance_mean=lpips,
        lpips_distance_std=0.0,
        hps_score_mean=hps,
        hps_score_std=0.0,
        aesthetics_score_mean=aes,
        aesthetics_score_std=0.0,
    )


def _make_branch(
    branch_id: int,
    metrics: AggregatedMetrics | None = None,
) -> BranchState:
    """Helper to build a BranchState with a minimal template."""
    template = PromptTemplate()
    return BranchState(
        branch_id=branch_id,
        current_template=template,
        best_template=template,
        best_metrics=metrics,
    )


class TestFindGlobalBest:
    """_find_global_best should pick the branch with the highest composite_score."""

    def test_no_branches(self) -> None:
        template, metrics = _find_global_best([])
        assert template is None
        assert metrics is None

    def test_all_branches_without_metrics(self) -> None:
        branches = [_make_branch(0), _make_branch(1)]
        template, metrics = _find_global_best(branches)
        assert template is None
        assert metrics is None

    def test_single_branch_with_metrics(self) -> None:
        m = _make_metrics(dino=0.9, lpips=0.1, hps=0.3, aes=8.0)
        branch = _make_branch(0, metrics=m)
        template, metrics = _find_global_best([branch])
        assert template is branch.best_template
        assert metrics is m

    def test_selects_best_composite(self) -> None:
        # composite_score = 0.4*dino - 0.2*lpips + 0.2*hps + 0.2*(aes/10)
        # Branch 0: 0.4*0.5 - 0.2*0.5 + 0.2*0.2 + 0.2*0.5 = 0.20 - 0.10 + 0.04 + 0.10 = 0.24
        m_low = _make_metrics(dino=0.5, lpips=0.5, hps=0.2, aes=5.0)
        # Branch 1: 0.4*0.9 - 0.2*0.1 + 0.2*0.3 + 0.2*0.8 = 0.36 - 0.02 + 0.06 + 0.16 = 0.56
        m_high = _make_metrics(dino=0.9, lpips=0.1, hps=0.3, aes=8.0)

        branch_low = _make_branch(0, metrics=m_low)
        branch_high = _make_branch(1, metrics=m_high)

        template, metrics = _find_global_best([branch_low, branch_high])
        assert metrics is m_high
        assert template is branch_high.best_template

    def test_ignores_branches_without_metrics(self) -> None:
        m = _make_metrics(dino=0.7, lpips=0.3, hps=0.25, aes=6.0)
        branch_none = _make_branch(0)
        branch_with = _make_branch(1, metrics=m)

        template, metrics = _find_global_best([branch_none, branch_with])
        assert metrics is m
        assert template is branch_with.best_template
