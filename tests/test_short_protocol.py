"""Unit tests for --protocol short behavior (3-iter cheap foundation pass).

Short-protocol semantics:
- Iteration budget is hard-clamped to 3 regardless of --max-iterations (phase-1 foundation pass).
- Plateau window clamped to 3 (same as classic default under the new regime).
- The portfolio category quota (A4) and headroom-weighted scoring (A6) apply to both protocols.

This file covers the budget clamp. Iter-specific gates (iter 1 slack-gate bold-only, iter 2
canon-only) are out of scope for the current pragmatic skeleton and are tracked separately.
"""

from __future__ import annotations

from pathlib import Path

from art_style_search.config import parse_args


def _args(tmp_path: Path, *extra: str) -> list[str]:
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    (ref_dir / "dummy.png").touch()
    return [
        "--reference-dir",
        str(ref_dir),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--run",
        "short_test",
        "--new",
        "--anthropic-api-key",
        "test",
        "--google-api-key",
        "test",
        *extra,
    ]


class TestShortProtocolDefaults:
    def test_short_is_default_protocol(self, tmp_path: Path) -> None:
        cfg = parse_args(_args(tmp_path))
        assert cfg.protocol == "short"

    def test_classic_still_available(self, tmp_path: Path) -> None:
        cfg = parse_args(_args(tmp_path, "--protocol", "classic"))
        assert cfg.protocol == "classic"


class TestShortProtocolClampsMaxIterations:
    def test_short_clamps_max_iterations_to_3(self, tmp_path: Path) -> None:
        """--max-iterations 50 under short protocol → hard-clamped to 3 (phase-1 budget)."""
        cfg = parse_args(_args(tmp_path, "--max-iterations", "50"))
        assert cfg.max_iterations == 3, f"short protocol must hard-clamp max_iterations to 3; got {cfg.max_iterations}"

    def test_classic_respects_max_iterations(self, tmp_path: Path) -> None:
        """Under classic protocol the CLI value is respected (no clamp)."""
        cfg = parse_args(_args(tmp_path, "--protocol", "classic", "--max-iterations", "50"))
        assert cfg.max_iterations == 50

    def test_short_default_max_iterations_is_3(self, tmp_path: Path) -> None:
        cfg = parse_args(_args(tmp_path))
        assert cfg.max_iterations == 3
