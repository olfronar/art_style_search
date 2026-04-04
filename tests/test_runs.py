"""Unit tests for art_style_search.runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from art_style_search.runs import list_runs, next_auto_name, remove_all_runs, remove_run, resolve_run_dir


class TestNextAutoName:
    def test_empty_dir(self, tmp_path: Path) -> None:
        assert next_auto_name(tmp_path) == "run_001"

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert next_auto_name(tmp_path / "nope") == "run_001"

    def test_increments(self, tmp_path: Path) -> None:
        (tmp_path / "run_001").mkdir()
        (tmp_path / "run_002").mkdir()
        assert next_auto_name(tmp_path) == "run_003"

    def test_finds_max_with_gaps(self, tmp_path: Path) -> None:
        (tmp_path / "run_001").mkdir()
        (tmp_path / "run_005").mkdir()
        assert next_auto_name(tmp_path) == "run_006"

    def test_ignores_custom_names(self, tmp_path: Path) -> None:
        (tmp_path / "my-experiment").mkdir()
        (tmp_path / "run_003").mkdir()
        assert next_auto_name(tmp_path) == "run_004"

    def test_ignores_non_numeric_suffix(self, tmp_path: Path) -> None:
        (tmp_path / "run_abc").mkdir()
        assert next_auto_name(tmp_path) == "run_001"


class TestResolveRunDir:
    def test_no_run_flag_auto_names(self, tmp_path: Path) -> None:
        result = resolve_run_dir(tmp_path, None, False)
        assert result == tmp_path / "run_001"
        assert result.is_dir()

    def test_no_run_flag_increments(self, tmp_path: Path) -> None:
        (tmp_path / "run_001").mkdir(parents=True)
        result = resolve_run_dir(tmp_path, None, False)
        assert result == tmp_path / "run_002"

    def test_explicit_name_creates(self, tmp_path: Path) -> None:
        result = resolve_run_dir(tmp_path, "my-test", False)
        assert result == tmp_path / "my-test"
        assert result.is_dir()

    def test_explicit_name_resumes(self, tmp_path: Path) -> None:
        (tmp_path / "existing").mkdir()
        result = resolve_run_dir(tmp_path, "existing", False)
        assert result == tmp_path / "existing"

    def test_new_flag_creates(self, tmp_path: Path) -> None:
        result = resolve_run_dir(tmp_path, "fresh", True)
        assert result == tmp_path / "fresh"

    def test_new_flag_errors_if_exists(self, tmp_path: Path) -> None:
        (tmp_path / "taken").mkdir()
        with pytest.raises(SystemExit):
            resolve_run_dir(tmp_path, "taken", True)

    def test_invalid_name_slash(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            resolve_run_dir(tmp_path, "bad/name", False)

    def test_invalid_name_dotdot(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            resolve_run_dir(tmp_path, "..", False)

    def test_invalid_name_null(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            resolve_run_dir(tmp_path, "bad\0name", False)


class TestListRuns:
    def test_empty(self, tmp_path: Path) -> None:
        assert list_runs(tmp_path / "nope") == []

    def test_not_started(self, tmp_path: Path) -> None:
        (tmp_path / "run_001").mkdir()
        runs = list_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["name"] == "run_001"
        assert runs[0]["status"] == "not started"

    def test_in_progress(self, tmp_path: Path) -> None:
        d = tmp_path / "run_001"
        d.mkdir()
        (d / "state.json").write_text(json.dumps({"iteration": 3}))
        runs = list_runs(tmp_path)
        assert runs[0]["status"] == "in progress"
        assert runs[0]["iteration"] == 3

    def test_converged(self, tmp_path: Path) -> None:
        d = tmp_path / "run_001"
        d.mkdir()
        (d / "state.json").write_text(json.dumps({"iteration": 10, "converged": True, "convergence_reason": "PLATEAU"}))
        runs = list_runs(tmp_path)
        assert runs[0]["status"] == "converged (PLATEAU)"

    def test_corrupt_state(self, tmp_path: Path) -> None:
        d = tmp_path / "run_001"
        d.mkdir()
        (d / "state.json").write_text("not json{{{")
        runs = list_runs(tmp_path)
        assert runs[0]["status"] == "corrupt"

    def test_multiple_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "beta").mkdir()
        (tmp_path / "alpha").mkdir()
        runs = list_runs(tmp_path)
        assert [r["name"] for r in runs] == ["alpha", "beta"]


class TestRemoveRun:
    def test_removes(self, tmp_path: Path) -> None:
        d = tmp_path / "run_001"
        d.mkdir()
        (d / "state.json").write_text("{}")
        remove_run(tmp_path, "run_001")
        assert not d.exists()

    def test_nonexistent_errors(self, tmp_path: Path) -> None:
        tmp_path.mkdir(exist_ok=True)
        with pytest.raises(SystemExit):
            remove_run(tmp_path, "nope")


class TestRemoveAllRuns:
    def test_removes_all(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        (runs_dir / "run_001").mkdir(parents=True)
        (runs_dir / "run_002").mkdir()
        remove_all_runs(runs_dir)
        assert not runs_dir.exists()

    def test_nonexistent_ok(self, tmp_path: Path) -> None:
        remove_all_runs(tmp_path / "nope")  # should not raise
