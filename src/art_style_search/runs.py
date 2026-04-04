"""Run directory management — isolation, naming, listing, cleanup."""

from __future__ import annotations

import contextlib
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_RUNS_DIR = Path("runs")


def next_auto_name(runs_dir: Path) -> str:
    """Return the next auto-incremented run name like ``run_001``."""
    max_num = 0
    if runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                try:
                    num = int(d.name[4:])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
    return f"run_{max_num + 1:03d}"


def _validate_run_name(run_name: str) -> None:
    """Exit if *run_name* contains path-traversal characters."""
    if "/" in run_name or "\\" in run_name or run_name in (".", "..") or "\0" in run_name:
        sys.exit(f"Invalid run name: {run_name!r}")


def resolve_run_dir(runs_dir: Path, run_name: str | None, new: bool) -> Path:
    """Resolve and create the target run directory.

    For auto-named runs, the directory is created atomically to guard
    against races.  For named runs, the directory is created if absent.

    Raises ``SystemExit`` on invalid input.
    """
    if run_name is not None:
        _validate_run_name(run_name)

    if run_name is None:
        name = next_auto_name(runs_dir)
        target = runs_dir / name
        # Guard against race: try mkdir without exist_ok
        try:
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            name = next_auto_name(runs_dir)
            target = runs_dir / name
            target.mkdir(parents=True, exist_ok=False)
        return target

    target = runs_dir / run_name
    if new and target.exists():
        sys.exit(f"Run {run_name!r} already exists. Remove it first or drop --new.")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _read_state_summary(state_file: Path) -> dict | None:
    """Read only the top-level scalar fields from a state.json file.

    Avoids deserializing the full multi-MB object graph.
    """
    try:
        with open(state_file) as f:
            # State is pretty-printed (indent=2). Top-level scalars appear
            # in the first ~20 lines, before any nested object.  Read a
            # small prefix and close early.
            head = f.read(2048)
        # Parse the prefix — it won't be valid JSON, so extract fields manually.
        data = {}
        for key in ("iteration", "converged", "convergence_reason"):
            # Pattern: '  "key": value,' or '  "key": value\n'
            marker = f'"{key}":'
            idx = head.find(marker)
            if idx == -1:
                continue
            rest = head[idx + len(marker) :].lstrip()
            if rest.startswith('"'):
                end = rest.index('"', 1)
                data[key] = rest[1:end]
            elif rest.startswith("true"):
                data[key] = True
            elif rest.startswith("false"):
                data[key] = False
            elif rest.startswith("null"):
                data[key] = None
            else:
                # numeric
                end = min(
                    (rest.index(c) for c in (",", "\n", "}") if c in rest),
                    default=len(rest),
                )
                with contextlib.suppress(ValueError):
                    data[key] = int(rest[:end].strip())
        return data if data else None
    except OSError:
        return None


def list_runs(runs_dir: Path) -> list[dict]:
    """Return metadata for all runs, sorted by creation time."""
    if not runs_dir.is_dir():
        return []

    results = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        state_file = d / "state.json"
        info: dict = {
            "name": d.name,
            "status": "not started",
            "iteration": 0,
            "created": datetime.fromtimestamp(d.stat().st_ctime, tz=UTC).strftime("%Y-%m-%d %H:%M"),
        }
        if state_file.is_file():
            data = _read_state_summary(state_file)
            if data is not None:
                info["iteration"] = data.get("iteration", 0)
                if data.get("converged"):
                    reason = data.get("convergence_reason", "unknown")
                    info["status"] = f"converged ({reason})"
                else:
                    info["status"] = "in progress"
            else:
                info["status"] = "corrupt"
        results.append(info)
    return results


def remove_run(runs_dir: Path, run_name: str) -> None:
    """Remove a specific run directory."""
    _validate_run_name(run_name)
    target = runs_dir / run_name
    if not target.is_dir():
        available = [d.name for d in runs_dir.iterdir() if d.is_dir()] if runs_dir.is_dir() else []
        msg = f"Run {run_name!r} not found."
        if available:
            msg += f" Available: {', '.join(sorted(available))}"
        sys.exit(msg)
    shutil.rmtree(target)
    print(f"Removed {target}/")


def remove_all_runs(runs_dir: Path) -> None:
    """Remove the entire runs directory."""
    if runs_dir.is_dir():
        shutil.rmtree(runs_dir)
        print(f"Removed {runs_dir}/")
    else:
        print("Nothing to clean.")
