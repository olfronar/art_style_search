"""Run directory management — isolation, naming, listing, cleanup."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path


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


def resolve_run_dir(runs_dir: Path, run_name: str | None, new: bool) -> Path:
    """Resolve the target run directory (does NOT create it).

    Raises ``SystemExit`` on invalid input.
    """
    if run_name is not None and ("/" in run_name or "\\" in run_name or run_name in (".", "..") or "\0" in run_name):
        sys.exit(f"Invalid run name: {run_name!r}")

    if run_name is None:
        # No --run flag: always create a new auto-named run
        name = next_auto_name(runs_dir)
        target = runs_dir / name
        # Guard against race: try mkdir without exist_ok
        try:
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # Retry once with next number
            name = next_auto_name(runs_dir)
            target = runs_dir / name
            target.mkdir(parents=True, exist_ok=False)
        return target

    target = runs_dir / run_name
    if new and target.exists():
        sys.exit(f"Run {run_name!r} already exists. Remove it first or drop --new.")
    return target


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
            try:
                with open(state_file) as f:
                    data = json.load(f)
                info["iteration"] = data.get("iteration", 0)
                if data.get("converged"):
                    reason = data.get("convergence_reason", "unknown")
                    info["status"] = f"converged ({reason})"
                else:
                    info["status"] = "in progress"
            except (json.JSONDecodeError, OSError):
                info["status"] = "corrupt"
        results.append(info)
    return results


def remove_run(runs_dir: Path, run_name: str) -> None:
    """Remove a specific run directory."""
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
