"""Data-loading helpers for HTML reports."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from art_style_search.scoring import composite_score
from art_style_search.state import load_iteration_log, load_manifest, load_promotion_log, load_state
from art_style_search.types import IterationResult, LoopState, PromotionDecision, RunManifest

logger = logging.getLogger(__name__)

_LOG_PATTERN = re.compile(r"iter_(\d+)_branch_(\d+)\.json$")


@dataclass
class ReportData:
    """Everything the renderer needs for one run."""

    run_name: str
    run_dir: Path
    state: LoopState
    iteration_logs: dict[int, list[IterationResult]] = field(default_factory=dict)
    manifest: RunManifest | None = None
    promotion_decisions: list[PromotionDecision] = field(default_factory=list)
    holdout_summary: dict[str, Any] | None = None

    def iteration_numbers(self) -> list[int]:
        """Sorted list of iteration indices that have at least one log."""
        return sorted(self.iteration_logs.keys())

    def winner_of(self, iteration: int) -> IterationResult | None:
        """Return the experiment with the highest ``composite_score`` for *iteration*."""
        results = self.iteration_logs.get(iteration, [])
        if not results:
            return None
        return max(results, key=lambda r: composite_score(r.aggregated))


def _load_iteration_logs(log_dir: Path) -> dict[int, list[IterationResult]]:
    """Parse every ``iter_NNN_branch_M.json`` under *log_dir*."""
    result: dict[int, list[IterationResult]] = {}
    if not log_dir.is_dir():
        return result

    for path in sorted(log_dir.glob("iter_*_branch_*.json")):
        match = _LOG_PATTERN.search(path.name)
        if not match:
            continue
        try:
            record = load_iteration_log(path)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Skipping malformed iteration log %s: %s", path, exc)
            continue
        result.setdefault(record.iteration, []).append(record)

    for iteration_results in result.values():
        iteration_results.sort(key=lambda r: r.branch_id)
    return result


def _load_holdout_summary(run_dir: Path) -> dict[str, Any] | None:
    """Load holdout_summary.json if it exists and has silent images."""
    holdout_path = run_dir / "holdout_summary.json"
    if not holdout_path.exists():
        return None
    try:
        summary = json.loads(holdout_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping malformed holdout summary: %s", exc)
        return None
    if summary.get("silent_image_count", 0) == 0:
        return None
    return summary


def load_report_data(run_dir: Path) -> ReportData:
    """Load *state.json* and all iteration logs from *run_dir*."""
    state_file = run_dir / "state.json"
    state = load_state(state_file)
    if state is None:
        raise FileNotFoundError(f"No state.json found in {run_dir} — run not started yet")

    return ReportData(
        run_name=run_dir.name,
        run_dir=run_dir,
        state=state,
        iteration_logs=_load_iteration_logs(run_dir / "logs"),
        manifest=load_manifest(run_dir / "run_manifest.json"),
        promotion_decisions=load_promotion_log(run_dir / "promotion_log.jsonl"),
        holdout_summary=_load_holdout_summary(run_dir),
    )


def _rel(target: Path, report_dir: Path) -> str:
    """Return an ``<img src>``-safe relative path from *report_dir* to *target*."""
    try:
        rel = os.path.relpath(target.resolve(), report_dir.resolve())
    except ValueError:
        return target.resolve().as_uri()
    return rel.replace(os.sep, "/")
