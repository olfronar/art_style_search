"""Data-loading helpers for HTML reports."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from art_style_search.scoring import composite_score
from art_style_search.state import (
    load_iteration_log,
    load_iteration_proposals,
    load_manifest,
    load_promotion_log,
    load_state,
)
from art_style_search.types import IterationResult, LoopState, PromotionDecision, RunManifest
from art_style_search.workflow.proposal_recorder import ProposalBatchRecorder

logger = logging.getLogger(__name__)

_LOG_PATTERN = re.compile(r"iter_(\d+)_branch_(\d+)\.json$")
_PROPOSALS_LOG_PATTERN = re.compile(r"iter_(\d+)_proposals\.json$")
_SYNTHESIS_EXP_PREFIX = "Synthesis"


@dataclass(frozen=True)
class CaptionEntry:
    """One caption produced for a specific image by a specific experiment."""

    iteration: int
    branch_id: int
    text: str
    score: float
    hypothesis: str
    is_kept: bool
    is_top_raw: bool
    is_synthesis: bool


@dataclass
class ReportData:
    """Everything the renderer needs for one run."""

    run_name: str
    run_dir: Path
    state: LoopState
    iteration_logs: dict[int, list[IterationResult]] = field(default_factory=dict)
    iteration_proposals: dict[int, ProposalBatchRecorder] = field(default_factory=dict)
    manifest: RunManifest | None = None
    promotion_decisions: list[PromotionDecision] = field(default_factory=list)
    zero_step_captions: dict[Path, str] = field(default_factory=dict)

    def iteration_numbers(self) -> list[int]:
        """Sorted list of iteration indices that have at least one log."""
        return sorted(self.iteration_logs.keys())

    def reference_images(self) -> list[Path]:
        """Stable, sorted list of reference image paths seen across all iterations.

        Drawn from per-iteration ``Caption.image_path`` rather than ``state.fixed_references``
        so the report works for resumed/partial runs where state and disk diverge.
        """
        seen: dict[str, Path] = {}
        # Zero-step first, then iteration captions — earliest reference wins for stable ordering.
        for ref in sorted(self.zero_step_captions.keys()):
            seen.setdefault(str(ref), ref)
        for iteration in self.iteration_numbers():
            for result in self.iteration_logs[iteration]:
                for cap in result.iteration_captions:
                    seen.setdefault(str(cap.image_path), cap.image_path)
        # Fallback: pull any state.fixed_references not seen above (e.g. zero-iteration runs).
        for ref in self.state.fixed_references:
            seen.setdefault(str(ref), ref)
        return sorted(seen.values(), key=lambda p: p.as_posix())

    def caption_history_for(self, image_path: Path) -> dict[int, list[CaptionEntry]]:
        """Map iteration → ordered CaptionEntry list (winner first) for one reference image."""
        history: dict[int, list[CaptionEntry]] = {}
        target = str(image_path)
        for iteration in self.iteration_numbers():
            results = self.iteration_logs[iteration]
            kept_id = k.branch_id if (k := self.kept_of(iteration)) else None
            top_id = t.branch_id if (t := self.top_scoring_of(iteration)) else None
            entries: list[CaptionEntry] = []
            for result in results:
                cap_text = next(
                    (c.text for c in result.iteration_captions if str(c.image_path) == target),
                    None,
                )
                if cap_text is None:
                    continue
                exp = (result.experiment or "").strip()
                entries.append(
                    CaptionEntry(
                        iteration=iteration,
                        branch_id=result.branch_id,
                        text=cap_text,
                        score=composite_score(result.aggregated),
                        hypothesis=result.hypothesis,
                        is_kept=(result.branch_id == kept_id),
                        is_top_raw=(result.branch_id == top_id),
                        is_synthesis=exp.startswith(_SYNTHESIS_EXP_PREFIX),
                    )
                )
            if entries:
                # Pin kept first, then top_raw (if distinct), then by descending score.
                entries.sort(key=lambda e: (not e.is_kept, not e.is_top_raw, -e.score))
                history[iteration] = entries
        return history

    @property
    def requested_ref_count(self) -> int:
        if self.manifest is not None:
            return self.manifest.num_fixed_refs
        return len(self.state.fixed_references)

    @property
    def discovered_ref_count(self) -> int:
        if self.manifest is not None:
            if self.manifest.discovered_reference_count:
                return self.manifest.discovered_reference_count
            return len(self.manifest.reference_image_hashes)
        return len(self.state.fixed_references)

    @property
    def actual_ref_count(self) -> int:
        return len(self.state.fixed_references)

    def kept_of(self, iteration: int) -> IterationResult | None:
        """Return the kept experiment, falling back to highest ``composite_score``."""
        results = self.iteration_logs.get(iteration, [])
        if not results:
            return None
        kept = [r for r in results if r.kept]
        if kept:
            return max(kept, key=lambda r: composite_score(r.aggregated))
        return max(results, key=lambda r: composite_score(r.aggregated))

    def top_scoring_of(self, iteration: int) -> IterationResult | None:
        """Return the highest raw-composite experiment for an iteration."""
        results = self.iteration_logs.get(iteration, [])
        if not results:
            return None
        return max(results, key=lambda r: composite_score(r.aggregated))

    def winner_of(self, iteration: int) -> IterationResult | None:
        """Backward-compatible alias for the kept experiment."""
        return self.kept_of(iteration)


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


def _load_iteration_proposals(log_dir: Path) -> dict[int, ProposalBatchRecorder]:
    """Parse every ``iter_NNN_proposals.json`` under *log_dir* into a recorder per iteration."""
    result: dict[int, ProposalBatchRecorder] = {}
    if not log_dir.is_dir():
        return result
    for path in sorted(log_dir.glob("iter_*_proposals.json")):
        if not _PROPOSALS_LOG_PATTERN.search(path.name):
            continue
        try:
            recorder = load_iteration_proposals(path)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Skipping malformed proposals log %s: %s", path, exc)
            continue
        result[recorder.iteration] = recorder
    return result


def _load_zero_step_captions(captions_dir: Path) -> dict[Path, str]:
    """Read the per-image JSON files at ``<run>/logs/captions/`` written by the bootstrap captioner."""
    result: dict[Path, str] = {}
    if not captions_dir.is_dir():
        return result
    for path in sorted(captions_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping malformed zero-step caption %s: %s", path, exc)
            continue
        image_path_str = payload.get("image_path")
        text = payload.get("text")
        if not isinstance(image_path_str, str) or not isinstance(text, str):
            continue
        result[Path(image_path_str)] = text
    return result


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
        iteration_proposals=_load_iteration_proposals(run_dir / "logs"),
        manifest=load_manifest(run_dir / "run_manifest.json"),
        promotion_decisions=load_promotion_log(run_dir / "promotion_log.jsonl"),
        zero_step_captions=_load_zero_step_captions(run_dir / "logs" / "captions"),
    )


def _rel(target: Path, report_dir: Path) -> str:
    """Return an ``<img src>``-safe relative path from *report_dir* to *target*."""
    try:
        rel = os.path.relpath(target.resolve(), report_dir.resolve())
    except ValueError:
        return target.resolve().as_uri()
    return rel.replace(os.sep, "/")
