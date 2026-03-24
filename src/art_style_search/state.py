"""Persistence layer: save/load loop state and iteration logs as JSON."""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from art_style_search.types import (
    AggregatedMetrics,
    BranchState,
    Caption,
    ConvergenceReason,
    IterationResult,
    LoopState,
    MetricScores,
    PromptSection,
    PromptTemplate,
    StyleProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


class _Encoder(json.JSONEncoder):
    """Custom JSON encoder for Path, dataclasses, and enums."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return _to_dict(o)
        if isinstance(o, enum.Enum):
            return o.value
        return super().default(o)


def _to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass (or nested structure) to a JSON-safe dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            result[f.name] = _to_dict(getattr(obj, f.name))
        return result
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------


def _caption_from_dict(d: dict[str, Any]) -> Caption:
    return Caption(image_path=Path(d["image_path"]), text=d["text"])


def _metric_scores_from_dict(d: dict[str, Any]) -> MetricScores:
    return MetricScores(
        dino_similarity=d["dino_similarity"],
        lpips_distance=d["lpips_distance"],
        hps_score=d["hps_score"],
        aesthetics_score=d["aesthetics_score"],
    )


def _aggregated_metrics_from_dict(d: dict[str, Any]) -> AggregatedMetrics:
    return AggregatedMetrics(
        dino_similarity_mean=d["dino_similarity_mean"],
        dino_similarity_std=d["dino_similarity_std"],
        lpips_distance_mean=d["lpips_distance_mean"],
        lpips_distance_std=d["lpips_distance_std"],
        hps_score_mean=d["hps_score_mean"],
        hps_score_std=d["hps_score_std"],
        aesthetics_score_mean=d["aesthetics_score_mean"],
        aesthetics_score_std=d["aesthetics_score_std"],
    )


def _prompt_section_from_dict(d: dict[str, Any]) -> PromptSection:
    return PromptSection(name=d["name"], description=d["description"], value=d["value"])


def _prompt_template_from_dict(d: dict[str, Any]) -> PromptTemplate:
    return PromptTemplate(
        sections=[_prompt_section_from_dict(s) for s in d.get("sections", [])],
        negative_prompt=d.get("negative_prompt"),
    )


def _style_profile_from_dict(d: dict[str, Any]) -> StyleProfile:
    return StyleProfile(
        color_palette=d["color_palette"],
        composition=d["composition"],
        technique=d["technique"],
        mood_atmosphere=d["mood_atmosphere"],
        subject_matter=d["subject_matter"],
        influences=d["influences"],
        gemini_raw_analysis=d["gemini_raw_analysis"],
        claude_raw_analysis=d["claude_raw_analysis"],
    )


def _iteration_result_from_dict(d: dict[str, Any]) -> IterationResult:
    return IterationResult(
        branch_id=d["branch_id"],
        iteration=d["iteration"],
        template=_prompt_template_from_dict(d["template"]),
        rendered_prompt=d["rendered_prompt"],
        image_paths=[Path(p) for p in d["image_paths"]],
        per_image_scores=[_metric_scores_from_dict(s) for s in d["per_image_scores"]],
        aggregated=_aggregated_metrics_from_dict(d["aggregated"]),
        claude_analysis=d["claude_analysis"],
        template_changes=d["template_changes"],
        kept=d["kept"],
        hypothesis=d.get("hypothesis", ""),
        experiment=d.get("experiment", ""),
        vision_feedback=d.get("vision_feedback", ""),
        roundtrip_feedback=d.get("roundtrip_feedback", ""),
        iteration_captions=[_caption_from_dict(c) for c in d.get("iteration_captions", [])],
    )


def _branch_state_from_dict(d: dict[str, Any]) -> BranchState:
    return BranchState(
        branch_id=d["branch_id"],
        current_template=_prompt_template_from_dict(d["current_template"]),
        best_template=_prompt_template_from_dict(d["best_template"]),
        best_metrics=_aggregated_metrics_from_dict(d["best_metrics"]) if d.get("best_metrics") is not None else None,
        history=[_iteration_result_from_dict(h) for h in d.get("history", [])],
        research_log=d.get("research_log", ""),
        plateau_counter=d.get("plateau_counter", 0),
        stopped=d.get("stopped", False),
        stop_reason=ConvergenceReason(d["stop_reason"]) if d.get("stop_reason") is not None else None,
    )


def _loop_state_from_dict(d: dict[str, Any]) -> LoopState:
    return LoopState(
        iteration=d["iteration"],
        branches=[_branch_state_from_dict(b) for b in d["branches"]],
        captions=[_caption_from_dict(c) for c in d["captions"]],
        style_profile=_style_profile_from_dict(d["style_profile"]),
        fixed_references=[Path(p) for p in d.get("fixed_references", [])],
        global_best_prompt=d.get("global_best_prompt", ""),
        global_best_metrics=(
            _aggregated_metrics_from_dict(d["global_best_metrics"])
            if d.get("global_best_metrics") is not None
            else None
        ),
        converged=d.get("converged", False),
        convergence_reason=(
            ConvergenceReason(d["convergence_reason"]) if d.get("convergence_reason") is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_state(state: LoopState, path: Path) -> None:
    """Serialize *state* to JSON, writing atomically via temp-file + rename."""
    data = _to_dict(state)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory, then atomically rename.
    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp_path = Path(tmp_path_str)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=_Encoder, ensure_ascii=False, indent=2)
        tmp_path.rename(path)
        logger.info("State saved to %s", path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_state(path: Path) -> LoopState | None:
    """Load a previously saved *LoopState* from *path*, or return None if the file does not exist."""
    if not path.exists():
        logger.debug("No state file at %s", path)
        return None

    logger.info("Loading state from %s", path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _loop_state_from_dict(raw)


def save_iteration_log(result: IterationResult, log_dir: Path) -> None:
    """Write a single iteration result to ``{log_dir}/iter_{NNN}_branch_{M}.json``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"iter_{result.iteration:03d}_branch_{result.branch_id}.json"
    log_path = log_dir / filename

    data = _to_dict(result)
    log_path.write_text(json.dumps(data, cls=_Encoder, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Iteration log written to %s", log_path)
