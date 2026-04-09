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
    Caption,
    CategoryProgress,
    ConvergenceReason,
    Hypothesis,
    IterationResult,
    KnowledgeBase,
    LoopState,
    MetricScores,
    OpenProblem,
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
    # Backward compat: old state.json had dino_similarity + lpips_distance
    dreamsim = d.get("dreamsim_similarity", 0.0)
    if dreamsim == 0.0 and "dino_similarity" in d:
        dreamsim = d["dino_similarity"]  # approximate: use old DINO score as DreamSim stand-in
    return MetricScores(
        dreamsim_similarity=dreamsim,
        hps_score=d["hps_score"],
        aesthetics_score=d["aesthetics_score"],
        color_histogram=d.get("color_histogram", 0.0),
        ssim=d.get("ssim", 0.0),
        vision_style=d.get("vision_style", 0.5),
        vision_subject=d.get("vision_subject", 0.5),
        vision_composition=d.get("vision_composition", 0.5),
    )


def _aggregated_metrics_from_dict(d: dict[str, Any]) -> AggregatedMetrics:
    # Backward compat: old state.json had dino_similarity + lpips_distance fields
    ds_mean = d.get("dreamsim_similarity_mean", 0.0)
    ds_std = d.get("dreamsim_similarity_std", 0.0)
    if ds_mean == 0.0 and "dino_similarity_mean" in d:
        ds_mean = d["dino_similarity_mean"]
        ds_std = d.get("dino_similarity_std", 0.0)
    return AggregatedMetrics(
        dreamsim_similarity_mean=ds_mean,
        dreamsim_similarity_std=ds_std,
        hps_score_mean=d["hps_score_mean"],
        hps_score_std=d["hps_score_std"],
        aesthetics_score_mean=d["aesthetics_score_mean"],
        aesthetics_score_std=d["aesthetics_score_std"],
        color_histogram_mean=d.get("color_histogram_mean", 0.0),
        color_histogram_std=d.get("color_histogram_std", 0.0),
        ssim_mean=d.get("ssim_mean", 0.0),
        ssim_std=d.get("ssim_std", 0.0),
        style_consistency=d.get("style_consistency", 0.0),
        vision_style=d.get("vision_style", 0.5),
        vision_subject=d.get("vision_subject", 0.5),
        vision_composition=d.get("vision_composition", 0.5),
        vision_style_std=d.get("vision_style_std", 0.0),
        vision_subject_std=d.get("vision_subject_std", 0.0),
        vision_composition_std=d.get("vision_composition_std", 0.0),
    )


def _prompt_section_from_dict(d: dict[str, Any]) -> PromptSection:
    return PromptSection(name=d["name"], description=d["description"], value=d["value"])


def _prompt_template_from_dict(d: dict[str, Any]) -> PromptTemplate:
    return PromptTemplate(
        sections=[_prompt_section_from_dict(s) for s in d.get("sections", [])],
        negative_prompt=d.get("negative_prompt"),
        caption_sections=d.get("caption_sections", []),
        caption_length_target=d.get("caption_length_target", 0),
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


def _hypothesis_from_dict(d: dict[str, Any]) -> Hypothesis:
    return Hypothesis(
        id=d["id"],
        iteration=d["iteration"],
        parent_id=d.get("parent_id"),
        statement=d["statement"],
        experiment=d["experiment"],
        category=d["category"],
        outcome=d["outcome"],
        metric_delta=d.get("metric_delta", {}),
        kept=d["kept"],
        lesson=d.get("lesson", ""),
    )


def _open_problem_from_dict(d: dict[str, Any]) -> OpenProblem:
    return OpenProblem(
        text=d["text"],
        category=d["category"],
        priority=d["priority"],
        metric_gap=d.get("metric_gap"),
        since_iteration=d.get("since_iteration", 0),
    )


def _category_progress_from_dict(d: dict[str, Any]) -> CategoryProgress:
    return CategoryProgress(
        category=d["category"],
        best_perceptual_delta=d.get("best_perceptual_delta", d.get("best_dino_delta")),
        confirmed_insights=d.get("confirmed_insights", []),
        rejected_approaches=d.get("rejected_approaches", []),
        hypothesis_ids=d.get("hypothesis_ids", []),
    )


def _knowledge_base_from_dict(d: dict[str, Any]) -> KnowledgeBase:
    return KnowledgeBase(
        hypotheses=[_hypothesis_from_dict(h) for h in d.get("hypotheses", [])],
        categories={k: _category_progress_from_dict(v) for k, v in d.get("categories", {}).items()},
        open_problems=[_open_problem_from_dict(p) for p in d.get("open_problems", [])],
        next_id=d.get("next_id", 1),
    )


def _loop_state_from_dict(d: dict[str, Any]) -> LoopState:
    return LoopState(
        iteration=d["iteration"],
        current_template=_prompt_template_from_dict(d["current_template"]),
        best_template=_prompt_template_from_dict(d["best_template"]),
        best_metrics=_aggregated_metrics_from_dict(d["best_metrics"]) if d.get("best_metrics") is not None else None,
        knowledge_base=_knowledge_base_from_dict(d["knowledge_base"]) if d.get("knowledge_base") else KnowledgeBase(),
        captions=[_caption_from_dict(c) for c in d["captions"]],
        style_profile=_style_profile_from_dict(d["style_profile"]),
        fixed_references=[Path(p) for p in d.get("fixed_references", [])],
        experiment_history=[_iteration_result_from_dict(r) for r in d.get("experiment_history", [])],
        last_iteration_results=[_iteration_result_from_dict(r) for r in d.get("last_iteration_results", [])],
        prev_best_captions=[_caption_from_dict(c) for c in d.get("prev_best_captions", [])],
        plateau_counter=d.get("plateau_counter", 0),
        global_best_prompt=d.get("global_best_prompt", ""),
        global_best_metrics=(
            _aggregated_metrics_from_dict(d["global_best_metrics"])
            if d.get("global_best_metrics") is not None
            else None
        ),
        review_feedback=d.get("review_feedback", ""),
        pairwise_feedback=d.get("pairwise_feedback", ""),
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


def load_iteration_log(path: Path) -> IterationResult:
    """Load a single iteration-log JSON file into an ``IterationResult``.

    Inverse of :func:`save_iteration_log`.  Used by the HTML report generator
    (``art_style_search.report``) to read all per-experiment logs for a run.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _iteration_result_from_dict(raw)
