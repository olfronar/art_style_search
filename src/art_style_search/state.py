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
    # Note: old state.json files may contain "ssim" — silently ignored for backward compat.
    return MetricScores(
        dino_similarity=d["dino_similarity"],
        lpips_distance=d["lpips_distance"],
        hps_score=d["hps_score"],
        aesthetics_score=d["aesthetics_score"],
        color_histogram=d.get("color_histogram", 0.0),
        texture=d.get("texture", 0.0),
    )


def _aggregated_metrics_from_dict(d: dict[str, Any]) -> AggregatedMetrics:
    # Note: old state.json files may contain "ssim_mean"/"ssim_std" — silently ignored for backward compat.
    return AggregatedMetrics(
        dino_similarity_mean=d["dino_similarity_mean"],
        dino_similarity_std=d["dino_similarity_std"],
        lpips_distance_mean=d["lpips_distance_mean"],
        lpips_distance_std=d["lpips_distance_std"],
        hps_score_mean=d["hps_score_mean"],
        hps_score_std=d["hps_score_std"],
        aesthetics_score_mean=d["aesthetics_score_mean"],
        aesthetics_score_std=d["aesthetics_score_std"],
        color_histogram_mean=d.get("color_histogram_mean", 0.0),
        color_histogram_std=d.get("color_histogram_std", 0.0),
        texture_mean=d.get("texture_mean", 0.0),
        texture_std=d.get("texture_std", 0.0),
        vision_style=d.get("vision_style", 5.0),
        vision_subject=d.get("vision_subject", 5.0),
        vision_composition=d.get("vision_composition", 5.0),
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
        best_dino_delta=d.get("best_dino_delta"),
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


def _branch_state_from_dict(d: dict[str, Any]) -> BranchState:
    kb_raw = d.get("knowledge_base")
    knowledge_base = _knowledge_base_from_dict(kb_raw) if kb_raw else KnowledgeBase()
    return BranchState(
        branch_id=d["branch_id"],
        current_template=_prompt_template_from_dict(d["current_template"]),
        best_template=_prompt_template_from_dict(d["best_template"]),
        best_metrics=_aggregated_metrics_from_dict(d["best_metrics"]) if d.get("best_metrics") is not None else None,
        history=[_iteration_result_from_dict(h) for h in d.get("history", [])],
        research_log=d.get("research_log", ""),
        knowledge_base=knowledge_base,
        plateau_counter=d.get("plateau_counter", 0),
        stopped=d.get("stopped", False),
        stop_reason=ConvergenceReason(d["stop_reason"]) if d.get("stop_reason") is not None else None,
    )


def _loop_state_from_dict(d: dict[str, Any]) -> LoopState:
    # Handle new experiment-based format
    if "current_template" in d:
        return LoopState(
            iteration=d["iteration"],
            current_template=_prompt_template_from_dict(d["current_template"]),
            best_template=_prompt_template_from_dict(d["best_template"]),
            best_metrics=(
                _aggregated_metrics_from_dict(d["best_metrics"]) if d.get("best_metrics") is not None else None
            ),
            knowledge_base=_knowledge_base_from_dict(d["knowledge_base"])
            if d.get("knowledge_base")
            else KnowledgeBase(),
            captions=[_caption_from_dict(c) for c in d["captions"]],
            style_profile=_style_profile_from_dict(d["style_profile"]),
            fixed_references=[Path(p) for p in d.get("fixed_references", [])],
            experiment_history=[_iteration_result_from_dict(r) for r in d.get("experiment_history", [])],
            last_iteration_results=[_iteration_result_from_dict(r) for r in d.get("last_iteration_results", [])],
            plateau_counter=d.get("plateau_counter", 0),
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

    # Migrate from old branch-based format
    branches = [_branch_state_from_dict(b) for b in d["branches"]]

    # Find the best branch to use as current template
    best_branch = branches[0]
    from art_style_search.types import composite_score

    for b in branches:
        if b.best_metrics is not None and (
            best_branch.best_metrics is None
            or composite_score(b.best_metrics) > composite_score(best_branch.best_metrics)
        ):
            best_branch = b

    # Merge all branch KBs into one shared KB
    merged_kb = KnowledgeBase()
    for b in branches:
        for h in b.knowledge_base.hypotheses:
            merged_kb.hypotheses.append(h)
        for cat_name, cat in b.knowledge_base.categories.items():
            if cat_name not in merged_kb.categories:
                merged_kb.categories[cat_name] = cat
            else:
                existing = merged_kb.categories[cat_name]
                for insight in cat.confirmed_insights:
                    if insight not in existing.confirmed_insights:
                        existing.confirmed_insights.append(insight)
                for rej in cat.rejected_approaches:
                    if rej not in existing.rejected_approaches:
                        existing.rejected_approaches.append(rej)
        merged_kb.next_id = max(merged_kb.next_id, b.knowledge_base.next_id)

    # Collect all experiment history from branches
    all_history: list[IterationResult] = []
    for b in branches:
        all_history.extend(b.history)

    return LoopState(
        iteration=d["iteration"],
        current_template=best_branch.current_template,
        best_template=best_branch.best_template,
        best_metrics=best_branch.best_metrics,
        knowledge_base=merged_kb,
        captions=[_caption_from_dict(c) for c in d["captions"]],
        style_profile=_style_profile_from_dict(d["style_profile"]),
        fixed_references=[Path(p) for p in d.get("fixed_references", [])],
        experiment_history=all_history,
        plateau_counter=0,
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
