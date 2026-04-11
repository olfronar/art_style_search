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
    PromotionDecision,
    PromptSection,
    PromptTemplate,
    RunManifest,
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
            return to_dict(o)
        if isinstance(o, enum.Enum):
            return o.value
        return super().default(o)


def to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass (or nested structure) to a JSON-safe dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            result[f.name] = to_dict(getattr(obj, f.name))
        return result
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
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
        is_fallback=d.get("is_fallback", False),
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
        completion_rate=d.get("completion_rate", 1.0),
    )


def _prompt_section_from_dict(d: dict[str, Any]) -> PromptSection:
    return PromptSection(name=d["name"], description=d["description"], value=d["value"])


def prompt_template_from_dict(d: dict[str, Any]) -> PromptTemplate:
    return PromptTemplate(
        sections=[_prompt_section_from_dict(s) for s in d.get("sections", [])],
        negative_prompt=d.get("negative_prompt"),
        caption_sections=d.get("caption_sections", []),
        caption_length_target=d.get("caption_length_target", 0),
    )


def style_profile_from_dict(d: dict[str, Any]) -> StyleProfile:
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
        template=prompt_template_from_dict(d["template"]),
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
        n_images_attempted=d.get("n_images_attempted", 0),
        n_images_succeeded=d.get("n_images_succeeded", 0),
        changed_section=d.get("changed_section", ""),
        target_category=d.get("target_category", ""),
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
        current_template=prompt_template_from_dict(d["current_template"]),
        best_template=prompt_template_from_dict(d["best_template"]),
        best_metrics=_aggregated_metrics_from_dict(d["best_metrics"]) if d.get("best_metrics") is not None else None,
        knowledge_base=_knowledge_base_from_dict(d["knowledge_base"]) if d.get("knowledge_base") else KnowledgeBase(),
        captions=[_caption_from_dict(c) for c in d["captions"]],
        style_profile=style_profile_from_dict(d["style_profile"]),
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
        seed=d.get("seed", 0),
        protocol=d.get("protocol", "classic"),
        feedback_refs=[Path(p) for p in d.get("feedback_refs", [])],
        silent_refs=[Path(p) for p in d.get("silent_refs", [])],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_SCHEMA_VERSION = 3  # v1: dino_similarity era, v2: vision+KB+rigorous, v3: changed_section+target_category
_ITERATION_LOG_SCHEMA_VERSION = 1
_MANIFEST_SCHEMA_VERSION = 1
_PROMOTION_LOG_SCHEMA_VERSION = 1


def _migrate_metric_scores_payload(data: dict[str, Any]) -> dict[str, Any]:
    if "dreamsim_similarity" not in data and "dino_similarity" in data:
        data["dreamsim_similarity"] = data.pop("dino_similarity")
    return data


def _migrate_aggregated_metrics_payload(data: dict[str, Any]) -> dict[str, Any]:
    if "dreamsim_similarity_mean" not in data and "dino_similarity_mean" in data:
        data["dreamsim_similarity_mean"] = data.pop("dino_similarity_mean")
    if "dreamsim_similarity_std" not in data and "dino_similarity_std" in data:
        data["dreamsim_similarity_std"] = data.pop("dino_similarity_std")
    return data


def _migrate_iteration_result_payload(data: dict[str, Any]) -> dict[str, Any]:
    if "aggregated" in data and isinstance(data["aggregated"], dict):
        data["aggregated"] = _migrate_aggregated_metrics_payload(dict(data["aggregated"]))
    if "per_image_scores" in data and isinstance(data["per_image_scores"], list):
        data["per_image_scores"] = [
            _migrate_metric_scores_payload(dict(score)) if isinstance(score, dict) else score
            for score in data["per_image_scores"]
        ]
    data.setdefault("changed_section", "")
    data.setdefault("target_category", "")
    data.setdefault("vision_feedback", "")
    data.setdefault("roundtrip_feedback", "")
    data.setdefault("iteration_captions", [])
    return data


def _migrate_state_payload(raw: dict[str, Any], version: int) -> dict[str, Any]:
    data = dict(raw)
    if version < 2:
        data.setdefault("knowledge_base", {})
        data.setdefault("review_feedback", "")
        data.setdefault("pairwise_feedback", "")
        data.setdefault("protocol", "classic")
        data.setdefault("feedback_refs", [])
        data.setdefault("silent_refs", [])
    if version < 3:
        results = data.get("experiment_history", [])
        data["experiment_history"] = [
            _migrate_iteration_result_payload(dict(result)) if isinstance(result, dict) else result
            for result in results
        ]
        last_results = data.get("last_iteration_results", [])
        data["last_iteration_results"] = [
            _migrate_iteration_result_payload(dict(result)) if isinstance(result, dict) else result
            for result in last_results
        ]
    if "best_metrics" in data and isinstance(data["best_metrics"], dict):
        data["best_metrics"] = _migrate_aggregated_metrics_payload(dict(data["best_metrics"]))
    if "global_best_metrics" in data and isinstance(data["global_best_metrics"], dict):
        data["global_best_metrics"] = _migrate_aggregated_metrics_payload(dict(data["global_best_metrics"]))
    return data


def _migrate_iteration_log_payload(raw: dict[str, Any], version: int) -> dict[str, Any]:
    data = dict(raw)
    if version < 1:
        data = _migrate_iteration_result_payload(data)
    return data


def _migrate_manifest_payload(raw: dict[str, Any], version: int) -> dict[str, Any]:
    data = dict(raw)
    if version < 1:
        data.setdefault("uv_lock_hash", None)
    return data


def _migrate_promotion_payload(raw: dict[str, Any], version: int) -> dict[str, Any]:
    data = dict(raw)
    if version < 1:
        data.setdefault("candidate_hypothesis", "")
        data.setdefault("replicate_scores", None)
        data.setdefault("p_value", None)
        data.setdefault("test_statistic", None)
    return data


def save_state(state: LoopState, path: Path) -> None:
    """Serialize *state* to JSON, writing atomically via temp-file + rename."""
    data = to_dict(state)
    # Transient fields — always empty on resume, no need to persist.
    data.pop("review_feedback", None)
    data.pop("pairwise_feedback", None)
    data["_schema_version"] = _SCHEMA_VERSION
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
    version = raw.pop("_schema_version", 1)
    logger.info("State schema version: %d (current: %d)", version, _SCHEMA_VERSION)
    return _loop_state_from_dict(_migrate_state_payload(raw, version))


def save_iteration_log(result: IterationResult, log_dir: Path) -> None:
    """Write a single iteration result to ``{log_dir}/iter_{NNN}_branch_{M}.json``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"iter_{result.iteration:03d}_branch_{result.branch_id}.json"
    log_path = log_dir / filename

    data = to_dict(result)
    data["_schema_version"] = _ITERATION_LOG_SCHEMA_VERSION
    log_path.write_text(json.dumps(data, cls=_Encoder, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Iteration log written to %s", log_path)


def load_iteration_log(path: Path) -> IterationResult:
    """Load a single iteration-log JSON file into an ``IterationResult``.

    Inverse of :func:`save_iteration_log`.  Used by the HTML report generator
    (``art_style_search.report``) to read all per-experiment logs for a run.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.pop("_schema_version", 0)
    return _iteration_result_from_dict(_migrate_iteration_log_payload(raw, version))


# ---------------------------------------------------------------------------
# Run manifest (write-once provenance record)
# ---------------------------------------------------------------------------


def save_manifest(manifest: RunManifest, path: Path) -> None:
    """Write the run manifest to *path* (JSON, write-once)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = to_dict(manifest)
    data["_schema_version"] = _MANIFEST_SCHEMA_VERSION
    path.write_text(json.dumps(data, cls=_Encoder, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Run manifest written to %s", path)


def load_manifest(path: Path) -> RunManifest | None:
    """Load a previously saved run manifest, or return None if not found."""
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.pop("_schema_version", 0)
    raw = _migrate_manifest_payload(raw, version)
    return RunManifest(
        protocol_version=raw.get("protocol_version", "classic"),
        seed=raw.get("seed", 0),
        cli_args=raw.get("cli_args", {}),
        model_names=raw.get("model_names", {}),
        reasoning_provider=raw.get("reasoning_provider", ""),
        git_sha=raw.get("git_sha"),
        python_version=raw.get("python_version", ""),
        platform=raw.get("platform", ""),
        timestamp_utc=raw.get("timestamp_utc", ""),
        reference_image_hashes=raw.get("reference_image_hashes", {}),
        num_fixed_refs=raw.get("num_fixed_refs", 0),
        uv_lock_hash=raw.get("uv_lock_hash"),
    )


# ---------------------------------------------------------------------------
# Promotion decision log (append-only JSONL)
# ---------------------------------------------------------------------------


def append_promotion_log(decision: PromotionDecision, path: Path) -> None:
    """Append one promotion decision as a JSON line to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict(decision)
    payload["_schema_version"] = _PROMOTION_LOG_SCHEMA_VERSION
    line = json.dumps(payload, cls=_Encoder, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_promotion_log(path: Path) -> list[PromotionDecision]:
    """Load all promotion decisions from a JSONL file."""
    if not path.exists():
        return []
    decisions: list[PromotionDecision] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        version = d.pop("_schema_version", 0)
        d = _migrate_promotion_payload(d, version)
        decisions.append(
            PromotionDecision(
                iteration=d["iteration"],
                candidate_score=d["candidate_score"],
                baseline_score=d["baseline_score"],
                epsilon=d["epsilon"],
                delta=d["delta"],
                decision=d["decision"],
                reason=d["reason"],
                candidate_branch_id=d["candidate_branch_id"],
                candidate_hypothesis=d.get("candidate_hypothesis", ""),
                replicate_scores=d.get("replicate_scores"),
                p_value=d.get("p_value"),
                test_statistic=d.get("test_statistic"),
            )
        )
    return decisions
