"""Persistence layer façade for state, logs, manifests, and promotion records."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from art_style_search.state_codec import (
    _aggregated_metrics_from_dict,
    _caption_from_dict,
    _Encoder,
    _iteration_result_from_dict,
    _knowledge_base_from_dict,
    _loop_state_from_dict,
    _metric_scores_from_dict,
    _prompt_section_from_dict,
    prompt_template_from_dict,
    style_profile_from_dict,
    to_dict,
)
from art_style_search.state_migrations import (
    _ITERATION_LOG_SCHEMA_VERSION,
    _MANIFEST_SCHEMA_VERSION,
    _PROMOTION_LOG_SCHEMA_VERSION,
    _SCHEMA_VERSION,
    _migrate_iteration_log_payload,
    _migrate_manifest_payload,
    _migrate_promotion_payload,
    _migrate_state_payload,
)
from art_style_search.types import IterationResult, LoopState, PromotionDecision, RunManifest

logger = logging.getLogger(__name__)

__all__ = [
    "_aggregated_metrics_from_dict",
    "_caption_from_dict",
    "_iteration_result_from_dict",
    "_knowledge_base_from_dict",
    "_loop_state_from_dict",
    "_metric_scores_from_dict",
    "_prompt_section_from_dict",
    "append_promotion_log",
    "load_iteration_log",
    "load_manifest",
    "load_promotion_log",
    "load_state",
    "prompt_template_from_dict",
    "save_iteration_log",
    "save_manifest",
    "save_state",
    "style_profile_from_dict",
    "to_dict",
]


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
