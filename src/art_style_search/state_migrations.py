"""Schema version constants and migration helpers for persisted state."""

from __future__ import annotations

from typing import Any

_SCHEMA_VERSION = (
    # v1: dino_similarity era, v2: vision+KB+rigorous, v3: changed_section+target_category, v4: direction metadata,
    # v5: medium-class + proportions prompt contract (invalidates cached style analyses, adds diagnostic vision dims),
    # v6: canon-first metric split (style_boilerplate_purity → style_canon_fidelity + observation_boilerplate_purity);
    #     adds vision style_gap notes + KnowledgeBase.style_gap_observations
    # v7: canon edit ledger — LoopState.canon_edit_ledger ring buffer recording cross-iteration canon edits +
    #     measured effect, rendered into the reasoner's brainstorm context as "Canon Edit History"
    7
)
_ITERATION_LOG_SCHEMA_VERSION = 1
_MANIFEST_SCHEMA_VERSION = 3
_PROMOTION_LOG_SCHEMA_VERSION = 1


def _migrate_metric_scores_payload(data: dict[str, Any]) -> dict[str, Any]:
    if "dreamsim_similarity" not in data and "dino_similarity" in data:
        data["dreamsim_similarity"] = data.pop("dino_similarity")
    # v5: diagnostic per-image vision dims
    data.setdefault("vision_medium", 0.5)
    data.setdefault("vision_proportions", 0.5)
    # v6: vision judge style_gap observation
    data.setdefault("style_gap", "")
    return data


def _migrate_aggregated_metrics_payload(data: dict[str, Any]) -> dict[str, Any]:
    if "dreamsim_similarity_mean" not in data and "dino_similarity_mean" in data:
        data["dreamsim_similarity_mean"] = data.pop("dino_similarity_mean")
    if "dreamsim_similarity_std" not in data and "dino_similarity_std" in data:
        data["dreamsim_similarity_std"] = data.pop("dino_similarity_std")
    # v5: diagnostic vision dims (neutral defaults preserve pre-upgrade semantics)
    data.setdefault("vision_medium", 0.5)
    data.setdefault("vision_medium_std", 0.0)
    data.setdefault("vision_proportions", 0.5)
    data.setdefault("vision_proportions_std", 0.0)
    # v6: canon-first metric split. Old style_boilerplate_purity measured the INVERSE (paraphrasing
    # rewarded). Drop it — the semantics flipped with the new captioner contract, so stale values
    # would mislead. Callers get neutral defaults for the new fields until the next evaluation.
    data.pop("style_boilerplate_purity", None)
    data.setdefault("style_canon_fidelity", 1.0)
    data.setdefault("observation_boilerplate_purity", 1.0)
    data.setdefault("style_gap_notes", [])
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
    data.setdefault("changed_sections", [data["changed_section"]] if data.get("changed_section") else [])
    data.setdefault("target_category", "")
    data.setdefault("direction_id", "")
    data.setdefault("direction_summary", "")
    data.setdefault("failure_mechanism", "")
    data.setdefault("intervention_type", "")
    data.setdefault("risk_level", "targeted")
    data.setdefault("expected_primary_metric", "")
    data.setdefault("expected_tradeoff", "")
    data.setdefault("vision_feedback", "")
    data.setdefault("roundtrip_feedback", "")
    data.setdefault("iteration_captions", [])
    return data


def _migrate_hypothesis_payload(data: dict[str, Any]) -> dict[str, Any]:
    data.setdefault("direction_id", "")
    data.setdefault("direction_summary", "")
    data.setdefault("failure_mechanism", "")
    data.setdefault("intervention_type", "")
    data.setdefault("risk_level", "targeted")
    data.setdefault("expected_primary_metric", "")
    data.setdefault("expected_tradeoff", "")
    data.setdefault("changed_sections", [])
    return data


def _migrate_category_progress_payload(data: dict[str, Any]) -> dict[str, Any]:
    if "best_perceptual_delta" not in data and "best_dino_delta" in data:
        data["best_perceptual_delta"] = data.pop("best_dino_delta")
    data.setdefault("last_mechanism_tried", "")
    data.setdefault("last_confirmed_mechanism", "")
    return data


def _migrate_knowledge_base_payload(data: dict[str, Any]) -> dict[str, Any]:
    categories = data.get("categories", {})
    if isinstance(categories, dict):
        data["categories"] = {
            k: _migrate_category_progress_payload(dict(v)) if isinstance(v, dict) else v for k, v in categories.items()
        }
    hypotheses = data.get("hypotheses", [])
    if isinstance(hypotheses, list):
        data["hypotheses"] = [_migrate_hypothesis_payload(dict(h)) if isinstance(h, dict) else h for h in hypotheses]
    # v6: style-gap observations ring buffer fed by the vision judge
    data.setdefault("style_gap_observations", [])
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
    if version < 4:
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
    if "knowledge_base" in data and isinstance(data["knowledge_base"], dict):
        data["knowledge_base"] = _migrate_knowledge_base_payload(dict(data["knowledge_base"]))
    # v7: canon edit ledger — backfill empty on pre-v7 payloads
    data.setdefault("canon_edit_ledger", [])
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
    if version < 2:
        data.setdefault("discovered_reference_count", 0)
    if version < 3:
        data.setdefault("comparison_provider", "gemini")
    return data


def _migrate_promotion_payload(raw: dict[str, Any], version: int) -> dict[str, Any]:
    data = dict(raw)
    if version < 1:
        data.setdefault("candidate_hypothesis", "")
        data.setdefault("replicate_scores", None)
        data.setdefault("p_value", None)
        data.setdefault("test_statistic", None)
    return data
