"""Main orchestration loop with experiment-based optimization.

The loop optimizes a **meta-prompt** — instructions for how to caption images
so that the captions can recreate the originals via image generation.

Each iteration:
1. The reasoning model proposes N diverse experiments (hypothesis-driven template variants)
2. Each experiment: caption + generate + evaluate in parallel
3. Top experiments are synthesised; pairwise vision comparison and independent review run
4. Best experiment updates the shared state; all results feed into the Knowledge Base
5. Check convergence (plateau / reasoning-model stop / max iterations)

``run()`` is intentionally thin — it orchestrates a pipeline of named phase
helpers (``_setup_run_context`` → ``_zero_step`` → per-iteration pipeline →
``_finalize_run``).  Each helper owns one responsibility and can be audited
in isolation.  The per-iteration helpers communicate via a small
``IterationRanking`` dataclass so no mutable state is captured through
closure.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import math
import platform as _platform
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from google import genai

from art_style_search.analyze import analyze_style
from art_style_search.caption import caption_references
from art_style_search.config import Config
from art_style_search.evaluate import pairwise_compare_experiments
from art_style_search.experiment import (
    ExperimentProposal,
    best_kept_result,
    collect_experiment_results,
    run_experiment,
)
from art_style_search.knowledge import IterationDecision, build_caption_diffs, update_knowledge_base
from art_style_search.models import ModelRegistry
from art_style_search.prompt import (
    Lessons,
    enforce_hypothesis_diversity,
    propose_experiments,
    propose_initial_templates,
    review_iteration,
    synthesize_templates,
    validate_template,
)
from art_style_search.scoring import adaptive_composite_score, composite_score, improvement_epsilon
from art_style_search.state import (
    append_promotion_log,
    load_manifest,
    load_state,
    save_iteration_log,
    save_manifest,
    save_state,
)
from art_style_search.types import (
    AggregatedMetrics,
    ConvergenceReason,
    IterationResult,
    KnowledgeBase,
    LoopState,
    PromotionDecision,
    PromotionTestResult,
    PromptTemplate,
    RunManifest,
)
from art_style_search.utils import IMAGE_EXTENSIONS, ReasoningClient, build_ref_gen_pairs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _discover_images(directory: Path) -> list[Path]:
    """Find all image files in a directory, sorted for determinism."""
    paths = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    paths.sort()
    return paths


def _sample(items: list[Path], max_count: int, rng: random.Random | None = None) -> list[Path]:
    """Random sample up to max_count items from a list."""
    if len(items) <= max_count:
        return items
    if rng is not None:
        return rng.sample(items, max_count)
    return random.sample(items, max_count)


def _apply_best_result(state: LoopState, result: IterationResult) -> None:
    """Update state with a genuine improvement — updates everything including global best."""
    result.kept = True
    state.current_template = result.template
    state.best_template = result.template
    state.best_metrics = result.aggregated
    global_score = composite_score(state.global_best_metrics) if state.global_best_metrics else float("-inf")
    if composite_score(result.aggregated) > global_score:
        state.global_best_prompt = result.rendered_prompt
        state.global_best_metrics = result.aggregated


def _apply_exploration_result(state: LoopState, result: IterationResult) -> None:
    """Adopt a result for exploration — guides next proposals but preserves improvement baseline."""
    result.kept = True
    state.current_template = result.template
    state.best_template = result.template


def _save_best_prompt(state: LoopState, log_dir: Path) -> None:
    """Write the best meta-prompt to a standalone file for easy access."""
    if not state.global_best_prompt:
        return
    prompt_file = log_dir / "best_prompt.txt"
    prompt_file.write_text(state.global_best_prompt, encoding="utf-8")
    logger.info("Best meta-prompt saved to %s", prompt_file)


def _log_experiment_results(results: list[IterationResult], log_dir: Path) -> None:
    """Save and log each experiment result."""
    for r in results:
        save_iteration_log(r, log_dir)
        m = r.aggregated
        logger.info(
            "Exp %d — DS=%.3f Color=%.3f SSIM=%.3f HPS=%.3f Aes=%.1f V[S=%.2f Su=%.2f Co=%.2f] %s",
            r.branch_id,
            m.dreamsim_similarity_mean,
            m.color_histogram_mean,
            m.ssim_mean,
            m.hps_score_mean,
            m.aesthetics_score_mean,
            m.vision_style,
            m.vision_subject,
            m.vision_composition,
            "KEPT" if r.kept else "discarded",
        )


def _filter_feedback_by_refs(feedback_text: str, feedback_refs: frozenset[Path]) -> str:
    """Filter multi-line per-image feedback to include only lines mentioning feedback_ref filenames."""
    if not feedback_text or not feedback_refs:
        return feedback_text
    ref_names = {p.name for p in feedback_refs}
    lines = feedback_text.split("\n")
    kept: list[str] = []
    keep_current = True
    for line in lines:
        # Section headers (##) and empty lines are always kept
        if line.startswith("##") or not line.strip():
            kept.append(line)
            keep_current = True
            continue
        # Lines starting with ** or "Image (" are per-image — check if it matches a feedback ref
        if line.startswith("**") or line.startswith("Image ("):
            keep_current = any(name in line for name in ref_names)
        if keep_current:
            kept.append(line)
    return "\n".join(kept)


# Max experiment results to persist in state.json (older ones are in iteration logs)
_MAX_PERSISTED_HISTORY = 30

# Exploration (plateau-escape) tuning: once we've plateaued for this many
# iterations, adopt the second-best experiment every ``_EXPLORATION_CADENCE``
# plateau ticks, to escape local optima without abandoning the baseline.
_EXPLORATION_MIN_PLATEAU = 2
_EXPLORATION_CADENCE = 2
# After adopting a second-best for exploration, reset plateau to this value
# (not 0) so the plateau window can still terminate the run if exploration
# doesn't help.
_EXPLORATION_RESET_PLATEAU = 1

# Minimum fraction of ``max_iterations`` that must elapse before the loop
# honors a ``[CONVERGED]`` signal from the reasoning model. Prevents 2-flat-
# iteration runs from self-terminating before exploring untried directions.
_MIN_ITER_FRACTION_FOR_STOP = 0.5


def _ref_cache_key(paths: list[Path]) -> str:
    """Deterministic hash from sorted reference paths + mtimes for cross-run caching."""
    parts = sorted(f"{p}:{p.stat().st_mtime}" for p in paths)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _validate_template_or_raise(template: PromptTemplate, *, context: str) -> None:
    """Raise a RuntimeError when a model-produced template violates required invariants."""
    errors = validate_template(template)
    if not errors:
        return
    msg = f"{context} produced invalid template: {'; '.join(errors)}"
    raise RuntimeError(msg)


def _sanitize_initial_templates(
    templates: list[PromptTemplate],
    *,
    fallback: PromptTemplate,
) -> list[PromptTemplate]:
    """Replace empty or invalid initial templates with the validated compiled fallback."""
    validated: list[PromptTemplate] = []
    for i, template in enumerate(templates):
        errors = validate_template(template)
        if template.sections and not errors:
            validated.append(template)
            continue
        if errors:
            logger.warning("Initial template %d invalid — falling back: %s", i, "; ".join(errors))
        validated.append(fallback)
    return validated


def _candidate_results_for_validation(ranking: IterationRanking) -> list[IterationResult]:
    """Return the top proposal candidates plus synthesis candidate, if present."""
    proposal_results = [
        result for result in ranking.exp_results if ranking.synth_result is None or result is not ranking.synth_result
    ]
    sorted_proposals = sorted(proposal_results, key=lambda r: ranking.adaptive_scores[id(r)], reverse=True)
    candidates = sorted_proposals[:2]
    if ranking.synth_result is not None and ranking.synth_result not in candidates:
        candidates.append(ranking.synth_result)
    return candidates


def _hash_reference_images(ref_dir: Path) -> dict[str, str]:
    """SHA-256 hash every image in *ref_dir*, keyed by filename."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in _discover_images(ref_dir)}


def _build_manifest(config: Config) -> RunManifest:
    """Build a RunManifest from the current config and environment."""

    git_sha: str | None = None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    uv_lock_hash: str | None = None
    uv_lock = Path("uv.lock")
    if uv_lock.exists():
        uv_lock_hash = hashlib.sha256(uv_lock.read_bytes()).hexdigest()

    ref_hashes = _hash_reference_images(config.reference_dir)

    protocol_version = "rigorous_v1" if config.protocol == "rigorous" else "classic"

    return RunManifest(
        protocol_version=protocol_version,
        seed=config.seed,
        cli_args={
            "max_iterations": config.max_iterations,
            "plateau_window": config.plateau_window,
            "num_branches": config.num_branches,
            "aspect_ratio": config.aspect_ratio,
            "num_fixed_refs": config.num_fixed_refs,
            "protocol": config.protocol,
        },
        model_names={
            "caption_model": config.caption_model,
            "generator_model": config.generator_model,
            "reasoning_model": config.reasoning_model,
        },
        reasoning_provider=config.reasoning_provider,
        git_sha=git_sha,
        python_version=sys.version,
        platform=_platform.platform(),
        timestamp_utc=datetime.now(UTC).isoformat(),
        reference_image_hashes=ref_hashes,
        num_fixed_refs=config.num_fixed_refs,
        uv_lock_hash=uv_lock_hash,
    )


def _verify_manifest(config: Config, manifest: RunManifest) -> None:
    """Verify on resume that the manifest matches current config. Warn or abort."""
    if config.protocol == "rigorous":
        if manifest.protocol_version != "rigorous_v1":
            msg = f"Protocol mismatch: state has '{manifest.protocol_version}', CLI has 'rigorous'"
            raise RuntimeError(msg)
        if manifest.seed != config.seed:
            msg = f"Seed mismatch: manifest has {manifest.seed}, CLI has {config.seed}"
            raise RuntimeError(msg)
        # Verify reference image hashes
        if _hash_reference_images(config.reference_dir) != manifest.reference_image_hashes:
            msg = "Reference images changed since run started — aborting resume in rigorous mode"
            raise RuntimeError(msg)
    else:
        if manifest.seed != config.seed:
            logger.warning("Seed drift: manifest=%d, CLI=%d", manifest.seed, config.seed)
        if manifest.protocol_version != ("rigorous_v1" if config.protocol == "rigorous" else "classic"):
            logger.warning("Protocol drift: manifest=%s, CLI=%s", manifest.protocol_version, config.protocol)


def _split_information_barrier(
    fixed_refs: list[Path], protocol: str, rng: random.Random
) -> tuple[list[Path], list[Path]]:
    """Split fixed refs into feedback (shown to reasoning model) and silent (hidden).

    In rigorous mode: ceil(0.7 * N) feedback, rest silent (min 2 silent).
    In classic mode: all feedback, none silent.
    """

    if protocol != "rigorous" or len(fixed_refs) < 4:
        return list(fixed_refs), []
    n_feedback = math.ceil(0.7 * len(fixed_refs))
    n_silent = len(fixed_refs) - n_feedback
    if n_silent < 2:
        n_feedback = len(fixed_refs) - 2
    shuffled = list(fixed_refs)
    rng.shuffle(shuffled)
    return shuffled[:n_feedback], shuffled[n_feedback:]


def _log_promotion_decision(
    state: LoopState,
    ranking: IterationRanking,
    decision: str,
    reason: str,
    config: Config,
    *,
    candidate: IterationResult | None = None,
    candidate_score: float | None = None,
    replicate_scores: list[float] | None = None,
    promotion_test: PromotionTestResult | None = None,
) -> None:
    """Log a promotion decision to promotion_log.jsonl."""
    selected = candidate or ranking.best_exp
    score = candidate_score if candidate_score is not None else composite_score(selected.aggregated)
    test_result = (
        promotion_test
        if promotion_test is not None
        else (ranking.promotion_test if selected is ranking.best_exp else None)
    )
    pd = PromotionDecision(
        iteration=state.iteration,
        candidate_score=score,
        baseline_score=ranking.baseline_score,
        epsilon=ranking.epsilon,
        delta=score - ranking.baseline_score,
        decision=decision,
        reason=reason,
        candidate_branch_id=selected.branch_id,
        candidate_hypothesis=selected.hypothesis,
        replicate_scores=replicate_scores,
        p_value=test_result.p_value if test_result is not None else None,
        test_statistic=test_result.statistic if test_result is not None else None,
    )
    append_promotion_log(pd, config.run_dir / "promotion_log.jsonl")


# ---------------------------------------------------------------------------
# Per-run and per-iteration data carriers
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Immutable per-run dependencies passed to iteration helpers.

    Holds clients, semaphores, registry, and config so per-phase helpers
    don't need 8-parameter signatures.  ``fixed_refs`` lives on ``LoopState``
    (authoritative, persisted) and is not duplicated here.
    """

    config: Config
    gemini_client: genai.Client
    reasoning_client: ReasoningClient
    registry: ModelRegistry
    gemini_semaphore: asyncio.Semaphore
    eval_semaphore: asyncio.Semaphore
    rng: random.Random = field(default_factory=random.Random)


@dataclass
class IterationRanking:
    """Ranking state that crosses phase boundaries within one iteration.

    Bundled so phase-3 → 3.5 → 3.7 → 4 can pass it explicitly instead of
    capturing locals via closure.  ``_run_synthesis_experiment`` appends
    to ``exp_results`` and stores its own output on ``synth_result``; every
    field here is required at construction time except ``synth_result``,
    which stays ``None`` when synthesis is skipped or fails.
    """

    exp_results: list[IterationResult]
    adaptive_scores: dict[int, float]
    best_exp: IterationResult
    best_score: float
    baseline_score: float
    epsilon: float
    synth_result: IterationResult | None = None
    promotion_test: PromotionTestResult | None = None
    best_replicate_scores: list[float] | None = None


# ---------------------------------------------------------------------------
# Phase 3.7 + 3.9 helpers (extracted to avoid B023 loop-variable capture)
# ---------------------------------------------------------------------------


async def _run_pairwise_comparison(
    ranking: IterationRanking,
    state: LoopState,
    ctx: RunContext,
) -> None:
    """Phase 3.7: SPO-inspired pairwise comparison of top experiments."""
    exp_results = ranking.exp_results
    if len(exp_results) < 2:
        state.pairwise_feedback = ""
        return
    sorted_by_score = sorted(exp_results, key=lambda r: ranking.adaptive_scores[id(r)], reverse=True)
    top_a, top_b = sorted_by_score[0], sorted_by_score[1]
    pairs_a = build_ref_gen_pairs(top_a)
    pairs_b = build_ref_gen_pairs(top_b)
    # Information barrier: only send feedback_refs pairs to pairwise comparison
    if state.silent_refs:
        feedback_set = frozenset(state.feedback_refs)
        pairs_a = [(ref, gen) for ref, gen in pairs_a if ref in feedback_set]
        pairs_b = [(ref, gen) for ref, gen in pairs_b if ref in feedback_set]
    if not pairs_a or not pairs_b:
        state.pairwise_feedback = ""
        return
    pairwise_rationale, pairwise_score = await pairwise_compare_experiments(
        pairs_a,
        pairs_b,
        client=ctx.gemini_client,
        model=ctx.config.caption_model,
        semaphore=ctx.gemini_semaphore,
    )
    winner = "A" if pairwise_score > 0.5 else "B" if pairwise_score < 0.5 else "TIE"
    logger.info(
        "Pairwise: Exp %d vs Exp %d → %s (%s)",
        top_a.branch_id,
        top_b.branch_id,
        winner,
        pairwise_rationale[:100],
    )
    state.pairwise_feedback = (
        f"Top experiment {top_a.branch_id} vs runner-up {top_b.branch_id}: Winner={winner}. {pairwise_rationale}"
    )


async def _run_independent_review(
    ranking: IterationRanking,
    proposals: list[ExperimentProposal],
    state: LoopState,
    ctx: RunContext,
) -> None:
    """Phase 3.9: CycleResearcher-inspired independent review."""
    review = await review_iteration(
        experiments=ranking.exp_results,
        proposals=proposals,
        baseline_metrics=state.best_metrics,
        knowledge_base=state.knowledge_base,
        client=ctx.reasoning_client,
        model=ctx.config.reasoning_model,
    )
    state.review_feedback = review.strategic_guidance
    if review.strategic_guidance:
        logger.info("Review guidance: %.200s", review.strategic_guidance)


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------


async def _setup_run_context(config: Config) -> RunContext:
    """Configure logging, executor, clients, semaphores and load eval models."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=config.eval_concurrency))

    gemini_client = genai.Client(api_key=config.google_api_key)
    reasoning_client = ReasoningClient(
        config.reasoning_provider,
        anthropic_api_key=config.anthropic_api_key,
        zai_api_key=config.zai_api_key,
        openai_api_key=config.openai_api_key,
        base_url=config.reasoning_base_url,
    )

    gemini_semaphore = asyncio.Semaphore(config.gemini_concurrency)
    eval_semaphore = asyncio.Semaphore(config.eval_concurrency)

    rng = random.Random(config.seed)
    logger.info("Run seed: %d, protocol: %s", config.seed, config.protocol)

    logger.info("Loading evaluation models...")
    registry = await asyncio.to_thread(ModelRegistry.load_all)

    return RunContext(
        config=config,
        gemini_client=gemini_client,
        reasoning_client=reasoning_client,
        registry=registry,
        gemini_semaphore=gemini_semaphore,
        eval_semaphore=eval_semaphore,
        rng=rng,
    )


def _finalize_run(state: LoopState, ctx: RunContext) -> LoopState:
    """Persist final state, write best prompt, and log the summary banner."""
    save_state(state, ctx.config.state_file)
    _save_best_prompt(state, ctx.config.log_dir)

    # Write holdout summary for silent images (information barrier)
    _write_holdout_summary(state, ctx)

    logger.info("=" * 60)
    if state.global_best_metrics:
        m = state.global_best_metrics
        logger.info(
            "FINAL BEST — DS=%.4f HPS=%.4f Aes=%.2f",
            m.dreamsim_similarity_mean,
            m.hps_score_mean,
            m.aesthetics_score_mean,
        )
    logger.info("BEST META-PROMPT: %s", state.global_best_prompt)
    logger.info("Convergence: %s", state.convergence_reason)
    logger.info("Total experiments: %d", len(state.experiment_history))
    logger.info("KB: %d hypotheses", len(state.knowledge_base.hypotheses))

    return state


def _extract_silent_scores(results: list[IterationResult], silent_set: frozenset[Path]) -> list[float]:
    """Extract per-image composite scores for silent images from results."""
    from art_style_search.scoring import per_image_composite

    scores: list[float] = []
    for r in results:
        for cap, sc in zip(r.iteration_captions, r.per_image_scores, strict=False):
            if cap.image_path in silent_set:
                scores.append(per_image_composite(sc))
    return scores


def _write_holdout_summary(state: LoopState, ctx: RunContext) -> None:
    """Compute and save holdout summary for silent images."""

    if not state.silent_refs:
        return

    silent_set = frozenset(state.silent_refs)

    # Iteration 0 scores (from experiment_history or last_iteration_results)
    iter0_results = [r for r in state.experiment_history if r.iteration == 0 and r.kept]
    final_results = [r for r in state.last_iteration_results if r.kept]
    if not final_results:
        final_results = state.last_iteration_results[:1]

    iter0_scores = _extract_silent_scores(iter0_results, silent_set)
    final_scores = _extract_silent_scores(final_results, silent_set)

    summary = {
        "silent_image_count": len(state.silent_refs),
        "silent_image_names": [p.name for p in state.silent_refs],
        "iteration_0_mean": sum(iter0_scores) / len(iter0_scores) if iter0_scores else None,
        "final_mean": sum(final_scores) / len(final_scores) if final_scores else None,
        "iteration_0_scores": iter0_scores,
        "final_scores": final_scores,
    }
    if summary["iteration_0_mean"] is not None and summary["final_mean"] is not None:
        summary["delta"] = summary["final_mean"] - summary["iteration_0_mean"]

    holdout_path = ctx.config.run_dir / "holdout_summary.json"
    holdout_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Holdout summary written to %s", holdout_path)


# ---------------------------------------------------------------------------
# Zero-step (fix refs → caption → analyze → initial templates → eval → apply)
# ---------------------------------------------------------------------------


async def _zero_step(ctx: RunContext, all_ref_paths: list[Path]) -> LoopState:
    """Zero-step: fix refs, caption, analyze style, propose N initial templates, evaluate, apply best."""
    config = ctx.config
    fixed_refs = _sample(all_ref_paths, config.num_fixed_refs, rng=ctx.rng)
    logger.info("Fixed %d reference images for optimization", len(fixed_refs))

    # Information barrier: split into feedback (visible to reasoning model) and silent
    feedback_refs, silent_refs = _split_information_barrier(fixed_refs, config.protocol, ctx.rng)
    if silent_refs:
        logger.info("Information barrier: %d feedback + %d silent images", len(feedback_refs), len(silent_refs))

    logger.info("Zero-step: captioning %d reference images...", len(fixed_refs))
    captions = await caption_references(
        fixed_refs,
        model=config.caption_model,
        client=ctx.gemini_client,
        cache_dir=config.log_dir / "captions",
        semaphore=ctx.gemini_semaphore,
        cache_key="initial",
    )

    logger.info("Zero-step: analyzing art style...")
    # Shared cache keyed by reference images — survives across runs with the same refs
    shared_cache_dir = config.run_dir.parent / ".cache"
    shared_cache_dir.mkdir(parents=True, exist_ok=True)
    shared_cache = shared_cache_dir / f"style_{_ref_cache_key(fixed_refs)}.json"
    run_cache = config.log_dir / "style_profile.json"

    style_profile, initial_template = await analyze_style(
        fixed_refs,
        captions,
        gemini_client=ctx.gemini_client,
        reasoning_client=ctx.reasoning_client,
        caption_model=config.caption_model,
        reasoning_model=config.reasoning_model,
        cache_path=shared_cache,
    )
    _validate_template_or_raise(initial_template, context="Zero-step compiled template")

    # Copy into run dir for provenance
    if shared_cache.exists() and not run_cache.exists():
        shutil.copy2(shared_cache, run_cache)

    logger.info("Zero-step: proposing %d initial meta-prompts...", config.num_branches)
    initial_templates = await propose_initial_templates(
        style_profile,
        config.num_branches,
        client=ctx.reasoning_client,
        model=config.reasoning_model,
    )
    initial_templates = _sanitize_initial_templates(initial_templates, fallback=initial_template)

    state = LoopState(
        iteration=0,
        current_template=initial_templates[0],
        best_template=initial_templates[0],
        best_metrics=None,
        knowledge_base=KnowledgeBase(),
        captions=captions,
        style_profile=style_profile,
        fixed_references=fixed_refs,
        seed=config.seed,
        protocol=config.protocol,
        feedback_refs=feedback_refs,
        silent_refs=silent_refs,
    )

    # Save state before evaluation so a crash doesn't lose analysis work
    save_state(state, config.state_file)

    # Evaluate initial templates as iteration 0
    logger.info("=== Iteration 0 — evaluating %d initial templates ===", len(initial_templates))
    try:
        init_tasks = [
            run_experiment(
                experiment_id=i,
                template=t,
                iteration=0,
                fixed_refs=fixed_refs,
                config=ctx.config,
                gemini_client=ctx.gemini_client,
                registry=ctx.registry,
                gemini_semaphore=ctx.gemini_semaphore,
                eval_semaphore=ctx.eval_semaphore,
                last_results=[],
                hypothesis=f"Initial template {i}",
                experiment_desc="Zero-step diverse template",
            )
            for i, t in enumerate(initial_templates)
        ]

        init_results = collect_experiment_results(
            await asyncio.gather(*init_tasks, return_exceptions=True), "Initial experiment"
        )
    except Exception:
        logger.exception("Zero-step evaluation failed — partial state saved for resume")
        raise

    if init_results:
        best_init = max(init_results, key=lambda r: composite_score(r.aggregated))
        _apply_best_result(state, best_init)
        state.last_iteration_results = init_results
        state.experiment_history = list(init_results)
        _log_experiment_results(init_results, config.log_dir)

    state.iteration = 1
    save_state(state, config.state_file)
    _save_best_prompt(state, config.log_dir)
    return state


# ---------------------------------------------------------------------------
# Per-iteration phases
# ---------------------------------------------------------------------------


def _build_iteration_context(state: LoopState) -> tuple[str, str, str]:
    """Phase 0 of an iteration: build (vision_fb, roundtrip_fb, caption_diffs).

    Injects review and pairwise feedback from the previous iteration into the
    feedback strings so the reasoning model can incorporate them.
    """
    best_last = best_kept_result(state.last_iteration_results)
    vision_fb = best_last.vision_feedback if best_last else ""
    roundtrip_fb = best_last.roundtrip_feedback if best_last else ""

    # Information barrier: filter per-image feedback to feedback_refs only
    feedback_set = frozenset(state.feedback_refs) if state.silent_refs else None
    if feedback_set and best_last:
        vision_fb = _filter_feedback_by_refs(vision_fb, feedback_set)
        roundtrip_fb = _filter_feedback_by_refs(roundtrip_fb, feedback_set)

    caption_diffs = ""
    if best_last and best_last.iteration_captions:
        captions_for_diff = best_last.iteration_captions
        scores_for_diff = best_last.per_image_scores
        # Information barrier: only show caption diffs for feedback images
        if feedback_set:
            paired = [
                (c, s) for c, s in zip(captions_for_diff, scores_for_diff, strict=False) if c.image_path in feedback_set
            ]
            captions_for_diff = [c for c, _ in paired]
            scores_for_diff = [s for _, s in paired]
        sorted_caps = sorted(
            zip(captions_for_diff, scores_for_diff, strict=False),
            key=lambda x: x[1].dreamsim_similarity,
        )
        worst_caps = [c for c, _ in sorted_caps[:3]]
        caption_diffs = build_caption_diffs(state.prev_best_captions, worst_caps)

    if state.review_feedback:
        roundtrip_fb = f"## Independent Review of Last Iteration\n{state.review_feedback}\n\n{roundtrip_fb}"
    if state.pairwise_feedback:
        vision_fb = f"## Pairwise Experiment Comparison\n{state.pairwise_feedback}\n\n{vision_fb}"

    # Clear consumed feedback so it doesn't bleed into iteration N+2 if the
    # upcoming pairwise/review phases fail to produce fresh guidance.
    state.review_feedback = ""
    state.pairwise_feedback = ""

    return vision_fb, roundtrip_fb, caption_diffs


async def _propose_iteration_experiments(
    state: LoopState,
    ctx: RunContext,
    vision_fb: str,
    roundtrip_fb: str,
    caption_diffs: str,
) -> tuple[list[ExperimentProposal], bool]:
    """Phase 1: propose N experiments, dedup by category, convert to ExperimentProposal.

    Returns ``(proposals, converged)``.  ``converged`` is True when the
    reasoning model signalled stop via ``<CONVERGED>`` or no valid proposals
    parsed.
    """
    refinements = await propose_experiments(
        state.style_profile,
        state.current_template,
        state.knowledge_base,
        state.best_metrics,
        state.last_iteration_results,
        client=ctx.reasoning_client,
        model=ctx.config.reasoning_model,
        num_experiments=ctx.config.num_branches,
        vision_feedback=vision_fb,
        roundtrip_feedback=roundtrip_fb,
        caption_diffs=caption_diffs,
    )

    refinements = enforce_hypothesis_diversity(refinements, state.current_template)

    proposals: list[ExperimentProposal] = []
    for refinement in refinements:
        if refinement.should_stop:
            if _should_honor_stop(state, ctx, reason="reasoning model emitted [CONVERGED]"):
                logger.info("Reasoning model signaled convergence — honored")
                state.converged = True
                state.convergence_reason = ConvergenceReason.REASONING_STOP
                return [], True
            # Guard rejected: drop the stop flag and keep the refinement as a real proposal.
            refinement.should_stop = False
        proposals.append(
            ExperimentProposal(
                template=refinement.template,
                hypothesis=refinement.hypothesis,
                experiment_desc=refinement.experiment,
                builds_on=refinement.builds_on,
                open_problems=refinement.open_problems,
                lessons=refinement.lessons,
                analysis=refinement.analysis,
                template_changes=refinement.template_changes,
                changed_section=refinement.changed_section,
                target_category=refinement.target_category,
            )
        )

    if not proposals:
        if _should_honor_stop(state, ctx, reason="no experiments proposed"):
            logger.warning("No experiments proposed — honoring stop")
            state.converged = True
            state.convergence_reason = ConvergenceReason.REASONING_STOP
            return [], True
        # Guard rejected: fall through with an empty batch. _run_experiments_parallel
        # handles an empty proposals list, the outer loop bumps plateau_counter, and
        # the next propose call benefits from the widened ranked-category pressure.
        logger.warning("No experiments proposed — guard rejected, continuing with empty batch")
        return [], False

    # Validate templates — reject proposals with structural invariant violations
    valid_proposals: list[ExperimentProposal] = []
    for p in proposals:
        errors = validate_template(p.template, p.changed_section)
        if errors:
            logger.warning("Skipping invalid proposal (hyp: %.80s): %s", p.hypothesis, "; ".join(errors))
            continue
        valid_proposals.append(p)
    proposals = valid_proposals

    return proposals, False


async def _run_experiments_parallel(
    state: LoopState,
    ctx: RunContext,
    proposals: list[ExperimentProposal],
    iteration: int,
) -> list[IterationResult]:
    """Phase 2: gather all experiment runs in parallel.

    Returns the collected results (may be empty if all failed).
    """
    exp_tasks = [
        run_experiment(
            experiment_id=i,
            template=p.template,
            iteration=iteration,
            fixed_refs=state.fixed_references,
            config=ctx.config,
            gemini_client=ctx.gemini_client,
            registry=ctx.registry,
            gemini_semaphore=ctx.gemini_semaphore,
            eval_semaphore=ctx.eval_semaphore,
            last_results=state.last_iteration_results,
            hypothesis=p.hypothesis,
            experiment_desc=p.experiment_desc,
            analysis=p.analysis,
            template_changes=p.template_changes,
            changed_section=p.changed_section,
            target_category=p.target_category,
        )
        for i, p in enumerate(proposals)
    ]

    return collect_experiment_results(await asyncio.gather(*exp_tasks, return_exceptions=True), "Experiment")


def _score_and_rank(exp_results: list[IterationResult], state: LoopState) -> IterationRanking:
    """Phase 3: compute aggregates, adaptive scores, pick best, return bundled ranking."""
    all_agg = [r.aggregated for r in exp_results]
    adaptive_scores = {id(r): adaptive_composite_score(r.aggregated, all_agg) for r in exp_results}
    best_exp = max(exp_results, key=lambda r: adaptive_scores[id(r)])
    best_score = composite_score(best_exp.aggregated)
    baseline_score = composite_score(state.best_metrics) if state.best_metrics else float("-inf")
    return IterationRanking(
        exp_results=exp_results,
        adaptive_scores=adaptive_scores,
        best_exp=best_exp,
        best_score=best_score,
        baseline_score=baseline_score,
        epsilon=improvement_epsilon(baseline_score),
    )


async def _confirmatory_validation(
    ranking: IterationRanking,
    state: LoopState,
    ctx: RunContext,
    iteration: int,
) -> None:
    """Phase 3.1: replicate top-2 candidates + incumbent for statistical testing.

    Only runs in rigorous mode. Updates ranking.best_exp/best_score with median scores.
    """
    from art_style_search.experiment import replicate_experiment
    from art_style_search.scoring import paired_promotion_test

    if ctx.config.protocol != "rigorous":
        return

    candidates = _candidate_results_for_validation(ranking)
    if not candidates:
        return

    logger.info("Confirmatory validation: replicating %d candidates + incumbent", len(candidates))

    # Build replicate tasks: top-2 candidates (existing scores as rep 0) + incumbent
    tasks = []
    for exp in candidates:
        tasks.append(
            replicate_experiment(
                template=exp.template,
                branch_id=exp.branch_id,
                iteration=iteration,
                fixed_refs=state.fixed_references,
                config=ctx.config,
                gemini_client=ctx.gemini_client,
                registry=ctx.registry,
                gemini_semaphore=ctx.gemini_semaphore,
                eval_semaphore=ctx.eval_semaphore,
                n_replicates=3,
                existing_scores=exp.per_image_scores,
            )
        )
    # Incumbent replication (fresh, no existing scores)
    if state.best_template is not None:
        tasks.append(
            replicate_experiment(
                template=state.best_template,
                branch_id=900,  # sentinel ID for incumbent
                iteration=iteration,
                fixed_refs=state.fixed_references,
                config=ctx.config,
                gemini_client=ctx.gemini_client,
                registry=ctx.registry,
                gemini_semaphore=ctx.gemini_semaphore,
                eval_semaphore=ctx.eval_semaphore,
                n_replicates=3,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    candidate_evals = []
    incumbent_eval = None
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning("Confirmatory replicate %d failed: %s", i, result)
            continue
        if i < len(candidates):
            candidate_evals.append((candidates[i], result))
        else:
            incumbent_eval = result

    if not candidate_evals or incumbent_eval is None:
        logger.warning("Confirmatory validation incomplete — falling back to single-pass scores")
        return

    # Replace single-pass candidate results with replicated medians before promotion or persistence.
    for exp, evaluation in candidate_evals:
        exp.per_image_scores = list(evaluation.median_per_image)
        exp.aggregated = evaluation.median_aggregated

    updated_agg = [r.aggregated for r in ranking.exp_results]
    ranking.adaptive_scores = {id(r): adaptive_composite_score(r.aggregated, updated_agg) for r in ranking.exp_results}

    best_candidate_exp, best_candidate_eval = max(
        candidate_evals, key=lambda x: composite_score(x[1].median_aggregated)
    )
    ranking.best_exp = best_candidate_exp
    ranking.best_score = composite_score(best_candidate_exp.aggregated)
    ranking.best_replicate_scores = [composite_score(agg) for agg in best_candidate_eval.replicate_aggregated]

    # Run statistical test: best candidate vs incumbent
    test_result = paired_promotion_test(
        best_candidate_eval.median_per_image,
        incumbent_eval.median_per_image,
    )
    logger.info(
        "Promotion test: p=%.4f, effect=%.5f, CI=[%.5f, %.5f], passed=%s",
        test_result.p_value,
        test_result.effect_size,
        test_result.ci_lower,
        test_result.ci_upper,
        test_result.passed,
    )

    # Store test result on ranking for use in _apply_iteration_result
    ranking.promotion_test = test_result
    ranking.baseline_score = composite_score(incumbent_eval.median_aggregated)


async def _synthesize_reasoning(
    ranking: IterationRanking,
    state: LoopState,
    ctx: RunContext,
) -> tuple[PromptTemplate, str] | None:
    """Phase 3.5a: reasoning call to merge top experiments into one template.

    Returns (merged_template, merged_hypothesis) or None if <2 experiments.
    """
    if len(ranking.exp_results) < 2:
        return None

    ranked_for_synth = sorted(ranking.exp_results, key=lambda r: ranking.adaptive_scores[id(r)], reverse=True)
    top_exps = ranked_for_synth[:3]
    logger.info("Synthesizing top %d experiments into merged template", len(top_exps))

    return await synthesize_templates(
        top_exps,
        state.style_profile,
        client=ctx.reasoning_client,
        model=ctx.config.reasoning_model,
    )


async def _run_synthesis_experiment(
    synth_template_result: tuple[PromptTemplate, str],
    ranking: IterationRanking,
    state: LoopState,
    ctx: RunContext,
    iteration: int,
) -> None:
    """Phase 3.5b: run the synthesis experiment and update ranking."""
    merged_template, merged_hypothesis = synth_template_result
    errors = validate_template(merged_template)
    if errors:
        logger.warning("Synthesis template invalid — skipping: %s", "; ".join(errors))
        return

    synth_result = await run_experiment(
        experiment_id=len(ranking.exp_results),
        template=merged_template,
        iteration=iteration,
        fixed_refs=state.fixed_references,
        config=ctx.config,
        gemini_client=ctx.gemini_client,
        registry=ctx.registry,
        gemini_semaphore=ctx.gemini_semaphore,
        eval_semaphore=ctx.eval_semaphore,
        last_results=state.last_iteration_results,
        hypothesis=merged_hypothesis,
        experiment_desc="Synthesis of top experiments",
    )
    merged_score = composite_score(synth_result.aggregated)
    logger.info(
        "Synthesis result: DS=%.4f (best individual: %.4f)",
        synth_result.aggregated.dreamsim_similarity_mean,
        ranking.best_exp.aggregated.dreamsim_similarity_mean,
    )

    ranking.exp_results.append(synth_result)
    ranking.synth_result = synth_result
    updated_agg = [r.aggregated for r in ranking.exp_results]
    ranking.adaptive_scores = {id(r): adaptive_composite_score(r.aggregated, updated_agg) for r in ranking.exp_results}
    if merged_score > ranking.best_score:
        ranking.best_exp = synth_result
        ranking.best_score = merged_score
        logger.info("Synthesis beat best individual — adopting merged template")


def _update_knowledge_base_for_iteration(
    state: LoopState,
    ranking: IterationRanking,
    proposals: list[ExperimentProposal],
    baseline_metrics: AggregatedMetrics | None,
    iteration: int,
    decision: IterationDecision = "rejected",
) -> None:
    """Phase 4 (KB): update the knowledge base after the iteration decision is known."""
    decision_by_id: dict[int, IterationDecision] = {r.branch_id: "rejected" for r in ranking.exp_results}
    selected = next((result for result in ranking.exp_results if result.kept), None)
    if selected is not None:
        decision_by_id[selected.branch_id] = decision

    # Synthesis result (if any) is appended to exp_results but has no matching
    # proposal — zip() with strict=False stops at the shorter of the two.
    for exp_result, proposal in zip(ranking.exp_results, proposals, strict=False):
        update_knowledge_base(
            state.knowledge_base,
            exp_result,
            exp_result.template,
            baseline_metrics,
            proposal,
            iteration,
            decision=decision_by_id[exp_result.branch_id],
        )

    if ranking.synth_result is not None:
        synth_result = ranking.synth_result
        synth_proposal = ExperimentProposal(
            template=synth_result.template,
            hypothesis=synth_result.hypothesis,
            experiment_desc=synth_result.experiment,
            builds_on=None,
            open_problems=[],
            lessons=Lessons(),
            target_category=synth_result.target_category,
        )
        update_knowledge_base(
            state.knowledge_base,
            synth_result,
            synth_result.template,
            baseline_metrics,
            synth_proposal,
            iteration,
            decision=decision_by_id[synth_result.branch_id],
        )


def _apply_iteration_result(state: LoopState, ranking: IterationRanking, config: Config) -> IterationDecision:
    """Decide improvement vs plateau and update state accordingly.

    On a genuine improvement: call ``_apply_best_result`` and reset plateau.
    On plateau: increment counter.  On even plateau counts with ≥2 experiments,
    adopt second-best via ``_apply_exploration_result`` and reset plateau to 1
    to give exploration runway.
    """
    improved = ranking.best_score > ranking.baseline_score + ranking.epsilon

    # In rigorous mode, also require statistical confirmation
    if improved and config.protocol == "rigorous":
        promotion_test = ranking.promotion_test
        if promotion_test is not None and not promotion_test.passed:
            logger.info(
                "Epsilon check passed but statistical test failed (p=%.4f) — rejecting promotion",
                promotion_test.p_value,
            )
            improved = False

    if improved:
        _apply_best_result(state, ranking.best_exp)
        state.plateau_counter = 0
        _log_promotion_decision(
            state,
            ranking,
            "promoted",
            "Exceeded baseline + epsilon",
            config,
            replicate_scores=ranking.best_replicate_scores,
        )
        return "promoted"

    state.plateau_counter += 1
    # Exploration: on even plateau counts, adopt second-best to escape local optima.
    # Needs at least 2 experiments to pick a second-best from.
    can_explore = (
        state.plateau_counter >= _EXPLORATION_MIN_PLATEAU
        and state.plateau_counter % _EXPLORATION_CADENCE == 0
        and len(ranking.exp_results) >= 2
    )
    if can_explore:
        ranked = sorted(
            ranking.exp_results,
            key=lambda r: ranking.adaptive_scores[id(r)],
            reverse=True,
        )
        second_best = ranked[1]
        logger.info("Exploration: adopting second-best experiment to escape potential local optimum")
        _apply_exploration_result(state, second_best)
        state.plateau_counter = _EXPLORATION_RESET_PLATEAU
        _log_promotion_decision(
            state,
            ranking,
            "exploration",
            "Plateau escape via second-best",
            config,
            candidate=second_best,
        )
        return "exploration"
    else:
        _log_promotion_decision(
            state,
            ranking,
            "rejected",
            f"Delta {ranking.best_score - ranking.baseline_score:.5f} < epsilon {ranking.epsilon:.5f}",
            config,
            replicate_scores=ranking.best_replicate_scores,
        )
        return "rejected"


def _record_iteration_state(
    state: LoopState,
    ranking: IterationRanking,
    iteration: int,
    ctx: RunContext,
) -> None:
    """Persist iteration results: prev captions, history, logs, save_state."""
    # Preserve current best captions for next iteration's diff (N-1 vs N-2)
    current_best = best_kept_result(state.last_iteration_results)
    if current_best and current_best.iteration_captions:
        state.prev_best_captions = list(current_best.iteration_captions)

    state.last_iteration_results = ranking.exp_results
    state.experiment_history.extend(ranking.exp_results)
    # Cap persisted history to avoid unbounded state.json growth
    if len(state.experiment_history) > _MAX_PERSISTED_HISTORY:
        state.experiment_history = state.experiment_history[-_MAX_PERSISTED_HISTORY:]

    _log_experiment_results(ranking.exp_results, ctx.config.log_dir)
    # Strip heavy fields from non-kept results AFTER logging (iteration logs keep full data)
    for r in ranking.exp_results:
        if not r.kept:
            r.iteration_captions = []
            r.rendered_prompt = ""
    save_state(state, ctx.config.state_file)
    _save_best_prompt(state, ctx.config.log_dir)


def _check_plateau_convergence(state: LoopState, ctx: RunContext) -> bool:
    """Return True if the plateau window has been hit. Sets convergence fields."""
    if state.plateau_counter >= ctx.config.plateau_window:
        logger.info("Plateau detected (%d iterations without improvement)", state.plateau_counter)
        state.converged = True
        state.convergence_reason = ConvergenceReason.PLATEAU
        return True
    return False


def _should_honor_stop(state: LoopState, ctx: RunContext, reason: str) -> bool:
    """Gate the reasoning model's ``[CONVERGED]`` signal behind substantive conditions.

    All three clauses must hold for a stop to be honored:

    1. **Iteration floor** — at least ``_MIN_ITER_FRACTION_FOR_STOP`` of ``max_iterations``
       has elapsed. Stops short runs from self-terminating after 2 flat iterations.
    2. **Plateau depth** — ``plateau_counter`` is within 1 of ``plateau_window`` (with a
       floor of 2). Ensures the plateau is real, not a transient flat iteration.
    3. **Category coverage** — every canonical hypothesis category in ``CATEGORY_SYNONYMS``
       has at least one hypothesis in the KB. If any synonym-map category is still
       unexplored, there is by definition a concrete untried direction — reject the stop
       and let the widened ranked list steer the next propose call toward it.

    When the guard rejects, logs the reason so post-hoc debugging is trivial.
    """
    from art_style_search.utils import CATEGORY_SYNONYMS

    min_iter_floor = ctx.config.max_iterations * _MIN_ITER_FRACTION_FOR_STOP
    if state.iteration + 1 < min_iter_floor:
        logger.info(
            "Rejecting stop (%s): iteration %d below floor %.1f",
            reason,
            state.iteration + 1,
            min_iter_floor,
        )
        return False

    plateau_floor = max(ctx.config.plateau_window - 1, 2)
    if state.plateau_counter < plateau_floor:
        logger.info(
            "Rejecting stop (%s): plateau_counter %d below floor %d",
            reason,
            state.plateau_counter,
            plateau_floor,
        )
        return False

    tried = {cat for cat, prog in state.knowledge_base.categories.items() if prog.hypothesis_ids}
    untried = sorted(cat for cat in CATEGORY_SYNONYMS if cat not in tried)
    if untried:
        logger.info("Rejecting stop (%s): unexplored categories=%s", reason, untried)
        return False

    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run(config: Config) -> LoopState:
    """Run the full optimization loop."""
    ctx = await _setup_run_context(config)

    all_ref_paths = _discover_images(config.reference_dir)
    if not all_ref_paths:
        msg = f"No images found in {config.reference_dir}"
        raise FileNotFoundError(msg)
    logger.info("Found %d reference images", len(all_ref_paths))

    # Write or verify run manifest
    manifest_path = config.run_dir / "run_manifest.json"
    existing_manifest = load_manifest(manifest_path)
    if existing_manifest is None:
        manifest = _build_manifest(config)
        save_manifest(manifest, manifest_path)
    else:
        _verify_manifest(config, existing_manifest)

    # Resume from disk, or run zero-step
    state = load_state(config.state_file)
    if state is not None:
        logger.info("Resumed from iteration %d with %d fixed references", state.iteration, len(state.fixed_references))
        if state.converged:
            logger.info("Previous run already converged (%s) — skipping loop", state.convergence_reason)
            return _finalize_run(state, ctx)
    else:
        state = await _zero_step(ctx, all_ref_paths)

    # Main optimization loop
    for iteration in range(state.iteration, config.max_iterations):
        state.iteration = iteration
        logger.info("=== Iteration %d/%d ===", iteration + 1, config.max_iterations)

        vision_fb, roundtrip_fb, caption_diffs = _build_iteration_context(state)

        proposals, converged = await _propose_iteration_experiments(state, ctx, vision_fb, roundtrip_fb, caption_diffs)
        if converged:
            break

        exp_results = await _run_experiments_parallel(state, ctx, proposals, iteration)
        if not exp_results:
            logger.warning("All experiments failed this iteration")
            state.plateau_counter += 1
            state.iteration = iteration + 1  # advance so resume skips this iteration
            save_state(state, config.state_file)
            if state.plateau_counter >= config.plateau_window:
                state.converged = True
                state.convergence_reason = ConvergenceReason.PLATEAU
                break
            continue

        ranking = _score_and_rank(exp_results, state)

        # Run synthesis reasoning + pairwise + review in parallel
        synth_result, pairwise_result, review_result = await asyncio.gather(
            _synthesize_reasoning(ranking, state, ctx),
            _run_pairwise_comparison(ranking, state, ctx),
            _run_independent_review(ranking, proposals, state, ctx),
            return_exceptions=True,
        )
        if isinstance(pairwise_result, BaseException):
            logger.warning("Pairwise comparison failed — skipping: %s", pairwise_result)
        if isinstance(review_result, BaseException):
            logger.warning("Independent review failed — skipping: %s", review_result)

        # Synthesis experiment needs the template from the reasoning call
        synth_template = synth_result
        if isinstance(synth_template, BaseException):
            logger.warning("Synthesis reasoning failed — skipping: %s", synth_template)
        elif synth_template is not None:
            try:
                await _run_synthesis_experiment(synth_template, ranking, state, ctx, iteration)
            except Exception:
                logger.warning("Synthesis experiment failed — continuing with individual experiments", exc_info=True)

        # Phase 3.1: Confirmatory validation (rigorous mode only)
        try:
            await _confirmatory_validation(ranking, state, ctx, iteration)
        except Exception:
            logger.warning("Confirmatory validation failed — falling back to single-pass", exc_info=True)

        baseline_metrics = state.best_metrics
        decision = _apply_iteration_result(state, ranking, ctx.config)
        _update_knowledge_base_for_iteration(state, ranking, proposals, baseline_metrics, iteration, decision)
        _record_iteration_state(state, ranking, iteration, ctx)

        if _check_plateau_convergence(state, ctx):
            break
    else:
        state.converged = True
        state.convergence_reason = ConvergenceReason.MAX_ITERATIONS

    return _finalize_run(state, ctx)
