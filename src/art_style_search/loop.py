"""Main orchestration loop with experiment-based optimization.

The loop optimizes a **meta-prompt** — instructions for how to caption images
so that the captions can recreate the originals via image generation.

Each iteration:
1. Claude proposes N diverse experiments (hypothesis-driven template variants)
2. Each experiment: caption + generate + evaluate in parallel
3. Best experiment updates the shared state; all results feed into the Knowledge Base
4. Check convergence (plateau / Claude stop / max iterations)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import random
from pathlib import Path

from google import genai

from art_style_search.analyze import analyze_style
from art_style_search.caption import caption_references
from art_style_search.config import Config
from art_style_search.experiment import (
    ExperimentProposal,
    best_kept_result,
    collect_experiment_results,
    run_experiment,
)
from art_style_search.knowledge import build_caption_diffs, update_knowledge_base
from art_style_search.models import ModelRegistry
from art_style_search.prompt import (
    Lessons,
    propose_experiments,
    propose_initial_templates,
    synthesize_templates,
)
from art_style_search.state import load_state, save_iteration_log, save_state
from art_style_search.types import (
    IMPROVEMENT_EPSILON,
    ConvergenceReason,
    IterationResult,
    KnowledgeBase,
    LoopState,
    adaptive_composite_score,
    composite_score,
)
from art_style_search.utils import ReasoningClient

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _discover_images(directory: Path) -> list[Path]:
    """Find all image files in a directory, sorted for determinism."""
    paths = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    paths.sort()
    return paths


def _sample(items: list[Path], max_count: int) -> list[Path]:
    """Random sample up to max_count items from a list."""
    if len(items) <= max_count:
        return items
    return random.sample(items, max_count)


def _apply_best_result(state: LoopState, result: IterationResult) -> None:
    """Update state with a new best experiment result."""
    result.kept = True
    state.current_template = result.template
    state.best_template = result.template
    state.best_metrics = result.aggregated
    state.global_best_prompt = result.rendered_prompt
    state.global_best_metrics = result.aggregated


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
            "Exp %d — DINO=%.3f LPIPS=%.3f Color=%.3f Tex=%.3f HPS=%.3f Aes=%.1f V[S=%.0f Su=%.0f Co=%.0f] %s",
            r.branch_id,
            m.dino_similarity_mean,
            m.lpips_distance_mean,
            m.color_histogram_mean,
            m.texture_mean,
            m.hps_score_mean,
            m.aesthetics_score_mean,
            m.vision_style,
            m.vision_subject,
            m.vision_composition,
            "KEPT" if r.kept else "discarded",
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

# Max experiment results to persist in state.json (older ones are in iteration logs)
_MAX_PERSISTED_HISTORY = 30


async def run(config: Config) -> LoopState:
    """Run the full optimization loop."""
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
    )

    gemini_semaphore = asyncio.Semaphore(config.gemini_concurrency)
    eval_semaphore = asyncio.Semaphore(config.eval_concurrency)

    logger.info("Loading evaluation models...")
    registry = await asyncio.to_thread(ModelRegistry.load_all)

    all_ref_paths = _discover_images(config.reference_dir)
    if not all_ref_paths:
        msg = f"No images found in {config.reference_dir}"
        raise FileNotFoundError(msg)
    logger.info("Found %d reference images", len(all_ref_paths))

    # Shared kwargs for run_experiment
    exp_kwargs = {
        "config": config,
        "gemini_client": gemini_client,
        "registry": registry,
        "gemini_semaphore": gemini_semaphore,
        "eval_semaphore": eval_semaphore,
    }

    # Try to resume from state
    state = load_state(config.state_file)
    if state is not None:
        logger.info("Resumed from iteration %d with %d fixed references", state.iteration, len(state.fixed_references))
        fixed_refs = state.fixed_references
    else:
        fixed_refs = _sample(all_ref_paths, 10)
        logger.info("Fixed %d reference images for optimization", len(fixed_refs))

        logger.info("Zero-step: captioning %d reference images...", len(fixed_refs))
        captions = await caption_references(
            fixed_refs,
            model=config.caption_model,
            client=gemini_client,
            cache_dir=config.log_dir / "captions",
            semaphore=gemini_semaphore,
            cache_key="initial",
        )

        logger.info("Zero-step: analyzing art style...")
        style_profile, initial_template = await analyze_style(
            fixed_refs,
            captions,
            gemini_client=gemini_client,
            reasoning_client=reasoning_client,
            caption_model=config.caption_model,
            reasoning_model=config.reasoning_model,
            cache_path=config.log_dir / "style_profile.json",
        )

        logger.info("Zero-step: proposing %d initial meta-prompts...", config.num_branches)
        initial_templates = await propose_initial_templates(
            style_profile,
            config.num_branches,
            client=reasoning_client,
            model=config.reasoning_model,
        )

        for i, t in enumerate(initial_templates):
            if not t.sections:
                initial_templates[i] = initial_template

        state = LoopState(
            iteration=0,
            current_template=initial_templates[0],
            best_template=initial_templates[0],
            best_metrics=None,
            knowledge_base=KnowledgeBase(),
            captions=captions,
            style_profile=style_profile,
            fixed_references=fixed_refs,
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
                    last_results=[],
                    hypothesis=f"Initial template {i}",
                    experiment_desc="Zero-step diverse template",
                    **exp_kwargs,
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

    # Main optimization loop
    for iteration in range(state.iteration, config.max_iterations):
        state.iteration = iteration

        logger.info("=== Iteration %d/%d ===", iteration + 1, config.max_iterations)

        # Use the best experiment's feedback from last iteration
        best_last = best_kept_result(state.last_iteration_results)
        vision_fb = best_last.vision_feedback if best_last else ""
        roundtrip_fb = best_last.roundtrip_feedback if best_last else ""

        caption_diffs = ""
        if best_last and best_last.iteration_captions:
            sorted_caps = sorted(
                zip(best_last.iteration_captions, best_last.per_image_scores, strict=False),
                key=lambda x: x[1].dino_similarity,
            )
            worst_caps = [c for c, _ in sorted_caps[:3]]
            caption_diffs = build_caption_diffs(state.last_iteration_results, worst_caps)

        # Phase 1: Claude proposes N experiments in a single call
        refinements = await propose_experiments(
            state.style_profile,
            state.current_template,
            state.knowledge_base,
            state.best_metrics,
            state.last_iteration_results,
            client=reasoning_client,
            model=config.reasoning_model,
            num_experiments=config.num_branches,
            vision_feedback=vision_fb,
            roundtrip_feedback=roundtrip_fb,
            caption_diffs=caption_diffs,
        )

        proposals: list[ExperimentProposal] = []
        for refinement in refinements:
            if refinement.should_stop:
                logger.info("Claude signaled convergence")
                state.converged = True
                state.convergence_reason = ConvergenceReason.CLAUDE_STOP
                break
            proposals.append(
                ExperimentProposal(
                    template=refinement.template,
                    hypothesis=refinement.hypothesis,
                    experiment_desc=refinement.experiment,
                    builds_on=refinement.builds_on,
                    open_problems=refinement.open_problems,
                    lessons=refinement.lessons,
                )
            )

        if state.converged:
            break

        if not proposals:
            logger.warning("No experiments proposed — stopping")
            state.converged = True
            state.convergence_reason = ConvergenceReason.CLAUDE_STOP
            break

        # Phase 2: Run all experiments in parallel
        exp_tasks = [
            run_experiment(
                experiment_id=i,
                template=p.template,
                iteration=iteration,
                fixed_refs=fixed_refs,
                last_results=state.last_iteration_results,
                hypothesis=p.hypothesis,
                experiment_desc=p.experiment_desc,
                **exp_kwargs,
            )
            for i, p in enumerate(proposals)
        ]

        exp_results = collect_experiment_results(await asyncio.gather(*exp_tasks, return_exceptions=True), "Experiment")

        if not exp_results:
            logger.warning("All experiments failed this iteration")
            state.plateau_counter += 1
            if state.plateau_counter >= config.plateau_window:
                state.converged = True
                state.convergence_reason = ConvergenceReason.PLATEAU
                break
            continue

        # Phase 3: Find best experiment
        # Adaptive scoring ranks experiments against each other (relative).
        # composite_score is used for improvement checks (absolute, same scale).
        all_agg = [r.aggregated for r in exp_results]
        best_exp = max(exp_results, key=lambda r: adaptive_composite_score(r.aggregated, all_agg))
        best_score = composite_score(best_exp.aggregated)
        baseline_score = composite_score(state.best_metrics) if state.best_metrics else float("-inf")

        # Pre-compute composite scores to avoid repeated calls
        exp_scores = {id(r): composite_score(r.aggregated) for r in exp_results}

        # Phase 3.5: Synthesis — merge top experiments if multiple improved
        synth_result: IterationResult | None = None
        improved_exps = [r for r in exp_results if exp_scores[id(r)] > baseline_score + IMPROVEMENT_EPSILON]
        if len(improved_exps) >= 2:
            logger.info("Synthesizing %d improved experiments into merged template", len(improved_exps))
            improved_exps.sort(key=lambda r: exp_scores[id(r)], reverse=True)
            top_exps = improved_exps[:3]

            merged_template, merged_hypothesis = await synthesize_templates(
                top_exps,
                state.style_profile,
                client=reasoning_client,
                model=config.reasoning_model,
            )

            # Validate the merged template
            synth_result = await run_experiment(
                experiment_id=len(exp_results),
                template=merged_template,
                iteration=iteration,
                fixed_refs=fixed_refs,
                last_results=state.last_iteration_results,
                hypothesis=merged_hypothesis,
                experiment_desc="Synthesis of top experiments",
                **exp_kwargs,
            )
            merged_score = composite_score(synth_result.aggregated)
            logger.info(
                "Synthesis result: DINO=%.4f (best individual: %.4f)",
                synth_result.aggregated.dino_similarity_mean,
                best_exp.aggregated.dino_similarity_mean,
            )

            exp_results.append(synth_result)
            if merged_score > best_score:
                best_exp = synth_result
                best_score = merged_score
                logger.info("Synthesis beat best individual — adopting merged template")

        # Update state with best result (epsilon threshold filters generation noise)
        improved = best_score > baseline_score + IMPROVEMENT_EPSILON

        if improved:
            _apply_best_result(state, best_exp)
            state.plateau_counter = 0
        else:
            state.plateau_counter += 1

            # Exploration: on even plateau counts (2, 4, 6, ...), adopt the second-best
            # experiment to escape potential local optima.  Odd counts stay greedy so we
            # alternate between exploration and exploitation.
            if state.plateau_counter >= 2 and state.plateau_counter % 2 == 0 and len(exp_results) >= 2:
                # Rank by adaptive score (diversity-aware) to pick a genuinely different runner-up
                ranked = sorted(
                    exp_results,
                    key=lambda r: adaptive_composite_score(r.aggregated, all_agg),
                    reverse=True,
                )
                second_best = ranked[1]
                logger.info("Exploration: adopting second-best experiment to escape potential local optimum")
                _apply_best_result(state, second_best)

        state.last_iteration_results = exp_results
        state.experiment_history.extend(exp_results)
        # Cap persisted history to avoid unbounded state.json growth
        if len(state.experiment_history) > _MAX_PERSISTED_HISTORY:
            state.experiment_history = state.experiment_history[-_MAX_PERSISTED_HISTORY:]

        # Phase 4: Update shared KB with ALL experiment results
        for exp_result, proposal in zip(exp_results, proposals, strict=False):
            update_knowledge_base(
                state.knowledge_base,
                exp_result,
                exp_result.template,
                state.best_metrics,
                proposal,
                iteration,
            )
        # Record synthesis experiment separately (it has no matching proposal)
        if synth_result is not None:
            synth_proposal = ExperimentProposal(
                template=synth_result.template,
                hypothesis=synth_result.hypothesis,
                experiment_desc=synth_result.experiment,
                builds_on=None,
                open_problems=[],
                lessons=Lessons(),
            )
            update_knowledge_base(
                state.knowledge_base,
                synth_result,
                synth_result.template,
                state.best_metrics,
                synth_proposal,
                iteration,
            )

        _log_experiment_results(exp_results, config.log_dir)
        save_state(state, config.state_file)
        _save_best_prompt(state, config.log_dir)

        if state.plateau_counter >= config.plateau_window:
            logger.info("Plateau detected (%d iterations without improvement)", state.plateau_counter)
            state.converged = True
            state.convergence_reason = ConvergenceReason.PLATEAU
            break

    else:
        state.converged = True
        state.convergence_reason = ConvergenceReason.MAX_ITERATIONS

    save_state(state, config.state_file)
    _save_best_prompt(state, config.log_dir)

    logger.info("=" * 60)
    if state.global_best_metrics:
        m = state.global_best_metrics
        logger.info(
            "FINAL BEST — DINO=%.4f LPIPS=%.4f HPS=%.4f Aes=%.2f",
            m.dino_similarity_mean,
            m.lpips_distance_mean,
            m.hps_score_mean,
            m.aesthetics_score_mean,
        )
    logger.info("BEST META-PROMPT: %s", state.global_best_prompt)
    logger.info("Convergence: %s", state.convergence_reason)
    logger.info("Total experiments: %d", len(state.experiment_history))
    logger.info("KB: %d hypotheses", len(state.knowledge_base.hypotheses))

    return state
