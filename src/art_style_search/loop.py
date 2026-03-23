"""Main orchestration loop using Bulk Synchronous Parallel (BSP) rounds."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import random
from pathlib import Path

import anthropic
from google import genai

from art_style_search.analyze import analyze_style
from art_style_search.caption import caption_references
from art_style_search.config import Config
from art_style_search.evaluate import evaluate_images
from art_style_search.generate import generate_images
from art_style_search.models import ModelRegistry
from art_style_search.prompt import propose_initial_templates, refine_template
from art_style_search.state import load_state, save_iteration_log, save_state
from art_style_search.types import (
    AggregatedMetrics,
    BranchState,
    ConvergenceReason,
    IterationResult,
    LoopState,
    PromptTemplate,
    composite_score,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


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


def _find_global_best(
    branches: list[BranchState],
) -> tuple[PromptTemplate | None, AggregatedMetrics | None]:
    """Find the best-performing template across all branches."""
    best_template: PromptTemplate | None = None
    best_metrics: AggregatedMetrics | None = None
    best_score = float("-inf")

    for branch in branches:
        if branch.best_metrics is not None:
            score = composite_score(branch.best_metrics)
            if score > best_score:
                best_score = score
                best_template = branch.best_template
                best_metrics = branch.best_metrics

    return best_template, best_metrics


async def _run_branch_iteration(
    branch: BranchState,
    iteration: int,
    all_ref_paths: list[Path],
    config: Config,
    *,
    gemini_client: genai.Client,
    anthropic_client: anthropic.AsyncAnthropic,
    registry: ModelRegistry,
    gen_semaphore: asyncio.Semaphore,
    eval_semaphore: asyncio.Semaphore,
    style_profile: object,
    global_best_template: PromptTemplate | None,
    global_best_metrics: AggregatedMetrics | None,
) -> IterationResult:
    """Execute one full iteration for a single branch: generate → evaluate → refine."""
    from art_style_search.types import StyleProfile

    assert isinstance(style_profile, StyleProfile)

    rendered = branch.current_template.render()
    logger.info("Branch %d iter %d — prompt: %.80s...", branch.branch_id, iteration, rendered)

    # Phase 2: Generate images
    image_paths = await generate_images(
        rendered,
        num_images=config.num_images,
        aspect_ratio=config.aspect_ratio,
        output_dir=config.output_dir,
        iteration=iteration,
        branch_id=branch.branch_id,
        client=gemini_client,
        model=config.generator_model,
        semaphore=gen_semaphore,
    )

    if not image_paths:
        logger.warning("Branch %d iter %d — no images generated", branch.branch_id, iteration)
        raise RuntimeError(f"Branch {branch.branch_id}: no images generated")

    # Phase 3: Evaluate (re-sample references for this iteration)
    eval_refs = _sample(all_ref_paths, config.max_eval_images)

    # Run metric evaluation and vision comparison in parallel
    from art_style_search.evaluate import compare_vision

    eval_task = evaluate_images(
        image_paths,
        eval_refs,
        rendered,
        registry=registry,
        semaphore=eval_semaphore,
    )
    vision_task = compare_vision(
        image_paths,
        eval_refs,
        client=gemini_client,
        model=config.caption_model,
        semaphore=gen_semaphore,
    )

    (scores, aggregated), vision_feedback = await asyncio.gather(eval_task, vision_task)
    logger.info("Branch %d iter %d — vision feedback: %.120s...", branch.branch_id, iteration, vision_feedback)

    # Phase 1 (next iter prep): Refine template
    new_template, analysis, template_changes, should_stop = await refine_template(
        style_profile,
        branch,
        global_best_template,
        global_best_metrics,
        client=anthropic_client,
        model=config.claude_model,
        vision_feedback=vision_feedback,
    )

    # Determine if this iteration improved
    current_score = composite_score(aggregated)
    kept = branch.best_metrics is None or current_score > composite_score(branch.best_metrics)

    result = IterationResult(
        branch_id=branch.branch_id,
        iteration=iteration,
        template=branch.current_template,
        rendered_prompt=rendered,
        image_paths=image_paths,
        per_image_scores=scores,
        aggregated=aggregated,
        claude_analysis=analysis,
        template_changes=template_changes,
        kept=kept,
        vision_feedback=vision_feedback,
    )

    # Update branch state
    branch.history.append(result)
    branch.current_template = new_template

    if kept:
        branch.best_template = branch.current_template
        branch.best_metrics = aggregated
        branch.plateau_counter = 0
    else:
        branch.plateau_counter += 1

    if should_stop:
        branch.stopped = True
        branch.stop_reason = ConvergenceReason.CLAUDE_STOP
        logger.info("Branch %d — Claude signaled convergence", branch.branch_id)
    elif branch.plateau_counter >= config.plateau_window:
        branch.stopped = True
        branch.stop_reason = ConvergenceReason.PLATEAU
        logger.info(
            "Branch %d — plateau detected (%d iterations without improvement)",
            branch.branch_id,
            branch.plateau_counter,
        )

    return result


async def run(config: Config) -> LoopState:
    """Run the full optimization loop."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Set thread pool size for torch evaluation
    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=config.eval_concurrency))

    # Initialize API clients
    gemini_client = genai.Client(api_key=config.google_api_key)
    anthropic_client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    # Semaphores
    gen_semaphore = asyncio.Semaphore(config.gemini_concurrency)
    eval_semaphore = asyncio.Semaphore(config.eval_concurrency)
    caption_semaphore = asyncio.Semaphore(config.gemini_concurrency)

    # Load evaluation models
    logger.info("Loading evaluation models...")
    registry = await asyncio.to_thread(ModelRegistry.load_all)

    # Discover reference images
    all_ref_paths = _discover_images(config.reference_dir)
    if not all_ref_paths:
        msg = f"No images found in {config.reference_dir}"
        raise FileNotFoundError(msg)
    logger.info("Found %d reference images", len(all_ref_paths))

    # Try to resume from state
    state = load_state(config.state_file)
    if state is not None:
        logger.info("Resumed from iteration %d", state.iteration)
    else:
        # Zero-step: caption + analyze
        analysis_sample = _sample(all_ref_paths, config.max_analysis_images)
        logger.info("Zero-step: captioning %d reference images...", len(analysis_sample))

        captions = await caption_references(
            analysis_sample,
            model=config.caption_model,
            client=gemini_client,
            cache_dir=config.log_dir / "captions",
            semaphore=caption_semaphore,
        )

        logger.info("Zero-step: analyzing art style...")
        style_profile, initial_template = await analyze_style(
            analysis_sample,
            captions,
            gemini_client=gemini_client,
            anthropic_client=anthropic_client,
            caption_model=config.caption_model,
            claude_model=config.claude_model,
            cache_path=config.log_dir / "style_profile.json",
        )

        logger.info("Zero-step: proposing %d initial templates...", config.num_branches)
        initial_templates = await propose_initial_templates(
            style_profile,
            config.num_branches,
            client=anthropic_client,
            model=config.claude_model,
        )

        # Use analyzed template as fallback for any empty templates
        for i, t in enumerate(initial_templates):
            if not t.sections:
                initial_templates[i] = initial_template

        branches = [
            BranchState(
                branch_id=i,
                current_template=t,
                best_template=t,
            )
            for i, t in enumerate(initial_templates)
        ]

        state = LoopState(
            iteration=0,
            branches=branches,
            captions=captions,
            style_profile=style_profile,
        )

    # Main loop
    for iteration in range(state.iteration, config.max_iterations):
        state.iteration = iteration
        active_branches = [b for b in state.branches if not b.stopped]

        if not active_branches:
            logger.info("All branches stopped — loop complete")
            state.converged = True
            state.convergence_reason = ConvergenceReason.PLATEAU
            break

        logger.info(
            "=== Iteration %d/%d — %d active branches ===", iteration + 1, config.max_iterations, len(active_branches)
        )

        global_best_template, global_best_metrics = _find_global_best(state.branches)

        # Run all active branches in parallel (BSP round)
        tasks = [
            _run_branch_iteration(
                branch,
                iteration,
                all_ref_paths,
                config,
                gemini_client=gemini_client,
                anthropic_client=anthropic_client,
                registry=registry,
                gen_semaphore=gen_semaphore,
                eval_semaphore=eval_semaphore,
                style_profile=state.style_profile,
                global_best_template=global_best_template,
                global_best_metrics=global_best_metrics,
            )
            for branch in active_branches
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, BaseException):
                logger.error("Branch iteration failed: %s", result)
            else:
                save_iteration_log(result, config.log_dir)

                # Log metrics
                m = result.aggregated
                logger.info(
                    "Branch %d — DINO=%.4f LPIPS=%.4f HPS=%.4f Aes=%.2f %s",
                    result.branch_id,
                    m.dino_similarity_mean,
                    m.lpips_distance_mean,
                    m.hps_score_mean,
                    m.aesthetics_score_mean,
                    "KEPT" if result.kept else "discarded",
                )

        # Update global best
        global_best_template, global_best_metrics = _find_global_best(state.branches)
        if global_best_template is not None:
            state.global_best_prompt = global_best_template.render()
            state.global_best_metrics = global_best_metrics

        # Save state after each iteration
        save_state(state, config.state_file)

    else:
        # max_iterations reached
        state.converged = True
        state.convergence_reason = ConvergenceReason.MAX_ITERATIONS

    save_state(state, config.state_file)

    # Final summary
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
    logger.info("BEST PROMPT: %s", state.global_best_prompt)
    logger.info("Convergence: %s", state.convergence_reason)

    return state
