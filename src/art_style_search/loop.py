"""Main orchestration loop using Bulk Synchronous Parallel (BSP) rounds.

The loop optimizes a **meta-prompt** — instructions for how to caption images
so that the captions can recreate the originals via image generation.

Each iteration:
1. Use meta-prompt + Gemini Pro to caption each fixed reference image
2. Generate images from those captions via Gemini Flash
3. Compare each (original, generated) pair — metrics + vision feedback
4. Claude analyzes gaps and refines the meta-prompt
"""

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
from art_style_search.evaluate import compare_vision, evaluate_images
from art_style_search.generate import _generate_single
from art_style_search.models import ModelRegistry
from art_style_search.prompt import propose_initial_templates, refine_template
from art_style_search.state import load_state, save_iteration_log, save_state
from art_style_search.types import (
    AggregatedMetrics,
    BranchState,
    Caption,
    ConvergenceReason,
    IterationResult,
    LoopState,
    PromptTemplate,
    StyleProfile,
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


def _get_previous_per_image_scores(branch: BranchState) -> dict[Path, float]:
    """Get per-image DINO scores from the previous iteration, keyed by image path."""
    if not branch.history:
        return {}
    prev = branch.history[-1]
    result: dict[Path, float] = {}
    for cap, sc in zip(prev.iteration_captions, prev.per_image_scores, strict=False):
        result[cap.image_path] = sc.dino_similarity
    return result


def _build_caption_diffs(branch: BranchState, worst_captions: list[Caption]) -> str:
    """Show how captions changed for worst-performing images between last iteration and current."""
    if not branch.history or not worst_captions:
        return ""
    prev = branch.history[-1]
    prev_by_path = {c.image_path: c.text for c in prev.iteration_captions}

    diffs: list[str] = []
    for cap in worst_captions:
        prev_text = prev_by_path.get(cap.image_path)
        if prev_text is None:
            continue
        if prev_text == cap.text:
            diffs.append(
                f"**{cap.image_path.name}**: Caption UNCHANGED (meta-prompt change had no effect on this image)"
            )
        else:
            diffs.append(f"**{cap.image_path.name}**:\n  PREV: {prev_text[:200]}...\n  NOW:  {cap.text[:200]}...")
    if not diffs:
        return ""
    return "## Caption Changes (worst 3 images, prev → current)\n" + "\n".join(diffs)


async def _caption_and_generate(
    ref_paths: list[Path],
    meta_prompt: str,
    *,
    config: Config,
    gemini_client: genai.Client,
    gemini_semaphore: asyncio.Semaphore,
    iteration: int,
    branch_id: int,
) -> tuple[list[Caption], list[Path], list[tuple[Path, Path]]]:
    """Caption reference images with the meta-prompt, then generate images from captions.

    Returns (captions, generated_paths, pairs) where pairs maps (original, generated).
    """
    # Step 1: Caption all reference images using the meta-prompt
    captions = await caption_references(
        ref_paths,
        model=config.caption_model,
        client=gemini_client,
        cache_dir=config.log_dir / f"iter_{iteration:03d}" / f"branch_{branch_id}" / "captions",
        semaphore=gemini_semaphore,
        prompt=meta_prompt,
        cache_key=f"iter{iteration}_b{branch_id}",
    )

    # Step 2: Generate images from captions
    gen_dir = config.output_dir / f"iter_{iteration:03d}" / f"branch_{branch_id}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    gen_tasks = [
        _generate_single(
            caption.text,
            index=i,
            aspect_ratio=config.aspect_ratio,
            output_path=gen_dir / f"{i:02d}.png",
            client=gemini_client,
            model=config.generator_model,
            semaphore=gemini_semaphore,
        )
        for i, caption in enumerate(captions)
    ]

    gen_results = await asyncio.gather(*gen_tasks, return_exceptions=True)

    # Build successful pairs
    generated_paths: list[Path] = []
    pairs: list[tuple[Path, Path]] = []
    for i, (caption, gen_result) in enumerate(zip(captions, gen_results, strict=True)):
        if isinstance(gen_result, BaseException):
            logger.warning("Branch %d: generation from caption %d failed: %s", branch_id, i, gen_result)
        else:
            generated_paths.append(gen_result)
            pairs.append((caption.image_path, gen_result))

    return captions, generated_paths, pairs


async def _run_branch_iteration(
    branch: BranchState,
    iteration: int,
    fixed_refs: list[Path],
    config: Config,
    *,
    gemini_client: genai.Client,
    anthropic_client: anthropic.AsyncAnthropic,
    registry: ModelRegistry,
    gemini_semaphore: asyncio.Semaphore,
    eval_semaphore: asyncio.Semaphore,
    style_profile: StyleProfile,
    global_best_template: PromptTemplate | None,
    global_best_metrics: AggregatedMetrics | None,
) -> IterationResult:
    """Execute one iteration: meta-prompt → caption → generate → compare → refine."""
    meta_prompt = branch.current_template.render()
    logger.info("Branch %d iter %d — meta-prompt: %.100s...", branch.branch_id, iteration, meta_prompt)

    # Phase 1: Caption references with meta-prompt, then generate from captions
    captions, generated_paths, pairs = await _caption_and_generate(
        fixed_refs,
        meta_prompt,
        config=config,
        gemini_client=gemini_client,
        gemini_semaphore=gemini_semaphore,
        iteration=iteration,
        branch_id=branch.branch_id,
    )

    if not generated_paths:
        logger.warning("Branch %d iter %d — no images generated", branch.branch_id, iteration)
        raise RuntimeError(f"Branch {branch.branch_id}: no images generated")

    logger.info(
        "Branch %d iter %d — %d/%d images generated successfully",
        branch.branch_id,
        iteration,
        len(generated_paths),
        len(fixed_refs),
    )

    # Phase 2: Evaluate — compare each generated image against its specific original
    gen_paths_for_eval = [gen for _, gen in pairs]
    ref_paths_for_eval = [orig for orig, _ in pairs]

    eval_task = evaluate_images(
        gen_paths_for_eval,
        ref_paths_for_eval,
        meta_prompt,
        registry=registry,
        semaphore=eval_semaphore,
    )

    # Caption compliance check
    from art_style_search.evaluate import check_caption_compliance

    section_names = [s.name for s in branch.current_template.sections]
    compliance = check_caption_compliance(section_names, captions)

    (scores, aggregated) = await eval_task

    # Sort pairs by DINO score (worst first) for vision comparison
    scored_items = list(zip(pairs, scores, captions, strict=False))
    scored_items.sort(key=lambda x: x[1].dino_similarity)

    sorted_pairs = [item[0] for item in scored_items]
    sorted_captions = [item[2] for item in scored_items]

    # Vision comparison — show worst 5 pairs deterministically
    vision_feedback = await compare_vision(
        sorted_pairs,
        meta_prompt,
        client=gemini_client,
        model=config.caption_model,
        semaphore=gemini_semaphore,
        max_pairs=5,
    )
    logger.info("Branch %d iter %d — vision feedback: %.120s...", branch.branch_id, iteration, vision_feedback)

    # Build per-image roundtrip summary sorted worst→best
    roundtrip_details: list[str] = []
    prev_scores = _get_previous_per_image_scores(branch)
    for (_orig, _gen), sc, cap in scored_items:
        prev_dino = prev_scores.get(cap.image_path, None)
        trend = ""
        if prev_dino is not None:
            arrow = "↑" if sc.dino_similarity > prev_dino else "↓" if sc.dino_similarity < prev_dino else "="
            trend = f" [prev DINO={prev_dino:.3f} → {sc.dino_similarity:.3f} {arrow}]"
        roundtrip_details.append(
            f"Image ({_orig.name}): DINO={sc.dino_similarity:.3f} LPIPS={sc.lpips_distance:.3f} "
            f"HPS={sc.hps_score:.3f} Aes={sc.aesthetics_score:.1f}{trend}\n"
            f"  Caption: {cap.text[:300]}..."
        )
    roundtrip_feedback = "\n".join(roundtrip_details)
    if compliance:
        roundtrip_feedback = compliance + "\n\n" + roundtrip_feedback

    # Show caption diffs for worst 3 images if we have previous iteration
    caption_diffs = _build_caption_diffs(branch, sorted_captions[:3])

    # Phase 3: Refine meta-prompt
    (
        new_template,
        analysis,
        template_changes,
        should_stop,
        hypothesis,
        experiment,
        lessons,
    ) = await refine_template(
        style_profile,
        branch,
        global_best_template,
        global_best_metrics,
        client=anthropic_client,
        model=config.claude_model,
        vision_feedback=vision_feedback,
        roundtrip_feedback=roundtrip_feedback,
        caption_diffs=caption_diffs,
    )

    # Determine if this iteration improved
    current_score = composite_score(aggregated)
    kept = branch.best_metrics is None or current_score > composite_score(branch.best_metrics)

    # Build research log entry
    prev_metrics = branch.best_metrics or (branch.history[-1].aggregated if branch.history else None)
    log_entry = f"\n## Iteration {iteration}"
    if hypothesis:
        log_entry += f"\nHypothesis: {hypothesis}"
    if experiment:
        log_entry += f"\nExperiment: {experiment}"
    log_entry += f"\nResult: DINO={aggregated.dino_similarity_mean:.3f} LPIPS={aggregated.lpips_distance_mean:.3f}"
    if prev_metrics:
        log_entry += (
            f" (prev DINO={prev_metrics.dino_similarity_mean:.3f} LPIPS={prev_metrics.lpips_distance_mean:.3f})"
        )
    log_entry += f" — {'KEPT' if kept else 'DISCARDED'}"
    if lessons.confirmed:
        log_entry += f"\nConfirmed: {lessons.confirmed}"
    if lessons.rejected:
        log_entry += f"\nRejected: {lessons.rejected}"
    if lessons.new_insight:
        log_entry += f"\nInsight: {lessons.new_insight}"
    branch.research_log += log_entry

    result = IterationResult(
        branch_id=branch.branch_id,
        iteration=iteration,
        template=branch.current_template,
        rendered_prompt=meta_prompt,
        image_paths=generated_paths,
        per_image_scores=scores,
        aggregated=aggregated,
        claude_analysis=analysis,
        template_changes=template_changes,
        kept=kept,
        hypothesis=hypothesis,
        experiment=experiment,
        vision_feedback=vision_feedback,
        roundtrip_feedback=roundtrip_feedback,
        iteration_captions=captions,
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

    # Semaphores — one shared Gemini semaphore for captioning + generation + vision
    gemini_semaphore = asyncio.Semaphore(config.gemini_concurrency)
    eval_semaphore = asyncio.Semaphore(config.eval_concurrency)

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
        logger.info("Resumed from iteration %d with %d fixed references", state.iteration, len(state.fixed_references))
        fixed_refs = state.fixed_references
    else:
        # Fix 10 reference images for the entire process
        fixed_refs = _sample(all_ref_paths, 10)
        logger.info("Fixed %d reference images for optimization", len(fixed_refs))

        # Zero-step: caption with default prompt + analyze style
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
            anthropic_client=anthropic_client,
            caption_model=config.caption_model,
            claude_model=config.claude_model,
            cache_path=config.log_dir / "style_profile.json",
        )

        logger.info("Zero-step: proposing %d initial meta-prompts...", config.num_branches)
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
            fixed_references=fixed_refs,
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
                fixed_refs,
                config,
                gemini_client=gemini_client,
                anthropic_client=anthropic_client,
                registry=registry,
                gemini_semaphore=gemini_semaphore,
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
    logger.info("BEST META-PROMPT: %s", state.global_best_prompt)
    logger.info("Convergence: %s", state.convergence_reason)

    return state
