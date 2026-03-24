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
import re
from pathlib import Path

import anthropic
from google import genai

from art_style_search.analyze import analyze_style
from art_style_search.caption import caption_references
from art_style_search.config import Config
from art_style_search.evaluate import check_caption_compliance, compare_vision, evaluate_images
from art_style_search.generate import _generate_single
from art_style_search.models import ModelRegistry
from art_style_search.prompt import propose_initial_templates, refine_template
from art_style_search.state import load_state, save_iteration_log, save_state
from art_style_search.types import (
    Caption,
    ConvergenceReason,
    IterationResult,
    KnowledgeBase,
    LoopState,
    OpenProblem,
    PromptTemplate,
    classify_hypothesis,
    composite_score,
    get_category_names,
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


def _get_previous_per_image_scores(last_results: list[IterationResult]) -> dict[Path, float]:
    """Get per-image DINO scores from the best previous experiment."""
    if not last_results:
        return {}
    # Use the kept result if one exists, else the first
    prev = next((r for r in last_results if r.kept), last_results[0])
    result: dict[Path, float] = {}
    for cap, sc in zip(prev.iteration_captions, prev.per_image_scores, strict=False):
        result[cap.image_path] = sc.dino_similarity
    return result


def _build_caption_diffs(last_results: list[IterationResult], worst_captions: list[Caption]) -> str:
    """Show how captions changed for worst-performing images vs previous iteration."""
    if not last_results or not worst_captions:
        return ""
    prev = next((r for r in last_results if r.kept), last_results[0])
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
    experiment_id: int,
) -> tuple[list[Caption], list[Path], list[tuple[Path, Path]]]:
    """Caption reference images with the meta-prompt, then generate images from captions.

    Returns (captions, generated_paths, pairs) where pairs maps (original, generated).
    """
    captions = await caption_references(
        ref_paths,
        model=config.caption_model,
        client=gemini_client,
        cache_dir=config.log_dir / f"iter_{iteration:03d}" / f"exp_{experiment_id}" / "captions",
        semaphore=gemini_semaphore,
        prompt=meta_prompt,
        cache_key=f"iter{iteration}_e{experiment_id}",
    )

    gen_dir = config.output_dir / f"iter_{iteration:03d}" / f"exp_{experiment_id}"
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

    generated_paths: list[Path] = []
    pairs: list[tuple[Path, Path]] = []
    for i, (caption, gen_result) in enumerate(zip(captions, gen_results, strict=True)):
        if isinstance(gen_result, BaseException):
            logger.warning("Exp %d: generation from caption %d failed: %s", experiment_id, i, gen_result)
        else:
            generated_paths.append(gen_result)
            pairs.append((caption.image_path, gen_result))

    return captions, generated_paths, pairs


async def _run_experiment(
    experiment_id: int,
    template: PromptTemplate,
    iteration: int,
    fixed_refs: list[Path],
    config: Config,
    *,
    gemini_client: genai.Client,
    registry: ModelRegistry,
    gemini_semaphore: asyncio.Semaphore,
    eval_semaphore: asyncio.Semaphore,
    last_results: list[IterationResult],
    hypothesis: str = "",
    experiment_desc: str = "",
) -> IterationResult:
    """Execute one experiment: caption → generate → evaluate (no Claude call here)."""
    meta_prompt = template.render()
    logger.info("Exp %d iter %d — meta-prompt: %.100s...", experiment_id, iteration, meta_prompt)

    # Phase 1: Caption + generate
    captions, generated_paths, pairs = await _caption_and_generate(
        fixed_refs,
        meta_prompt,
        config=config,
        gemini_client=gemini_client,
        gemini_semaphore=gemini_semaphore,
        iteration=iteration,
        experiment_id=experiment_id,
    )

    if not generated_paths:
        raise RuntimeError(f"Experiment {experiment_id}: no images generated")

    logger.info(
        "Exp %d iter %d — %d/%d images generated", experiment_id, iteration, len(generated_paths), len(fixed_refs)
    )

    # Phase 2: Evaluate — per-image paired comparison
    gen_paths_for_eval = [gen for _, gen in pairs]
    ref_paths_for_eval = [orig for orig, _ in pairs]
    caption_by_path = {c.image_path: c.text for c in captions}
    eval_captions = [caption_by_path[orig] for orig, _ in pairs]

    eval_task = evaluate_images(
        gen_paths_for_eval,
        ref_paths_for_eval,
        eval_captions,
        registry=registry,
        semaphore=eval_semaphore,
    )

    section_names = [s.name for s in template.sections]
    compliance = check_caption_compliance(section_names, captions)

    scores, aggregated = await eval_task

    # Sort by DINO worst first
    scored_items = list(zip(pairs, scores, captions, strict=False))
    scored_items.sort(key=lambda x: x[1].dino_similarity)

    sorted_pairs = [item[0] for item in scored_items]
    sorted_captions = [item[2] for item in scored_items]

    # Vision comparison with per-pair captions
    sorted_caption_texts = [c.text for c in sorted_captions]
    vision_feedback = await compare_vision(
        sorted_pairs,
        sorted_caption_texts,
        meta_prompt,
        client=gemini_client,
        model=config.caption_model,
        semaphore=gemini_semaphore,
        max_pairs=5,
    )

    # Build roundtrip feedback — full captions for worst 3
    roundtrip_details: list[str] = []
    prev_scores = _get_previous_per_image_scores(last_results)
    for idx, ((_orig, _gen), sc, cap) in enumerate(scored_items):
        prev_dino = prev_scores.get(cap.image_path, None)
        trend = ""
        if prev_dino is not None:
            arrow = "↑" if sc.dino_similarity > prev_dino else "↓" if sc.dino_similarity < prev_dino else "="
            trend = f" [prev DINO={prev_dino:.3f} → {sc.dino_similarity:.3f} {arrow}]"
        caption_text = cap.text if idx < 3 else f"{cap.text[:300]}..."
        roundtrip_details.append(
            f"Image ({_orig.name}): DINO={sc.dino_similarity:.3f} LPIPS={sc.lpips_distance:.3f} "
            f"HPS={sc.hps_score:.3f} Aes={sc.aesthetics_score:.1f}{trend}\n"
            f"  Caption: {caption_text}"
        )
    roundtrip_feedback = "\n".join(roundtrip_details)
    if compliance:
        roundtrip_feedback = compliance + "\n\n" + roundtrip_feedback

    return IterationResult(
        branch_id=experiment_id,
        iteration=iteration,
        template=template,
        rendered_prompt=meta_prompt,
        image_paths=generated_paths,
        per_image_scores=scores,
        aggregated=aggregated,
        claude_analysis="",
        template_changes="",
        kept=False,  # determined later by caller
        hypothesis=hypothesis,
        experiment=experiment_desc,
        vision_feedback=vision_feedback,
        roundtrip_feedback=roundtrip_feedback,
        iteration_captions=captions,
    )


def _update_knowledge_base(
    kb: KnowledgeBase,
    result: IterationResult,
    template: PromptTemplate,
    best_metrics: object | None,
    lessons_confirmed: str,
    lessons_rejected: str,
    lessons_new_insight: str,
    builds_on: str | None,
    raw_open_problems: list[str],
    iteration: int,
) -> None:
    """Update the shared KB with one experiment's results."""
    # Parse parent hypothesis
    parent_id: str | None = None
    if builds_on:
        parent_match = re.match(r"H(\d+)", builds_on)
        if parent_match:
            parent_id = f"H{parent_match.group(1)}"

    category_names = get_category_names(template)
    category = classify_hypothesis(result.hypothesis, category_names) if result.hypothesis else "general"

    # Metric deltas
    metric_delta: dict[str, float] = {}
    if best_metrics is not None:
        from art_style_search.types import AggregatedMetrics

        if isinstance(best_metrics, AggregatedMetrics):
            metric_delta = {
                "dino": result.aggregated.dino_similarity_mean - best_metrics.dino_similarity_mean,
                "lpips": result.aggregated.lpips_distance_mean - best_metrics.lpips_distance_mean,
                "hps": result.aggregated.hps_score_mean - best_metrics.hps_score_mean,
                "aesthetics": result.aggregated.aesthetics_score_mean - best_metrics.aesthetics_score_mean,
            }

    lesson_text = lessons_confirmed or lessons_new_insight or lessons_rejected or ""

    if result.hypothesis:
        kb.add_hypothesis(
            iteration=iteration,
            parent_id=parent_id,
            statement=result.hypothesis,
            experiment=result.experiment,
            category=category,
            kept=result.kept,
            metric_delta=metric_delta,
            lesson=lesson_text,
            confirmed=lessons_confirmed,
            rejected=lessons_rejected,
        )

    # Update open problems
    if raw_open_problems:
        scores = result.per_image_scores
        cat_dinos: dict[str, list[float]] = {}
        for sc in scores:
            cat_dinos.setdefault("all", []).append(sc.dino_similarity)
        best_cat_dino = max((sum(v) / len(v) for v in cat_dinos.values()), default=0.0)

        prev_problem_texts = {p.text: p.since_iteration for p in kb.open_problems}

        new_problems: list[OpenProblem] = []
        for prob_text in raw_open_problems:
            prob_cat = classify_hypothesis(prob_text, category_names)
            cat_progress = kb.categories.get(prob_cat)

            if cat_progress is None or not cat_progress.confirmed_insights:
                priority = "HIGH"
            elif cat_progress.rejected_approaches and len(cat_progress.rejected_approaches) >= len(
                cat_progress.confirmed_insights
            ):
                priority = "MED"
            else:
                priority = "LOW"

            gap = best_cat_dino - (
                cat_progress.best_dino_delta if cat_progress and cat_progress.best_dino_delta else 0.0
            )
            since = prev_problem_texts.get(prob_text, iteration)

            new_problems.append(
                OpenProblem(text=prob_text, category=prob_cat, priority=priority, metric_gap=gap, since_iteration=since)
            )
        kb.open_problems = new_problems


async def run(config: Config) -> LoopState:
    """Run the full optimization loop."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=config.eval_concurrency))

    gemini_client = genai.Client(api_key=config.google_api_key)
    anthropic_client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    gemini_semaphore = asyncio.Semaphore(config.gemini_concurrency)
    eval_semaphore = asyncio.Semaphore(config.eval_concurrency)

    logger.info("Loading evaluation models...")
    registry = await asyncio.to_thread(ModelRegistry.load_all)

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
        # Fix reference images
        fixed_refs = _sample(all_ref_paths, 10)
        logger.info("Fixed %d reference images for optimization", len(fixed_refs))

        # Zero-step: caption + analyze style
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

        # Propose initial diverse templates (used as first iteration's experiments)
        logger.info("Zero-step: proposing %d initial meta-prompts...", config.num_branches)
        initial_templates = await propose_initial_templates(
            style_profile,
            config.num_branches,
            client=anthropic_client,
            model=config.claude_model,
        )

        for i, t in enumerate(initial_templates):
            if not t.sections:
                initial_templates[i] = initial_template

        # Pick the first template as starting point
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

        # Run initial templates as experiment 0
        logger.info("=== Iteration 0 — evaluating %d initial templates ===", len(initial_templates))
        init_tasks = [
            _run_experiment(
                experiment_id=i,
                template=t,
                iteration=0,
                fixed_refs=fixed_refs,
                config=config,
                gemini_client=gemini_client,
                registry=registry,
                gemini_semaphore=gemini_semaphore,
                eval_semaphore=eval_semaphore,
                last_results=[],
                hypothesis=f"Initial template {i}",
                experiment_desc="Zero-step diverse template",
            )
            for i, t in enumerate(initial_templates)
        ]

        init_results_raw = await asyncio.gather(*init_tasks, return_exceptions=True)
        init_results: list[IterationResult] = []
        for r in init_results_raw:
            if isinstance(r, BaseException):
                logger.error("Initial experiment failed: %s", r)
            else:
                init_results.append(r)

        # Find best initial template
        if init_results:
            best_init = max(init_results, key=lambda r: composite_score(r.aggregated))
            best_init.kept = True
            state.current_template = best_init.template
            state.best_template = best_init.template
            state.best_metrics = best_init.aggregated
            state.global_best_prompt = best_init.rendered_prompt
            state.global_best_metrics = best_init.aggregated
            state.last_iteration_results = init_results
            state.experiment_history = list(init_results)

            for r in init_results:
                save_iteration_log(r, config.log_dir)
                m = r.aggregated
                logger.info(
                    "Exp %d — DINO=%.4f LPIPS=%.4f HPS=%.4f Aes=%.2f %s",
                    r.branch_id,
                    m.dino_similarity_mean,
                    m.lpips_distance_mean,
                    m.hps_score_mean,
                    m.aesthetics_score_mean,
                    "KEPT" if r.kept else "discarded",
                )

        state.iteration = 1
        save_state(state, config.state_file)

    # Main optimization loop
    for iteration in range(state.iteration, config.max_iterations):
        state.iteration = iteration

        logger.info("=== Iteration %d/%d ===", iteration + 1, config.max_iterations)

        # Phase 1: Claude proposes N experiments sequentially (for dedup)
        # Use the best experiment's feedback from last iteration for context
        best_last = (
            next((r for r in state.last_iteration_results if r.kept), None) if state.last_iteration_results else None
        )
        vision_fb = best_last.vision_feedback if best_last else ""
        roundtrip_fb = best_last.roundtrip_feedback if best_last else ""

        # Caption diffs from best last result
        caption_diffs = ""
        if best_last and best_last.iteration_captions:
            sorted_caps = sorted(
                zip(best_last.iteration_captions, best_last.per_image_scores, strict=False),
                key=lambda x: x[1].dino_similarity,
            )
            worst_caps = [c for c, _ in sorted_caps[:3]]
            caption_diffs = _build_caption_diffs(state.last_iteration_results, worst_caps)

        proposed_hypotheses: list[str] = []
        experiment_templates: list[tuple[PromptTemplate, str, str, str | None, list[str], str, str, str]] = []

        for exp_idx in range(config.num_branches):
            (
                new_template,
                _analysis,
                _template_changes,
                should_stop,
                hypothesis,
                experiment_desc,
                lessons,
                builds_on,
                open_problems,
            ) = await refine_template(
                state.style_profile,
                state.current_template,
                state.knowledge_base,
                state.best_metrics,
                state.last_iteration_results,
                client=anthropic_client,
                model=config.claude_model,
                vision_feedback=vision_fb,
                roundtrip_feedback=roundtrip_fb,
                caption_diffs=caption_diffs,
                already_proposed=proposed_hypotheses if proposed_hypotheses else None,
            )

            if should_stop:
                logger.info("Claude signaled convergence at experiment %d", exp_idx)
                state.converged = True
                state.convergence_reason = ConvergenceReason.CLAUDE_STOP
                break

            proposed_hypotheses.append(hypothesis)
            experiment_templates.append(
                (
                    new_template,
                    hypothesis,
                    experiment_desc,
                    builds_on,
                    open_problems,
                    lessons.confirmed,
                    lessons.rejected,
                    lessons.new_insight,
                )
            )

        if state.converged:
            break

        if not experiment_templates:
            logger.warning("No experiments proposed — stopping")
            state.converged = True
            state.convergence_reason = ConvergenceReason.CLAUDE_STOP
            break

        # Phase 2: Run all experiments in parallel
        exp_tasks = [
            _run_experiment(
                experiment_id=i,
                template=tpl,
                iteration=iteration,
                fixed_refs=fixed_refs,
                config=config,
                gemini_client=gemini_client,
                registry=registry,
                gemini_semaphore=gemini_semaphore,
                eval_semaphore=eval_semaphore,
                last_results=state.last_iteration_results,
                hypothesis=hyp,
                experiment_desc=exp_desc,
            )
            for i, (tpl, hyp, exp_desc, _bo, _op, _c, _r, _n) in enumerate(experiment_templates)
        ]

        exp_results_raw = await asyncio.gather(*exp_tasks, return_exceptions=True)
        exp_results: list[IterationResult] = []
        for r in exp_results_raw:
            if isinstance(r, BaseException):
                logger.error("Experiment failed: %s", r)
            else:
                exp_results.append(r)

        if not exp_results:
            logger.warning("All experiments failed this iteration")
            state.plateau_counter += 1
            if state.plateau_counter >= config.plateau_window:
                state.converged = True
                state.convergence_reason = ConvergenceReason.PLATEAU
                break
            continue

        # Phase 3: Find best experiment and update state
        best_exp = max(exp_results, key=lambda r: composite_score(r.aggregated))
        best_score = composite_score(best_exp.aggregated)
        improved = state.best_metrics is None or best_score > composite_score(state.best_metrics)

        if improved:
            best_exp.kept = True
            state.current_template = best_exp.template
            state.best_template = best_exp.template
            state.best_metrics = best_exp.aggregated
            state.global_best_prompt = best_exp.rendered_prompt
            state.global_best_metrics = best_exp.aggregated
            state.plateau_counter = 0
        else:
            state.plateau_counter += 1

        state.last_iteration_results = exp_results
        state.experiment_history.extend(exp_results)

        # Phase 4: Update shared KB with ALL experiment results
        for exp_result, exp_data in zip(exp_results, experiment_templates, strict=False):
            _, _, _, builds_on, open_probs, confirmed, rejected, new_insight = exp_data
            _update_knowledge_base(
                state.knowledge_base,
                exp_result,
                exp_result.template,
                state.best_metrics,
                confirmed,
                rejected,
                new_insight,
                builds_on,
                open_probs,
                iteration,
            )

        # Log and save
        for r in exp_results:
            save_iteration_log(r, config.log_dir)
            m = r.aggregated
            logger.info(
                "Exp %d — DINO=%.4f LPIPS=%.4f HPS=%.4f Aes=%.2f %s",
                r.branch_id,
                m.dino_similarity_mean,
                m.lpips_distance_mean,
                m.hps_score_mean,
                m.aesthetics_score_mean,
                "KEPT" if r.kept else "discarded",
            )

        save_state(state, config.state_file)

        # Convergence check
        if state.plateau_counter >= config.plateau_window:
            logger.info("Plateau detected (%d iterations without improvement)", state.plateau_counter)
            state.converged = True
            state.convergence_reason = ConvergenceReason.PLATEAU
            break

    else:
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
    logger.info("Total experiments: %d", len(state.experiment_history))
    logger.info("KB: %d hypotheses", len(state.knowledge_base.hypotheses))

    return state
