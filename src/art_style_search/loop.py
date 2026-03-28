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
from dataclasses import dataclass
from pathlib import Path

from google import genai

from art_style_search.analyze import analyze_style
from art_style_search.caption import caption_references
from art_style_search.config import Config
from art_style_search.evaluate import check_caption_compliance, compare_vision, evaluate_images
from art_style_search.generate import _generate_single
from art_style_search.models import ModelRegistry
from art_style_search.prompt import Lessons, propose_initial_templates, refine_template, synthesize_templates
from art_style_search.state import load_state, save_iteration_log, save_state
from art_style_search.types import (
    AggregatedMetrics,
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


def _best_kept_result(results: list[IterationResult]) -> IterationResult | None:
    """Return the kept result from a list, or the first result, or None."""
    if not results:
        return None
    return next((r for r in results if r.kept), results[0])


def _collect_experiment_results(raw: list[IterationResult | BaseException], label: str) -> list[IterationResult]:
    """Filter successful results from asyncio.gather output, logging failures."""
    results: list[IterationResult] = []
    for r in raw:
        if isinstance(r, BaseException):
            logger.error("%s failed: %s: %s", label, type(r).__name__, r, exc_info=r)
        else:
            results.append(r)
    return results


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
            "Exp %d — DINO=%.3f LPIPS=%.3f SSIM=%.3f Color=%.3f HPS=%.3f Aes=%.1f V[S=%.0f Su=%.0f C=%.0f Co=%.0f] %s",
            r.branch_id,
            m.dino_similarity_mean,
            m.lpips_distance_mean,
            m.ssim_mean,
            m.color_histogram_mean,
            m.hps_score_mean,
            m.aesthetics_score_mean,
            m.vision_style,
            m.vision_subject,
            m.vision_color,
            m.vision_composition,
            "KEPT" if r.kept else "discarded",
        )


# ---------------------------------------------------------------------------
# Experiment proposal dataclass (replaces 8-element tuple)
# ---------------------------------------------------------------------------


@dataclass
class _ExperimentProposal:
    """Holds Claude's proposed experiment before it's executed."""

    template: PromptTemplate
    hypothesis: str
    experiment_desc: str
    builds_on: str | None
    open_problems: list[str]
    lessons: Lessons


# ---------------------------------------------------------------------------
# Captioning + generation + evaluation
# ---------------------------------------------------------------------------


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

    # Evaluate — per-image paired comparison
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

    # Vision comparison with per-pair captions — returns text + structured scores
    sorted_caption_texts = [c.text for c in sorted_captions]
    vision_feedback, vision_scores = await compare_vision(
        sorted_pairs,
        sorted_caption_texts,
        meta_prompt,
        client=gemini_client,
        model=config.caption_model,
        semaphore=gemini_semaphore,
        max_pairs=5,
    )

    # Merge vision scores into aggregated metrics
    from dataclasses import replace

    aggregated = replace(
        aggregated,
        vision_style=vision_scores.style.score,
        vision_subject=vision_scores.subject.score,
        vision_color=vision_scores.color.score,
        vision_composition=vision_scores.composition.score,
    )

    # Build roundtrip feedback — full caption for worst image, truncated for rest
    roundtrip_details: list[str] = []
    prev = _best_kept_result(last_results)
    prev_scores: dict[Path, float] = {}
    if prev:
        for cap, sc in zip(prev.iteration_captions, prev.per_image_scores, strict=False):
            prev_scores[cap.image_path] = sc.dino_similarity

    for idx, ((_orig, _gen), sc, cap) in enumerate(scored_items):
        prev_dino = prev_scores.get(cap.image_path)
        trend = ""
        if prev_dino is not None:
            arrow = "↑" if sc.dino_similarity > prev_dino else "↓" if sc.dino_similarity < prev_dino else "="
            trend = f" [prev DINO={prev_dino:.3f} → {sc.dino_similarity:.3f} {arrow}]"
        caption_text = cap.text if idx < 3 else f"{cap.text[:300]}..."
        roundtrip_details.append(
            f"Image ({_orig.name}): DINO={sc.dino_similarity:.3f} LPIPS={sc.lpips_distance:.3f} "
            f"SSIM={sc.ssim:.3f} Color={sc.color_histogram:.3f} "
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
        kept=False,
        hypothesis=hypothesis,
        experiment=experiment_desc,
        vision_feedback=vision_feedback,
        roundtrip_feedback=roundtrip_feedback,
        iteration_captions=captions,
    )


# ---------------------------------------------------------------------------
# Knowledge Base maintenance
# ---------------------------------------------------------------------------


def _update_knowledge_base(
    kb: KnowledgeBase,
    result: IterationResult,
    template: PromptTemplate,
    best_metrics: AggregatedMetrics | None,
    proposal: _ExperimentProposal,
    iteration: int,
) -> None:
    """Update the shared KB with one experiment's results."""
    parent_id: str | None = None
    if proposal.builds_on:
        parent_match = re.match(r"H(\d+)", proposal.builds_on)
        if parent_match:
            parent_id = f"H{parent_match.group(1)}"

    category_names = get_category_names(template)
    category = classify_hypothesis(result.hypothesis, category_names) if result.hypothesis else "general"

    metric_delta: dict[str, float] = {}
    if best_metrics is not None:
        metric_delta = {
            "dino": result.aggregated.dino_similarity_mean - best_metrics.dino_similarity_mean,
            "lpips": result.aggregated.lpips_distance_mean - best_metrics.lpips_distance_mean,
            "hps": result.aggregated.hps_score_mean - best_metrics.hps_score_mean,
            "aesthetics": result.aggregated.aesthetics_score_mean - best_metrics.aesthetics_score_mean,
            "ssim": result.aggregated.ssim_mean - best_metrics.ssim_mean,
            "color_histogram": result.aggregated.color_histogram_mean - best_metrics.color_histogram_mean,
            "vision_style": result.aggregated.vision_style - best_metrics.vision_style,
            "vision_subject": result.aggregated.vision_subject - best_metrics.vision_subject,
            "vision_color": result.aggregated.vision_color - best_metrics.vision_color,
            "vision_composition": result.aggregated.vision_composition - best_metrics.vision_composition,
        }

    lessons = proposal.lessons
    lesson_text = lessons.confirmed or lessons.new_insight or lessons.rejected or ""

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
            confirmed=lessons.confirmed,
            rejected=lessons.rejected,
        )

    if proposal.open_problems:
        scores = result.per_image_scores
        cat_dinos: dict[str, list[float]] = {}
        for sc in scores:
            cat_dinos.setdefault("all", []).append(sc.dino_similarity)
        best_cat_dino = max((sum(v) / len(v) for v in cat_dinos.values()), default=0.0)

        prev_problem_texts = {p.text: p.since_iteration for p in kb.open_problems}

        new_problems: list[OpenProblem] = []
        for prob_text in proposal.open_problems:
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
        # Auto-add open problems from low Gemini vision dimension scores
        agg = result.aggregated
        vision_dims = [
            ("style", agg.vision_style, "technique"),
            ("subject", agg.vision_subject, "subject_matter"),
            ("color", agg.vision_color, "color_palette"),
            ("composition", agg.vision_composition, "composition"),
        ]
        for dim_name, score, cat_name in vision_dims:
            if score < 5.0:
                # Find the assessment text from the vision feedback if available
                assessment = f"Vision {dim_name} score: {score:.0f}/10"
                prob_text = f"{dim_name.title()} fidelity: {assessment}"
                # Only add if not already present
                if not any(dim_name in p.text.lower() for p in new_problems):
                    new_problems.append(
                        OpenProblem(
                            text=prob_text,
                            category=cat_name,
                            priority="HIGH" if score < 3.0 else "MED",
                            metric_gap=float((5.0 - score) / 10.0),
                            since_iteration=iteration,
                        )
                    )

        kb.open_problems = new_problems


def _build_caption_diffs(last_results: list[IterationResult], worst_captions: list[Caption]) -> str:
    """Show how captions changed for worst-performing images vs previous iteration."""
    if not last_results or not worst_captions:
        return ""
    prev = _best_kept_result(last_results)
    if not prev:
        return ""
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

    # Shared kwargs for _run_experiment
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

        # Evaluate initial templates as iteration 0
        logger.info("=== Iteration 0 — evaluating %d initial templates ===", len(initial_templates))
        init_tasks = [
            _run_experiment(
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

        init_results = _collect_experiment_results(
            await asyncio.gather(*init_tasks, return_exceptions=True), "Initial experiment"
        )

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
        best_last = _best_kept_result(state.last_iteration_results)
        vision_fb = best_last.vision_feedback if best_last else ""
        roundtrip_fb = best_last.roundtrip_feedback if best_last else ""

        caption_diffs = ""
        if best_last and best_last.iteration_captions:
            sorted_caps = sorted(
                zip(best_last.iteration_captions, best_last.per_image_scores, strict=False),
                key=lambda x: x[1].dino_similarity,
            )
            worst_caps = [c for c, _ in sorted_caps[:3]]
            caption_diffs = _build_caption_diffs(state.last_iteration_results, worst_caps)

        # Phase 1: Claude proposes N experiments sequentially (for dedup)
        proposals: list[_ExperimentProposal] = []
        proposed_hypotheses: list[str] = []

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
                client=reasoning_client,
                model=config.reasoning_model,
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
            proposals.append(
                _ExperimentProposal(
                    template=new_template,
                    hypothesis=hypothesis,
                    experiment_desc=experiment_desc,
                    builds_on=builds_on,
                    open_problems=open_problems,
                    lessons=lessons,
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
            _run_experiment(
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

        exp_results = _collect_experiment_results(
            await asyncio.gather(*exp_tasks, return_exceptions=True), "Experiment"
        )

        if not exp_results:
            logger.warning("All experiments failed this iteration")
            state.plateau_counter += 1
            if state.plateau_counter >= config.plateau_window:
                state.converged = True
                state.convergence_reason = ConvergenceReason.PLATEAU
                break
            continue

        # Phase 3: Find best experiment using adaptive scoring
        from art_style_search.types import adaptive_composite_score

        all_agg = [r.aggregated for r in exp_results]
        best_exp = max(exp_results, key=lambda r: adaptive_composite_score(r.aggregated, all_agg))
        best_score = adaptive_composite_score(best_exp.aggregated, all_agg)
        baseline_score = composite_score(state.best_metrics) if state.best_metrics else float("-inf")

        # Phase 3.5: Synthesis — merge top experiments if multiple improved
        improved_exps = [r for r in exp_results if adaptive_composite_score(r.aggregated, all_agg) > baseline_score]
        if len(improved_exps) >= 2:
            logger.info("Synthesizing %d improved experiments into merged template", len(improved_exps))
            improved_exps.sort(key=lambda r: adaptive_composite_score(r.aggregated, all_agg), reverse=True)
            top_exps = improved_exps[:3]

            merged_template, merged_hypothesis = await synthesize_templates(
                top_exps,
                state.style_profile,
                client=reasoning_client,
                model=config.reasoning_model,
            )

            # Validate the merged template
            merged_result = await _run_experiment(
                experiment_id=len(exp_results),
                template=merged_template,
                iteration=iteration,
                fixed_refs=fixed_refs,
                last_results=state.last_iteration_results,
                hypothesis=merged_hypothesis,
                experiment_desc="Synthesis of top experiments",
                **exp_kwargs,
            )
            merged_score = composite_score(merged_result.aggregated)
            logger.info(
                "Synthesis result: DINO=%.4f (best individual: %.4f)",
                merged_result.aggregated.dino_similarity_mean,
                best_exp.aggregated.dino_similarity_mean,
            )

            exp_results.append(merged_result)
            if merged_score > best_score:
                best_exp = merged_result
                best_score = merged_score
                logger.info("Synthesis beat best individual — adopting merged template")

        # Update state with best result
        improved = best_score > baseline_score

        if improved:
            _apply_best_result(state, best_exp)
            state.plateau_counter = 0
        else:
            state.plateau_counter += 1

        state.last_iteration_results = exp_results
        state.experiment_history.extend(exp_results)
        # Cap persisted history to avoid unbounded state.json growth
        if len(state.experiment_history) > _MAX_PERSISTED_HISTORY:
            state.experiment_history = state.experiment_history[-_MAX_PERSISTED_HISTORY:]

        # Phase 4: Update shared KB with ALL experiment results
        for exp_result, proposal in zip(exp_results, proposals, strict=False):
            _update_knowledge_base(
                state.knowledge_base,
                exp_result,
                exp_result.template,
                state.best_metrics,
                proposal,
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
