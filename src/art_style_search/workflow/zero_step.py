"""Zero-step workflow helpers."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from art_style_search.analyze import analyze_style
from art_style_search.caption import caption_references
from art_style_search.experiment import collect_experiment_results, run_experiment
from art_style_search.prompt import propose_initial_templates, validate_template
from art_style_search.scoring import composite_score
from art_style_search.state import save_iteration_log, save_state
from art_style_search.types import KnowledgeBase, LoopState, PromptTemplate
from art_style_search.workflow.context import (
    RunContext,
    _log_experiment_results,
    _ref_cache_key,
    _sample,
    _save_best_prompt,
    _split_information_barrier,
)
from art_style_search.workflow.policy import _apply_best_result

logger = logging.getLogger(__name__)


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


async def _zero_step(ctx: RunContext, all_ref_paths: list[Path]) -> LoopState:
    """Zero-step: fix refs, caption, analyze style, propose N initial templates, evaluate, apply best."""
    config = ctx.config
    fixed_refs = _sample(all_ref_paths, config.num_fixed_refs, rng=ctx.rng)
    logger.info("Fixed %d reference images for optimization", len(fixed_refs))

    feedback_refs, silent_refs = _split_information_barrier(fixed_refs, config.protocol, ctx.rng)
    if silent_refs:
        logger.info("Information barrier: %d feedback + %d silent images", len(feedback_refs), len(silent_refs))

    logger.info("Zero-step: captioning %d reference images...", len(fixed_refs))
    if ctx.services is None:
        captions = await caption_references(
            fixed_refs,
            model=config.caption_model,
            client=ctx.gemini_client,
            cache_dir=config.log_dir / "captions",
            semaphore=ctx.gemini_semaphore,
            cache_key="initial",
        )
    else:
        captions = await ctx.services.captioning.caption_references(
            fixed_refs,
            cache_dir=config.log_dir / "captions",
            cache_key="initial",
        )

    logger.info("Zero-step: analyzing art style...")
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
    cache_errors = validate_template(initial_template)
    if cache_errors:
        logger.warning("Cached style template invalid (%s) — re-running analysis", "; ".join(cache_errors))
        if shared_cache.exists():
            shared_cache.unlink()
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

    save_state(state, config.state_file)

    logger.info("=== Iteration 0 — evaluating %d initial templates ===", len(initial_templates))
    try:
        init_tasks = [
            run_experiment(
                experiment_id=i,
                template=template,
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
                services=ctx.services,
            )
            for i, template in enumerate(initial_templates)
        ]
        init_results = collect_experiment_results(await asyncio.gather(*init_tasks, return_exceptions=True), "Initial experiment")
    except Exception:
        logger.exception("Zero-step evaluation failed — partial state saved for resume")
        raise

    if init_results:
        best_init = max(init_results, key=lambda result: composite_score(result.aggregated))
        _apply_best_result(state, best_init)
        state.last_iteration_results = init_results
        state.experiment_history = list(init_results)
        _log_experiment_results(init_results, config.log_dir, save_iteration_log)

    state.iteration = 1
    save_state(state, config.state_file)
    _save_best_prompt(state, config.log_dir)
    return state
