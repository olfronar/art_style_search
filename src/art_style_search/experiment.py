"""Single-experiment execution: caption, generate, evaluate."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import statistics
from dataclasses import replace
from pathlib import Path

from art_style_search.caption import CAPTION_SYSTEM
from art_style_search.caption_sections import build_generation_prompt, extract_style_invariants
from art_style_search.config import Config
from art_style_search.evaluate import (
    aggregate,
    compute_caption_compliance,
    compute_style_consistency,
)
from art_style_search.knowledge import aggregate_style_gap_notes
from art_style_search.types import (
    AggregatedMetrics,
    Caption,
    DirectionId,
    IterationResult,
    MetricScores,
    PromptTemplate,
    ReplicatedEvaluation,
    RiskLevel,
    VisionScores,
    verdict_label,
)
from art_style_search.workflow.services import RunServices

logger = logging.getLogger(__name__)

# Reject experiments where fewer than this fraction of images were generated
_MIN_COMPLETION_RATE = 0.5


def _merge_vision(ms: MetricScores, vs: VisionScores) -> MetricScores:
    """Merge per-image vision scores into a MetricScores instance."""
    return replace(
        ms,
        vision_style=vs.style.score,
        vision_subject=vs.subject.score,
        vision_composition=vs.composition.score,
        vision_medium=vs.medium.score,
        vision_proportions=vs.proportions.score,
        style_gap=vs.style_gap,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_experiment_results(raw: list[IterationResult | BaseException], label: str) -> list[IterationResult]:
    """Filter successful results from asyncio.gather output, logging failures."""
    results: list[IterationResult] = []
    for r in raw:
        if isinstance(r, BaseException):
            logger.error("%s failed: %s: %s", label, type(r).__name__, r, exc_info=r)
        else:
            results.append(r)
    return results


def best_kept_result(results: list[IterationResult]) -> IterationResult | None:
    """Return the kept result from a list, or the first result, or None."""
    if not results:
        return None
    return next((r for r in results if r.kept), results[0])


def _format_experiment_feedback(
    original_scores: list[MetricScores],
    vision_feedbacks: list[str],
    captions: list[Caption],
    pairs: list[tuple[Path, Path]],
    last_results: list[IterationResult],
    compliance: str,
) -> tuple[str, str]:
    """Build vision and roundtrip feedback strings sorted by DreamSim worst-first."""
    order = sorted(range(len(original_scores)), key=lambda i: original_scores[i].dreamsim_similarity)

    # Vision feedback
    vision_parts: list[str] = []
    for i in order:
        sc, fb = original_scores[i], vision_feedbacks[i]
        ref_path = pairs[i][0]
        vl = (
            f"S={verdict_label(sc.vision_style)} "
            f"Su={verdict_label(sc.vision_subject)} "
            f"Co={verdict_label(sc.vision_composition)} "
            f"Me={verdict_label(sc.vision_medium)} "
            f"Pr={verdict_label(sc.vision_proportions)}"
        )
        vision_parts.append(f"**{ref_path.name}** [{vl}]: {fb[:300]}")
    vision_feedback = "\n".join(vision_parts)

    # Roundtrip feedback — full caption for worst images, truncated for rest
    sorted_pairs = [pairs[i] for i in order]
    sorted_captions_list = [captions[i] for i in order]
    sorted_scores = [original_scores[i] for i in order]

    prev = best_kept_result(last_results)
    prev_scores: dict[Path, float] = {}
    if prev:
        for cap, sc in zip(prev.iteration_captions, prev.per_image_scores, strict=False):
            prev_scores[cap.image_path] = sc.dreamsim_similarity

    roundtrip_details: list[str] = []
    for idx, ((ref_p, _), sc, cap) in enumerate(zip(sorted_pairs, sorted_scores, sorted_captions_list, strict=True)):
        prev_ds = prev_scores.get(cap.image_path)
        trend = ""
        if prev_ds is not None:
            arrow = "↑" if sc.dreamsim_similarity > prev_ds else "↓" if sc.dreamsim_similarity < prev_ds else "="
            trend = f" [prev DS={prev_ds:.3f} → {sc.dreamsim_similarity:.3f} {arrow}]"
        vl = (
            f"V[S={verdict_label(sc.vision_style)} "
            f"Su={verdict_label(sc.vision_subject)} "
            f"Co={verdict_label(sc.vision_composition)} "
            f"Me={verdict_label(sc.vision_medium)} "
            f"Pr={verdict_label(sc.vision_proportions)}]"
        )
        caption_text = cap.text if idx < 3 else f"{cap.text[:300]}..."
        roundtrip_details.append(
            f"Image ({ref_p.name}): DS={sc.dreamsim_similarity:.3f} "
            f"Color={sc.color_histogram:.3f} SSIM={sc.ssim:.3f} "
            f"HPS={sc.hps_score:.3f} Aes={sc.aesthetics_score:.1f} "
            f"Mega={sc.megastyle_similarity:.3f} {vl}{trend}\n"
            f"  Caption: {caption_text}"
        )
    roundtrip_feedback = "\n".join(roundtrip_details)
    if compliance:
        roundtrip_feedback = compliance + "\n\n" + roundtrip_feedback

    return vision_feedback, roundtrip_feedback


# ---------------------------------------------------------------------------
# Captioning + generation + evaluation
# ---------------------------------------------------------------------------


async def _caption_and_generate(
    ref_paths: list[Path],
    meta_prompt: str,
    *,
    negative_prompt: str | None,
    style_canon: str,
    config: Config,
    services: RunServices,
    iteration: int,
    experiment_id: int,
) -> tuple[list[Caption], list[Path], list[tuple[Path, Path]]]:
    """Caption and generate per-image in a pipeline (no serial boundary).

    Each image's caption→generate runs as a single chained task so generation
    starts as soon as each individual caption completes. Returns (captions,
    generated_paths, pairs) where pairs maps (original, generated).

    ``style_canon`` is the meta-prompt's ``style_foundation`` value; it's passed to
    ``build_generation_prompt`` so the generator sees the canonical style assertions
    even when the captioner's ``[Art Style]`` block is missing or truncated.
    """
    cache_dir = config.log_dir / f"iter_{iteration:03d}" / f"exp_{experiment_id}" / "captions"
    gen_dir = config.output_dir / f"iter_{iteration:03d}" / f"exp_{experiment_id}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    # Cache key bundles (meta_prompt, CAPTION_SYSTEM) so a captioner-contract change invalidates stale entries.
    caption_system_digest = hashlib.sha256(CAPTION_SYSTEM.encode()).hexdigest()[:4]
    cache_key = f"p{hashlib.sha256(meta_prompt.encode()).hexdigest()[:12]}-c{caption_system_digest}"

    # Style Invariants (MUST/NEVER rules) flow to the generator so they survive a paraphrased caption.
    style_invariants = extract_style_invariants(style_canon)

    async def _caption_then_generate(ref_path: Path, i: int) -> tuple[Caption, Path]:
        caption = await services.captioning.caption_single(
            ref_path,
            prompt=meta_prompt,
            cache_dir=cache_dir,
            cache_key=cache_key,
            style_canon=style_canon,
        )
        gen_path = await services.generation.generate_single(
            build_generation_prompt(caption.text, style_canon=style_canon),
            index=i,
            output_path=gen_dir / f"{i:02d}.png",
            negative_prompt=negative_prompt,
            style_invariants=style_invariants,
        )
        return caption, gen_path

    results = await asyncio.gather(
        *[_caption_then_generate(p, i) for i, p in enumerate(ref_paths)],
        return_exceptions=True,
    )

    captions: list[Caption] = []
    generated_paths: list[Path] = []
    pairs: list[tuple[Path, Path]] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning("Exp %d: image %d (%s) failed: %s", experiment_id, i, ref_paths[i].name, result)
        else:
            caption, gen_path = result
            captions.append(caption)
            generated_paths.append(gen_path)
            pairs.append((caption.image_path, gen_path))

    return captions, generated_paths, pairs


async def run_experiment(
    experiment_id: int,
    template: PromptTemplate,
    iteration: int,
    fixed_refs: list[Path],
    config: Config,
    *,
    services: RunServices,
    last_results: list[IterationResult],
    hypothesis: str = "",
    experiment_desc: str = "",
    analysis: str = "",
    template_changes: str = "",
    changed_section: str = "",
    changed_sections: list[str] | None = None,
    target_category: str = "",
    direction_id: DirectionId | str = "",
    direction_summary: str = "",
    failure_mechanism: str = "",
    intervention_type: str = "",
    risk_level: RiskLevel | str = "targeted",
    expected_primary_metric: str = "",
    expected_tradeoff: str = "",
) -> IterationResult:
    """Execute one experiment: caption -> generate -> evaluate (no reasoning-model call here)."""
    meta_prompt = template.render()
    style_canon = next((s.value for s in template.sections if s.name == "style_foundation"), "")
    logger.info("Exp %d iter %d — meta-prompt: %.100s...", experiment_id, iteration, meta_prompt)

    captions, generated_paths, pairs = await _caption_and_generate(
        fixed_refs,
        meta_prompt,
        negative_prompt=template.negative_prompt,
        style_canon=style_canon,
        config=config,
        services=services,
        iteration=iteration,
        experiment_id=experiment_id,
    )

    n_attempted = len(fixed_refs)
    n_succeeded = len(generated_paths)

    if not generated_paths:
        raise RuntimeError(f"Experiment {experiment_id}: no images generated")

    completion_rate_gen = n_succeeded / n_attempted
    if completion_rate_gen < _MIN_COMPLETION_RATE:
        raise RuntimeError(
            f"Experiment {experiment_id}: only {n_succeeded}/{n_attempted} "
            f"images generated ({completion_rate_gen:.0%}), below {_MIN_COMPLETION_RATE:.0%} threshold"
        )

    logger.info(
        "Exp %d iter %d — %d/%d images generated", experiment_id, iteration, len(generated_paths), len(fixed_refs)
    )

    gen_paths_for_eval = [gen for _, gen in pairs]
    ref_paths_for_eval = [orig for orig, _ in pairs]
    caption_by_path = {c.image_path: c.text for c in captions}
    eval_captions = [caption_by_path[orig] for orig, _ in pairs]

    # Run metric evaluation and vision comparison in parallel
    (metric_scores, n_eval_failed), (vision_feedbacks, vision_scores_list) = await asyncio.gather(
        services.evaluation.evaluate_images(gen_paths_for_eval, ref_paths_for_eval, eval_captions),
        services.evaluation.compare_vision_per_image(pairs, eval_captions),
    )

    section_names = [s.name for s in template.sections]
    compliance_stats, compliance = compute_caption_compliance(
        section_names,
        captions,
        caption_sections=template.caption_sections,
        meta_prompt=meta_prompt,
    )

    # Merge vision scores in original order (aligned with pairs/captions/paths)
    original_scores = [_merge_vision(ms, vs) for ms, vs in zip(metric_scores, vision_scores_list, strict=True)]

    # Completion rate accounts for both generation and evaluation failures
    total_succeeded = max(n_succeeded - n_eval_failed, 0)
    completion_rate = total_succeeded / n_attempted

    aggregated = aggregate(original_scores, completion_rate=completion_rate)
    style_con = compute_style_consistency(captions)
    style_gap_notes = aggregate_style_gap_notes([sc.style_gap for sc in original_scores])
    aggregated = replace(
        aggregated,
        style_consistency=style_con,
        compliance_topic_coverage=compliance_stats.section_topic_coverage,
        compliance_marker_coverage=compliance_stats.section_marker_coverage,
        section_ordering_rate=compliance_stats.section_ordering_rate,
        section_balance_rate=compliance_stats.section_balance_rate,
        subject_specificity_rate=compliance_stats.subject_specificity_rate,
        style_canon_fidelity=compliance_stats.style_canon_fidelity,
        observation_boilerplate_purity=compliance_stats.observation_boilerplate_purity,
        style_gap_notes=style_gap_notes,
        requested_ref_count=config.num_fixed_refs,
        actual_ref_count=n_attempted,
    )

    vision_feedback, roundtrip_feedback = _format_experiment_feedback(
        original_scores,
        vision_feedbacks,
        captions,
        pairs,
        last_results,
        compliance,
    )

    return IterationResult(
        branch_id=experiment_id,
        iteration=iteration,
        template=template,
        rendered_prompt=meta_prompt,
        image_paths=generated_paths,
        per_image_scores=original_scores,
        aggregated=aggregated,
        claude_analysis=analysis,
        template_changes=template_changes,
        kept=False,
        hypothesis=hypothesis,
        experiment=experiment_desc,
        vision_feedback=vision_feedback,
        roundtrip_feedback=roundtrip_feedback,
        iteration_captions=captions,
        n_images_attempted=n_attempted,
        n_images_succeeded=n_succeeded,
        changed_section=changed_section,
        target_category=target_category,
        changed_sections=list(changed_sections or ([changed_section] if changed_section else [])),
        direction_id=direction_id,
        direction_summary=direction_summary,
        failure_mechanism=failure_mechanism,
        intervention_type=intervention_type,
        risk_level=risk_level,
        expected_primary_metric=expected_primary_metric,
        expected_tradeoff=expected_tradeoff,
    )


# ---------------------------------------------------------------------------
# Replicated evaluation (A1 paired-replicate promotion gate + short-protocol synthesis)
# ---------------------------------------------------------------------------


def _median_metric_scores(replicate_scores: list[list[MetricScores]]) -> list[MetricScores]:
    """Compute per-image median across replicates.

    ``replicate_scores[r][i]`` is replicate *r*, image *i*.
    Returns a list of length n_images with median scores.
    """

    n_images = len(replicate_scores[0])
    result: list[MetricScores] = []
    for img_idx in range(n_images):
        scores_across_reps = [replicate_scores[r][img_idx] for r in range(len(replicate_scores))]
        result.append(
            MetricScores(
                dreamsim_similarity=statistics.median(s.dreamsim_similarity for s in scores_across_reps),
                hps_score=statistics.median(s.hps_score for s in scores_across_reps),
                aesthetics_score=statistics.median(s.aesthetics_score for s in scores_across_reps),
                color_histogram=statistics.median(s.color_histogram for s in scores_across_reps),
                ssim=statistics.median(s.ssim for s in scores_across_reps),
                vision_style=statistics.median(s.vision_style for s in scores_across_reps),
                vision_subject=statistics.median(s.vision_subject for s in scores_across_reps),
                vision_composition=statistics.median(s.vision_composition for s in scores_across_reps),
                vision_medium=statistics.median(s.vision_medium for s in scores_across_reps),
                vision_proportions=statistics.median(s.vision_proportions for s in scores_across_reps),
            )
        )
    return result


async def replicate_experiment(
    template: PromptTemplate,
    branch_id: int,
    iteration: int,
    fixed_refs: list[Path],
    config: Config,
    *,
    services: RunServices,
    n_replicates: int = 3,
    existing_result: IterationResult | None = None,
    existing_scores: list[MetricScores] | None = None,
) -> ReplicatedEvaluation:
    """Run replicated caption+generate+evaluate cycles for a single template.

    Used by the A1 paired-replicate promotion gate (classic refinement pass) and by the
    short protocol's iter-3 synthesis confirmation. If *existing_result* is provided, its
    full per-image result is used as replicate 0. Otherwise, *existing_scores* can seed
    replicate 0 with score data only. In both cases, only ``n_replicates - 1`` additional
    replicates are generated.
    """
    meta_prompt = template.render()
    style_canon = next((s.value for s in template.sections if s.name == "style_foundation"), "")
    all_replicate_scores: list[list[MetricScores]] = []
    all_replicate_agg: list[AggregatedMetrics] = []

    start_rep = 0
    if existing_result is not None:
        all_replicate_scores.append(list(existing_result.per_image_scores))
        all_replicate_agg.append(existing_result.aggregated)
        start_rep = 1
    elif existing_scores is not None:
        all_replicate_scores.append(existing_scores)
        all_replicate_agg.append(
            replace(
                aggregate(existing_scores),
                requested_ref_count=config.num_fixed_refs,
                actual_ref_count=len(fixed_refs),
            )
        )
        start_rep = 1

    async def _run_one_replicate(rep: int) -> tuple[list[MetricScores], AggregatedMetrics] | None:
        rep_id = branch_id * 100 + rep  # unique experiment_id per replicate
        captions, generated_paths, pairs = await _caption_and_generate(
            fixed_refs,
            meta_prompt,
            negative_prompt=template.negative_prompt,
            style_canon=style_canon,
            config=config,
            services=services,
            iteration=iteration,
            experiment_id=rep_id,
        )
        if not generated_paths:
            logger.warning("Replicate %d/%d for branch %d: no images generated", rep, n_replicates, branch_id)
            return None

        gen_paths = [gen for _, gen in pairs]
        ref_paths_eval = [orig for orig, _ in pairs]
        caption_by_path = {c.image_path: c.text for c in captions}
        eval_captions = [caption_by_path[orig] for orig, _ in pairs]

        (metric_scores, _n_eval_failed), (_, vision_scores_list) = await asyncio.gather(
            services.evaluation.evaluate_images(gen_paths, ref_paths_eval, eval_captions),
            services.evaluation.compare_vision_per_image(pairs, eval_captions),
        )

        scores = [_merge_vision(ms, vs) for ms, vs in zip(metric_scores, vision_scores_list, strict=True)]
        aggregated = aggregate(scores)
        compliance_stats, _ = compute_caption_compliance(
            [s.name for s in template.sections],
            captions,
            caption_sections=template.caption_sections,
            meta_prompt=meta_prompt,
        )
        return scores, replace(
            aggregated,
            style_consistency=compute_style_consistency(captions),
            compliance_topic_coverage=compliance_stats.section_topic_coverage,
            compliance_marker_coverage=compliance_stats.section_marker_coverage,
            section_ordering_rate=compliance_stats.section_ordering_rate,
            section_balance_rate=compliance_stats.section_balance_rate,
            subject_specificity_rate=compliance_stats.subject_specificity_rate,
            style_canon_fidelity=compliance_stats.style_canon_fidelity,
            observation_boilerplate_purity=compliance_stats.observation_boilerplate_purity,
            requested_ref_count=config.num_fixed_refs,
            actual_ref_count=len(fixed_refs),
        )

    # Run all replicates in parallel
    rep_results = await asyncio.gather(
        *[_run_one_replicate(rep) for rep in range(start_rep, n_replicates)],
        return_exceptions=True,
    )
    for r in rep_results:
        if isinstance(r, BaseException):
            logger.warning("Replicate failed for branch %d: %s", branch_id, r)
        elif r is not None:
            scores, agg = r
            all_replicate_scores.append(scores)
            all_replicate_agg.append(agg)

    if not all_replicate_scores:
        msg = f"All replicates failed for branch {branch_id}"
        raise RuntimeError(msg)

    median_scores = _median_metric_scores(all_replicate_scores)
    median_agg = aggregate(median_scores)
    style_con = statistics.median(agg.style_consistency for agg in all_replicate_agg)
    flat_gaps = [gap for agg in all_replicate_agg for gap in agg.style_gap_notes]
    median_agg = replace(
        median_agg,
        style_consistency=style_con,
        compliance_topic_coverage=statistics.median(agg.compliance_topic_coverage for agg in all_replicate_agg),
        compliance_marker_coverage=statistics.median(agg.compliance_marker_coverage for agg in all_replicate_agg),
        section_ordering_rate=statistics.median(agg.section_ordering_rate for agg in all_replicate_agg),
        section_balance_rate=statistics.median(agg.section_balance_rate for agg in all_replicate_agg),
        subject_specificity_rate=statistics.median(agg.subject_specificity_rate for agg in all_replicate_agg),
        style_canon_fidelity=statistics.median(agg.style_canon_fidelity for agg in all_replicate_agg),
        observation_boilerplate_purity=statistics.median(
            agg.observation_boilerplate_purity for agg in all_replicate_agg
        ),
        style_gap_notes=aggregate_style_gap_notes(flat_gaps),
        requested_ref_count=config.num_fixed_refs,
        actual_ref_count=len(fixed_refs),
    )

    return ReplicatedEvaluation(
        template=template,
        branch_id=branch_id,
        replicate_scores=all_replicate_scores,
        replicate_aggregated=all_replicate_agg,
        median_per_image=median_scores,
        median_aggregated=median_agg,
    )
