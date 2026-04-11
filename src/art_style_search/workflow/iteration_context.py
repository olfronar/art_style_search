"""Iteration context assembly helpers."""

from __future__ import annotations

from art_style_search.experiment import best_kept_result
from art_style_search.knowledge import build_caption_diffs
from art_style_search.types import LoopState


def _filter_feedback_by_refs(feedback_text: str, feedback_refs: frozenset) -> str:
    """Filter multi-line per-image feedback to include only lines mentioning feedback_ref filenames."""
    if not feedback_text or not feedback_refs:
        return feedback_text
    ref_names = {path.name for path in feedback_refs}
    lines = feedback_text.split("\n")
    kept: list[str] = []
    keep_current = True
    for line in lines:
        if line.startswith("##") or not line.strip():
            kept.append(line)
            keep_current = True
            continue
        if line.startswith("**") or line.startswith("Image ("):
            keep_current = any(name in line for name in ref_names)
        if keep_current:
            kept.append(line)
    return "\n".join(kept)


def _build_iteration_context(state: LoopState) -> tuple[str, str, str]:
    """Phase 0 of an iteration: build (vision_fb, roundtrip_fb, caption_diffs)."""
    best_last = best_kept_result(state.last_iteration_results)
    vision_fb = best_last.vision_feedback if best_last else ""
    roundtrip_fb = best_last.roundtrip_feedback if best_last else ""

    feedback_set = frozenset(state.feedback_refs) if state.silent_refs else None
    if feedback_set and best_last:
        vision_fb = _filter_feedback_by_refs(vision_fb, feedback_set)
        roundtrip_fb = _filter_feedback_by_refs(roundtrip_fb, feedback_set)

    caption_diffs = ""
    if best_last and best_last.iteration_captions:
        captions_for_diff = best_last.iteration_captions
        scores_for_diff = best_last.per_image_scores
        if feedback_set:
            paired = [
                (caption, score)
                for caption, score in zip(captions_for_diff, scores_for_diff, strict=False)
                if caption.image_path in feedback_set
            ]
            captions_for_diff = [caption for caption, _ in paired]
            scores_for_diff = [score for _, score in paired]
        sorted_caps = sorted(
            zip(captions_for_diff, scores_for_diff, strict=False),
            key=lambda item: item[1].dreamsim_similarity,
        )
        worst_caps = [caption for caption, _ in sorted_caps[:3]]
        caption_diffs = build_caption_diffs(state.prev_best_captions, worst_caps)

    if state.review_feedback:
        roundtrip_fb = f"## Independent Review of Last Iteration\n{state.review_feedback}\n\n{roundtrip_fb}"
    if state.pairwise_feedback:
        vision_fb = f"## Pairwise Experiment Comparison\n{state.pairwise_feedback}\n\n{vision_fb}"

    state.review_feedback = ""
    state.pairwise_feedback = ""
    return vision_fb, roundtrip_fb, caption_diffs
