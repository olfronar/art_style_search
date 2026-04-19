"""`verify-metrics` subcommand — sanity-check image evaluation on a pair of identical images.

Runs the full evaluation stack (local models + vision judge + caption-compliance) against
a randomly-chosen reference image from an existing run, compared against itself. Paired
metrics are expected to hit their maximum; the vision judge is expected to return MATCH
on every ternary dimension. HPS / Aesthetics / style_gap text / caption-compliance
sub-components are printed without assertion (content-dependent).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from google import genai  # type: ignore[attr-defined]

from art_style_search.evaluate import (
    aggregate,
    compare_vision_per_image,
    compute_caption_compliance,
    compute_style_consistency,
    evaluate_images,
)
from art_style_search.models import ModelRegistry
from art_style_search.runs import DEFAULT_RUNS_DIR
from art_style_search.scoring import composite_score, per_image_composite
from art_style_search.state_codec import prompt_template_from_dict
from art_style_search.types import Caption, MetricScores, PromptTemplate, VisionScores

logger = logging.getLogger(__name__)

_DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
_DEFAULT_XAI_MODEL = "grok-4.20-reasoning-latest"

# Floating-point slack for metric-identity invariants. Identical inputs should
# produce exact 1.0 outputs; slack is only to tolerate quantization in histogram
# / SSIM resizing paths.
_PAIRED_TOLERANCES: dict[str, float] = {
    "dreamsim_similarity": 0.9999,
    "color_histogram": 0.999,
    "ssim": 0.999,
}


# ---------------------------------------------------------------------------
# Public argument parser (exposed for __main__ dispatcher and tests)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="art_style_search verify-metrics",
        description="Run the full image-evaluation stack on a pair of identical images to sanity-check metrics.",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Base directory for all runs")
    parser.add_argument("--run", type=str, default=None, help="Run name (default: newest run with loadable data)")
    parser.add_argument(
        "--seed", type=int, default=None, help="Deterministic seed for random caption selection (default: random)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=("gemini", "xai"),
        help="Vision-judge provider (default: gemini)",
    )
    parser.add_argument("--model", type=str, default=None, help="Vision-judge model (default: per-provider)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON instead of a formatted table")
    return parser


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def _candidate_runs(runs_dir: Path) -> list[Path]:
    """Return run directories sorted newest-first by mtime."""
    if not runs_dir.is_dir():
        return []
    entries = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "logs").is_dir()]
    return sorted(entries, key=lambda p: p.stat().st_mtime, reverse=True)


def _has_loadable_data(run_dir: Path) -> bool:
    """A run is usable iff it has a meta-prompt source AND at least one iteration log."""
    log_dir = run_dir / "logs"
    has_prompt = (log_dir / "best_prompt.json").is_file() or (log_dir / "best_prompt.txt").is_file()
    has_iter = any(log_dir.glob("iter_*_branch_*.json"))
    return has_prompt and has_iter


def find_newest_run(runs_dir: Path) -> Path:
    """Return the newest run directory with loadable data. Raises FileNotFoundError if none."""
    for candidate in _candidate_runs(runs_dir):
        if _has_loadable_data(candidate):
            return candidate
    msg = f"No runs with loadable best-prompt + iteration logs under {runs_dir}/"
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Meta-prompt loading
# ---------------------------------------------------------------------------


def load_meta_prompt(log_dir: Path) -> tuple[str, str]:
    """Return ``(rendered, source)`` — source is ``"json"`` or ``"txt"``.

    Prefers ``best_prompt.json`` (re-ingestable template) and falls back to the
    flat ``best_prompt.txt`` (already rendered). Raises FileNotFoundError if neither
    exists.
    """
    json_path = log_dir / "best_prompt.json"
    txt_path = log_dir / "best_prompt.txt"
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        template: PromptTemplate = prompt_template_from_dict(payload["template"])
        return template.render(), "json"
    if txt_path.is_file():
        return txt_path.read_text(encoding="utf-8"), "txt"
    msg = f"No best_prompt.json or best_prompt.txt under {log_dir}/"
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Kept-branch loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchLog:
    """Flattened view of the branch-log fields needed by verify-metrics."""

    iteration: int
    branch_id: int
    kept: bool
    composite_score: float
    captions: list[Caption]
    path: Path


def _load_branch_log(path: Path) -> BranchLog | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read branch log %s: %s", path, exc)
        return None
    raw_captions = data.get("iteration_captions") or []
    captions = [Caption(image_path=Path(c["image_path"]), text=c["text"]) for c in raw_captions if c.get("text")]
    if not captions:
        return None
    agg = data.get("aggregated") or {}
    score = float(agg.get("dreamsim_similarity_mean", 0.0))  # placeholder for tie-breaking fallback
    return BranchLog(
        iteration=int(data.get("iteration", -1)),
        branch_id=int(data.get("branch_id", -1)),
        kept=bool(data.get("kept", False)),
        composite_score=score,
        captions=captions,
        path=path,
    )


def find_kept_branch(log_dir: Path) -> BranchLog:
    """Return the newest iteration's kept branch (fallback: highest-score branch).

    Raises FileNotFoundError if no iteration log carries captions.
    """
    logs = sorted(log_dir.glob("iter_*_branch_*.json"))
    branches: list[BranchLog] = []
    for path in logs:
        branch = _load_branch_log(path)
        if branch is not None:
            branches.append(branch)
    if not branches:
        msg = f"No branch logs with captions under {log_dir}/"
        raise FileNotFoundError(msg)
    newest_iter = max(b.iteration for b in branches)
    in_newest = [b for b in branches if b.iteration == newest_iter]
    kept = [b for b in in_newest if b.kept]
    if kept:
        return kept[0]
    # Fallback: highest composite-score branch in the newest iteration
    return max(in_newest, key=lambda b: b.composite_score)


def pick_random_caption(branch: BranchLog, seed: int | None) -> tuple[Path, str]:
    """Return ``(ref_path, caption_text)`` for a randomly chosen paired entry."""
    if not branch.captions:
        msg = "Branch has no captions"
        raise ValueError(msg)
    rng = random.Random(seed)
    idx = rng.randrange(len(branch.captions))
    cap = branch.captions[idx]
    return cap.image_path, cap.text


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricRow:
    name: str
    value: float | str
    expected: str
    status: str  # "OK" | "FAIL" | "INFO"


def _classify(value: float, *, minimum: float | None = None, exact: float | None = None) -> str:
    """Return ``"OK"`` when ``value`` meets the invariant, ``"FAIL"`` otherwise."""
    if exact is not None:
        return "OK" if value == exact else "FAIL"
    if minimum is not None:
        return "OK" if value >= minimum else "FAIL"
    return "INFO"


def _fmt_float(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"


def _render_table(
    *,
    image_path: Path,
    run_dir: Path,
    branch: BranchLog,
    seed: int | None,
    provider: str,
    model: str,
    vision: VisionScores,
    composite: float,
    per_image: float,
    rows: list[MetricRow],
) -> str:
    lines: list[str] = []
    lines.append(f"Image:    {image_path.name}")
    lines.append(f"Run:      {run_dir.name}   iter={branch.iteration}  branch={branch.branch_id}  seed={seed}")
    lines.append(f"Provider: {provider} ({model})")
    lines.append("")

    sections: dict[str, list[MetricRow]] = {"Local paired": [], "Vision judge": [], "Caption-level": []}
    for row in rows:
        if row.name.startswith("vision_"):
            sections["Vision judge"].append(row)
        elif row.name.startswith("caption_") or row.name == "style_consistency":
            sections["Caption-level"].append(row)
        else:
            sections["Local paired"].append(row)

    for heading, section_rows in sections.items():
        if not section_rows:
            continue
        lines.append(f"[{heading}]")
        lines.append(f"  {'Metric':<35} {'Value':>10}   {'Expected':<18}  Status")
        lines.append(f"  {'-' * 35} {'-' * 10}   {'-' * 18}  {'-' * 6}")
        for row in section_rows:
            value_str = _fmt_float(row.value) if isinstance(row.value, float) else str(row.value)
            lines.append(f"  {row.name:<35} {value_str:>10}   {row.expected:<18}  {row.status}")
        lines.append("")

    if vision.style_gap:
        lines.append("[style_gap]")
        lines.append(f"  {vision.style_gap}")
        lines.append("")

    lines.append("[Composite]")
    lines.append(f"  composite_score     {_fmt_float(composite, 4)}")
    lines.append(f"  per_image_composite {_fmt_float(per_image, 4)}")
    return "\n".join(lines)


def _build_rows(
    scores: MetricScores, vision: VisionScores, style_consistency: float, compliance_stats: Any
) -> list[MetricRow]:
    rows: list[MetricRow] = []

    rows.append(
        MetricRow(
            "dreamsim_similarity",
            scores.dreamsim_similarity,
            f">= {_PAIRED_TOLERANCES['dreamsim_similarity']}",
            _classify(scores.dreamsim_similarity, minimum=_PAIRED_TOLERANCES["dreamsim_similarity"]),
        )
    )
    rows.append(
        MetricRow(
            "color_histogram",
            scores.color_histogram,
            f">= {_PAIRED_TOLERANCES['color_histogram']}",
            _classify(scores.color_histogram, minimum=_PAIRED_TOLERANCES["color_histogram"]),
        )
    )
    rows.append(
        MetricRow(
            "ssim",
            scores.ssim,
            f">= {_PAIRED_TOLERANCES['ssim']}",
            _classify(scores.ssim, minimum=_PAIRED_TOLERANCES["ssim"]),
        )
    )
    rows.append(MetricRow("hps_score", scores.hps_score, "(informational)", "INFO"))
    rows.append(MetricRow("aesthetics_score", scores.aesthetics_score, "(informational)", "INFO"))

    for dim, score in (
        ("style", vision.style.score),
        ("subject", vision.subject.score),
        ("composition", vision.composition.score),
        ("medium", vision.medium.score),
        ("proportions", vision.proportions.score),
    ):
        rows.append(MetricRow(f"vision_{dim}", score, "== 1.0 (MATCH)", _classify(score, exact=1.0)))

    rows.append(MetricRow("style_consistency", style_consistency, "== 1.0", _classify(style_consistency, exact=1.0)))
    rows.append(MetricRow("caption_canon_fidelity", compliance_stats.style_canon_fidelity, "(informational)", "INFO"))
    rows.append(
        MetricRow(
            "caption_boilerplate_purity", compliance_stats.observation_boilerplate_purity, "(informational)", "INFO"
        )
    )
    rows.append(MetricRow("caption_topic_coverage", compliance_stats.section_topic_coverage, "(informational)", "INFO"))
    rows.append(
        MetricRow("caption_marker_coverage", compliance_stats.section_marker_coverage, "(informational)", "INFO")
    )
    rows.append(
        MetricRow("caption_section_ordering", compliance_stats.section_ordering_rate, "(informational)", "INFO")
    )
    rows.append(MetricRow("caption_section_balance", compliance_stats.section_balance_rate, "(informational)", "INFO"))
    rows.append(
        MetricRow("caption_subject_specificity", compliance_stats.subject_specificity_rate, "(informational)", "INFO")
    )

    return rows


def _render_json(
    *,
    image_path: Path,
    run_dir: Path,
    branch: BranchLog,
    seed: int | None,
    provider: str,
    model: str,
    rows: list[MetricRow],
    vision: VisionScores,
    composite: float,
    per_image: float,
) -> str:
    payload = {
        "image": str(image_path),
        "run": run_dir.name,
        "iteration": branch.iteration,
        "branch": branch.branch_id,
        "seed": seed,
        "provider": provider,
        "model": model,
        "metrics": [{"name": r.name, "value": r.value, "expected": r.expected, "status": r.status} for r in rows],
        "style_gap": vision.style_gap,
        "composite_score": composite,
        "per_image_composite": per_image,
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Vision-judge client wiring
# ---------------------------------------------------------------------------


def _build_vision_clients(provider: str) -> tuple[genai.Client | None, Any | None]:
    """Construct provider-specific clients for ``compare_vision_per_image``."""
    if provider == "gemini":
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            msg = "GOOGLE_API_KEY is required for --provider gemini"
            raise RuntimeError(msg)
        return genai.Client(api_key=key), None
    if provider == "xai":
        key = os.environ.get("XAI_API_KEY")
        if not key:
            msg = "XAI_API_KEY is required for --provider xai"
            raise RuntimeError(msg)
        from openai import AsyncOpenAI

        return None, AsyncOpenAI(
            api_key=key,
            base_url="https://api.x.ai/v1",
            timeout=httpx.Timeout(3600.0, connect=30.0),
        )
    msg = f"Unknown provider: {provider}"
    raise ValueError(msg)


def _resolve_model(provider: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    return _DEFAULT_GEMINI_MODEL if provider == "gemini" else _DEFAULT_XAI_MODEL


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    runs_dir: Path = args.runs_dir
    if args.run:
        run_dir = runs_dir / args.run
        if not run_dir.is_dir():
            print(f"Error: run {args.run!r} not found at {run_dir}", file=sys.stderr)
            return 2
    else:
        try:
            run_dir = find_newest_run(runs_dir)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    log_dir = run_dir / "logs"
    try:
        meta_prompt, _ = load_meta_prompt(log_dir)
        branch = find_kept_branch(log_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    ref_path, caption_text = pick_random_caption(branch, args.seed)
    if not ref_path.is_file():
        print(f"Error: reference image missing: {ref_path}", file=sys.stderr)
        return 2

    provider = args.provider
    model = _resolve_model(provider, args.model)
    try:
        gemini_client, xai_client = _build_vision_clients(provider)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    logger.info("Loading evaluation models (this may take a minute)...")
    registry = await asyncio.to_thread(ModelRegistry.load_all)

    eval_sem = asyncio.Semaphore(1)
    vision_sem = asyncio.Semaphore(1)

    # Pipeline: ref vs. itself.
    (metric_scores, n_eval_failed), (_, vision_scores_list) = await asyncio.gather(
        evaluate_images([ref_path], [ref_path], [caption_text], registry=registry, semaphore=eval_sem),
        compare_vision_per_image(
            [(ref_path, ref_path)],
            [caption_text],
            provider=provider,
            model=model,
            semaphore=vision_sem,
            client=gemini_client,
            xai_client=xai_client,
        ),
    )

    if n_eval_failed:
        print(f"Error: {n_eval_failed} local-metric evaluation(s) failed", file=sys.stderr)
        return 2

    scores = metric_scores[0]
    vision = vision_scores_list[0]
    # Merge vision into the MetricScores so composite_score sees the ternary dims.
    scores = replace(
        scores,
        vision_style=vision.style.score,
        vision_subject=vision.subject.score,
        vision_composition=vision.composition.score,
        vision_medium=vision.medium.score,
        vision_proportions=vision.proportions.score,
        style_gap=vision.style_gap,
    )

    cap_for_consistency = Caption(image_path=ref_path, text=caption_text)
    style_consistency = compute_style_consistency([cap_for_consistency, cap_for_consistency])
    # Section names derived from the rendered meta-prompt's `## <name>` headers — the txt
    # fallback only carries the rendered form, so header parsing is the common path.
    section_names = _parse_section_names(meta_prompt)
    compliance_stats, _ = compute_caption_compliance(
        section_names,
        [cap_for_consistency],
        caption_sections=None,
        meta_prompt=meta_prompt,
    )

    aggregated = aggregate([scores], completion_rate=1.0)
    aggregated = replace(
        aggregated,
        style_consistency=style_consistency,
        compliance_topic_coverage=compliance_stats.section_topic_coverage,
        compliance_marker_coverage=compliance_stats.section_marker_coverage,
        section_ordering_rate=compliance_stats.section_ordering_rate,
        section_balance_rate=compliance_stats.section_balance_rate,
        subject_specificity_rate=compliance_stats.subject_specificity_rate,
        style_canon_fidelity=compliance_stats.style_canon_fidelity,
        observation_boilerplate_purity=compliance_stats.observation_boilerplate_purity,
        requested_ref_count=1,
        actual_ref_count=1,
    )

    composite = composite_score(aggregated)
    per_image = per_image_composite(scores)
    rows = _build_rows(scores, vision, style_consistency, compliance_stats)

    if args.as_json:
        out = _render_json(
            image_path=ref_path,
            run_dir=run_dir,
            branch=branch,
            seed=args.seed,
            provider=provider,
            model=model,
            rows=rows,
            vision=vision,
            composite=composite,
            per_image=per_image,
        )
    else:
        out = _render_table(
            image_path=ref_path,
            run_dir=run_dir,
            branch=branch,
            seed=args.seed,
            provider=provider,
            model=model,
            vision=vision,
            composite=composite,
            per_image=per_image,
            rows=rows,
        )
    print(out)

    any_fail = any(row.status == "FAIL" for row in rows)
    return 1 if any_fail else 0


def _parse_section_names(rendered_meta_prompt: str) -> list[str]:
    """Extract `## <name>` headers from a rendered meta-prompt, skipping trailing ancillary blocks."""
    skipped = {"Negative Prompt", "Caption Sections (in order)", "Caption Length Target"}
    names: list[str] = []
    for line in rendered_meta_prompt.splitlines():
        if line.startswith("## "):
            name = line[3:].strip()
            if name and name not in skipped:
                names.append(name)
    return names


def run_verify_metrics(args: argparse.Namespace) -> int:
    """Entry point wired to ``python -m art_style_search verify-metrics``."""
    return asyncio.run(_run(args))
