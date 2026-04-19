"""Run-level workflow setup, teardown, and shared helpers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import platform as _platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from google import genai  # type: ignore[attr-defined]
from openai import AsyncOpenAI

from art_style_search.config import Config
from art_style_search.models import ModelRegistry
from art_style_search.scoring import composite_score
from art_style_search.state import load_manifest, save_manifest, save_state
from art_style_search.state_codec import _Encoder, to_dict
from art_style_search.types import IterationResult, LoopState, RunManifest
from art_style_search.utils import IMAGE_EXTENSIONS, ReasoningClient
from art_style_search.workflow.services import (
    CaptioningService,
    EvaluationService,
    GenerationService,
    ReasoningService,
    RunServices,
)

logger = logging.getLogger(__name__)


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


def _save_best_prompt(state: LoopState, log_dir: Path) -> None:
    """Write the best meta-prompt to a standalone file for easy access."""
    if not state.global_best_prompt:
        return
    prompt_file = log_dir / "best_prompt.txt"
    prompt_file.write_text(state.global_best_prompt, encoding="utf-8")
    logger.info("Best meta-prompt saved to %s", prompt_file)


def _save_best_prompt_md(state: LoopState, log_dir: Path, manifest: RunManifest | None) -> None:
    """Write the best meta-prompt as structured markdown with a YAML front-matter header.

    The body is ``state.global_best_prompt`` — already markdown-formatted by
    ``PromptTemplate.render()``. The front-matter captures run provenance
    (iteration, composite score, seed, git SHA, protocol, timestamp) so the
    file is readable standalone and diffable across iterations.
    """
    if not state.global_best_prompt:
        return
    front_matter: list[str] = ["---"]
    front_matter.append(f"iteration: {state.iteration}")
    if state.global_best_metrics is not None:
        front_matter.append(f"composite_score: {composite_score(state.global_best_metrics):.4f}")
    if manifest is not None:
        front_matter.append(f"seed: {manifest.seed}")
        front_matter.append(f"protocol: {manifest.protocol_version}")
        if manifest.git_sha:
            front_matter.append(f"git_sha: {manifest.git_sha}")
        front_matter.append(f"timestamp_utc: {manifest.timestamp_utc}")
    front_matter.append("---")
    prompt_file = log_dir / "best_prompt.md"
    prompt_file.write_text("\n".join(front_matter) + "\n\n" + state.global_best_prompt, encoding="utf-8")
    logger.info("Best meta-prompt (markdown) saved to %s", prompt_file)


def _save_best_prompt_json(state: LoopState, log_dir: Path) -> None:
    """Write the structured best template as JSON so it can be re-ingested as a seed."""
    if not state.best_template.sections:
        return
    payload = {
        "template": to_dict(state.best_template),
        "iteration": state.iteration,
        "composite_score": (
            composite_score(state.global_best_metrics) if state.global_best_metrics is not None else None
        ),
    }
    prompt_file = log_dir / "best_prompt.json"
    prompt_file.write_text(json.dumps(payload, cls=_Encoder, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Best template (json) saved to %s", prompt_file)


def _log_experiment_results(results: list[IterationResult], log_dir: Path, save_iteration_log) -> None:
    """Save and log each experiment result."""
    for result in results:
        save_iteration_log(result, log_dir)
        metrics = result.aggregated
        logger.info(
            "Exp %d — DS=%.3f Color=%.3f SSIM=%.3f HPS=%.3f Aes=%.1f V[S=%.2f Su=%.2f Co=%.2f] %s",
            result.branch_id,
            metrics.dreamsim_similarity_mean,
            metrics.color_histogram_mean,
            metrics.ssim_mean,
            metrics.hps_score_mean,
            metrics.aesthetics_score_mean,
            metrics.vision_style,
            metrics.vision_subject,
            metrics.vision_composition,
            "KEPT" if result.kept else "discarded",
        )


def _ref_cache_key(paths: list[Path]) -> str:
    """Deterministic hash from sorted reference paths + mtimes for cross-run caching."""
    parts = sorted(f"{p}:{p.stat().st_mtime}" for p in paths)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _hash_reference_images(ref_dir: Path) -> dict[str, str]:
    """SHA-256 hash every image in *ref_dir*, keyed by filename."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _discover_images(ref_dir)}


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

    return RunManifest(
        protocol_version=config.protocol,
        seed=config.seed,
        cli_args={
            "max_iterations": config.max_iterations,
            "plateau_window": config.plateau_window,
            "num_branches": config.num_branches,
            "raw_proposals": config.raw_proposals,
            "aspect_ratio": config.aspect_ratio,
            "num_fixed_refs": config.num_fixed_refs,
            "protocol": config.protocol,
            "caption_thinking_level": config.caption_thinking_level,
            "generation_thinking_level": config.generation_thinking_level,
        },
        model_names={
            "caption_model": config.caption_model,
            "generator_model": config.generator_model,
            "reasoning_model": config.reasoning_model,
            "comparison_model": config.comparison_model,
        },
        reasoning_provider=config.reasoning_provider,
        comparison_provider=config.comparison_provider,
        git_sha=git_sha,
        python_version=sys.version,
        platform=_platform.platform(),
        timestamp_utc=datetime.now(UTC).isoformat(),
        reference_image_hashes=_hash_reference_images(config.reference_dir),
        num_fixed_refs=config.num_fixed_refs,
        discovered_reference_count=len(_discover_images(config.reference_dir)),
        uv_lock_hash=uv_lock_hash,
    )


def _verify_manifest(config: Config, manifest: RunManifest) -> None:
    """Verify on resume that the manifest matches current config — warn on drift."""
    if manifest.seed != config.seed:
        logger.warning("Seed drift: manifest=%d, CLI=%d", manifest.seed, config.seed)
    if manifest.protocol_version != config.protocol:
        logger.info(
            "Protocol change on resume: manifest=%s, CLI=%s (allowed — short→classic is the refinement flow)",
            manifest.protocol_version,
            config.protocol,
        )


def ensure_manifest(config: Config) -> None:
    """Write or verify the run manifest for *config*."""
    manifest_path = config.run_dir / "run_manifest.json"
    existing_manifest = load_manifest(manifest_path)
    if existing_manifest is None:
        save_manifest(_build_manifest(config), manifest_path)
        return
    _verify_manifest(config, existing_manifest)


@dataclass
class RunContext:
    """Immutable per-run dependencies passed to iteration helpers."""

    config: Config
    gemini_client: genai.Client
    reasoning_client: ReasoningClient
    registry: ModelRegistry
    gemini_semaphore: asyncio.Semaphore
    eval_semaphore: asyncio.Semaphore
    services: RunServices
    rng: random.Random = field(default_factory=random.Random)


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
        xai_api_key=config.xai_api_key,
        base_url=config.reasoning_base_url,
        debug_dir=config.log_dir / "reasoning_debug",
        default_reasoning_effort=config.reasoning_effort,
    )
    xai_client = None
    if config.comparison_provider == "xai":
        xai_client = AsyncOpenAI(
            api_key=config.xai_api_key,
            base_url="https://api.x.ai/v1",
            timeout=httpx.Timeout(3600.0, connect=30.0),
        )

    gemini_semaphore = asyncio.Semaphore(config.gemini_concurrency)
    eval_semaphore = asyncio.Semaphore(config.eval_concurrency)

    rng = random.Random(config.seed)
    logger.info("Run seed: %d, protocol: %s", config.seed, config.protocol)

    logger.info("Loading evaluation models...")
    registry = await asyncio.to_thread(ModelRegistry.load_all)

    services = RunServices(
        captioning=CaptioningService(
            client=gemini_client,
            model=config.caption_model,
            semaphore=gemini_semaphore,
            thinking_level=config.caption_thinking_level,
        ),
        generation=GenerationService(
            client=gemini_client,
            model=config.generator_model,
            semaphore=gemini_semaphore,
            aspect_ratio=config.aspect_ratio,
            thinking_level=config.generation_thinking_level,
        ),
        evaluation=EvaluationService(
            gemini_client=gemini_client,
            registry=registry,
            comparison_provider=config.comparison_provider,
            comparison_model=config.comparison_model,
            gemini_semaphore=gemini_semaphore,
            eval_semaphore=eval_semaphore,
            xai_client=xai_client,
        ),
        reasoning=ReasoningService(
            client=reasoning_client,
            model=config.reasoning_model,
            effort=config.reasoning_effort,
        ),
    )
    return RunContext(
        config=config,
        gemini_client=gemini_client,
        reasoning_client=reasoning_client,
        registry=registry,
        gemini_semaphore=gemini_semaphore,
        eval_semaphore=eval_semaphore,
        services=services,
        rng=rng,
    )


def _finalize_run(state: LoopState, ctx: RunContext) -> LoopState:
    """Persist final state, write best prompt, and log the summary banner."""
    save_state(state, ctx.config.state_file)
    _save_best_prompt(state, ctx.config.log_dir)
    manifest = load_manifest(ctx.config.run_dir / "run_manifest.json")
    _save_best_prompt_md(state, ctx.config.log_dir, manifest)
    _save_best_prompt_json(state, ctx.config.log_dir)

    logger.info("=" * 60)
    if state.global_best_metrics:
        metrics = state.global_best_metrics
        logger.info(
            "FINAL BEST — DS=%.4f HPS=%.4f Aes=%.2f",
            metrics.dreamsim_similarity_mean,
            metrics.hps_score_mean,
            metrics.aesthetics_score_mean,
        )
    logger.info("BEST META-PROMPT: %s", state.global_best_prompt)
    logger.info("Convergence: %s", state.convergence_reason)
    logger.info("Total experiments: %d", len(state.experiment_history))
    logger.info("KB: %d hypotheses", len(state.knowledge_base.hypotheses))
    return state
