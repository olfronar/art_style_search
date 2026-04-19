"""CLI argument parsing and configuration."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from art_style_search.runs import DEFAULT_RUNS_DIR, resolve_run_dir

load_dotenv()


@dataclass(frozen=True)
class Config:
    """All configuration, built from CLI args + defaults."""

    # Paths
    reference_dir: Path
    output_dir: Path
    log_dir: Path
    state_file: Path
    run_dir: Path

    # Loop
    max_iterations: int
    plateau_window: int
    num_branches: int

    # Generation
    aspect_ratio: str

    # Sampling
    num_fixed_refs: int

    # Models
    caption_model: str
    generator_model: str
    reasoning_model: str
    reasoning_provider: str  # "anthropic", "zai", "openai", "xai", or "local"
    reasoning_base_url: str  # custom base URL for local/remote OpenAI-compatible servers

    # Concurrency
    gemini_concurrency: int
    eval_concurrency: int

    seed: int
    protocol: str  # "short" (3-iter foundation, default) or "classic" (5-iter refinement)

    # API keys
    anthropic_api_key: str
    google_api_key: str
    zai_api_key: str
    openai_api_key: str
    xai_api_key: str = ""
    comparison_provider: str = "gemini"  # "gemini" or "xai"
    comparison_model: str = ""
    raw_proposals: int = 9
    # A1 paired-replicate gate: N>1 enables replicate-based promotion decisions (classic pass).
    replicates: int = 1

    # Gemini extended-thinking levels ("MINIMAL" | "LOW" | "MEDIUM" | "HIGH")
    caption_thinking_level: str = "MINIMAL"
    generation_thinking_level: str = "MINIMAL"

    # Reasoning-model effort ("low" | "medium" | "high"). Mapped per provider by
    # ReasoningClient — Anthropic: disabled/adaptive/enabled+budget; OpenAI:
    # reasoning.effort; Z.AI/xAI/local: dropped with one-time warning.
    reasoning_effort: str = "medium"

    # Zero-step captioner. "gemini" uses Gemini Pro (same as per-iteration captioning);
    # "claude" routes the one-time bootstrap captions through the Anthropic reasoning client
    # at claude-opus-4-7 with reasoning_effort derived from caption_thinking_level.
    bootstrap_captioner: str = "gemini"
    bootstrap_caption_model: str = "claude-opus-4-7"


def parse_args(argv: list[str] | None = None) -> Config:
    """Parse CLI arguments into a Config."""
    parser = argparse.ArgumentParser(
        prog="art_style_search",
        description="Self-improving loop for art style prompt optimization",
    )

    # Paths
    paths = parser.add_argument_group("Paths")
    paths.add_argument("--reference-dir", type=Path, default=Path("reference_images"), help="Reference art directory")
    paths.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Base directory for all runs")
    paths.add_argument("--run", type=str, default=None, dest="run_name", help="Run name (auto-incremented if omitted)")
    paths.add_argument("--new", action="store_true", help="Force new run (error if name already exists)")

    # Loop control
    loop = parser.add_argument_group("Loop Control")
    loop.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help=(
            "Maximum optimization iterations (default 5 for --protocol classic; "
            "ignored for --protocol short which hard-codes 3 iterations)"
        ),
    )
    loop.add_argument(
        "--plateau-window", type=int, default=3, help="Iterations without improvement before branch stops"
    )
    loop.add_argument("--num-branches", type=int, default=9, help="Number of parallel population branches")
    loop.add_argument(
        "--raw-proposals",
        type=int,
        default=9,
        help="Number of raw proposals per iteration before portfolio selection (8-12)",
    )

    # Generation
    gen = parser.add_argument_group("Generation")
    gen.add_argument("--aspect-ratio", default="1:1", help="Aspect ratio for generated images")

    # Sampling
    samp = parser.add_argument_group("Sampling")
    samp.add_argument("--num-fixed-refs", type=int, default=20, help="Fixed reference images for optimization")

    # Models
    models = parser.add_argument_group("Models")
    models.add_argument("--caption-model", default="gemini-3.1-pro-preview", help="Gemini model for captioning")
    models.add_argument(
        "--generator-model",
        default="gemini-3.1-flash-image-preview",
        help="Gemini model for image generation",
    )
    models.add_argument(
        "--reasoning-provider",
        choices=["anthropic", "zai", "openai", "xai", "local"],
        default="anthropic",
        help=(
            "Reasoning model provider: anthropic (Claude), zai (GLM-5.1), openai (GPT-5.4), "
            "xai (Grok 4.20), or local (OpenAI-compatible)"
        ),
    )
    models.add_argument(
        "--reasoning-model",
        default=None,
        help="Reasoning model name (default: claude-opus-4-7 / glm-5.1 / gpt-5.4 / grok-4.20-reasoning-latest)",
    )
    models.add_argument(
        "--comparison-provider",
        choices=["gemini", "xai"],
        default="gemini",
        help="Image-comparison provider: gemini (default) or xai (Grok 4.20 multimodal)",
    )
    models.add_argument(
        "--comparison-model",
        default=None,
        help="Image-comparison model name (default: caption model for gemini / grok-4.20-reasoning-latest for xai)",
    )
    models.add_argument(
        "--reasoning-base-url",
        default="",
        help="Base URL for local/remote OpenAI-compatible server (e.g. http://gpu:8000/v1)",
    )
    models.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
        help=(
            "Reasoning-model effort (default medium). Anthropic: low=thinking disabled, "
            "medium=adaptive, high=enabled with a 16k budget. OpenAI: maps to reasoning.effort. "
            "Z.AI/xAI/local: dropped with a one-time warning."
        ),
    )

    # Protocol
    protocol_group = parser.add_argument_group("Protocol")
    protocol_group.add_argument(
        "--protocol",
        choices=["short", "classic"],
        default="short",
        help=(
            "Protocol: short (default — 3-iter foundation pass, cheap) or classic "
            "(5-iter refinement pass with replicate gate + diff-based canon editing; resumes on a prior short run)"
        ),
    )
    protocol_group.add_argument(
        "--seed", type=int, default=None, help="RNG seed for reproducibility (random if omitted)"
    )
    protocol_group.add_argument(
        "--replicates",
        type=int,
        default=1,
        help=(
            "A1 paired-replicate gate: replicates per candidate (default 1 — single-shot). "
            "Values ≥ 2 enable the replicate-based promotion gate: candidate's min replicate "
            "score must exceed baseline's max, AND candidate's median must exceed baseline's "
            "median + epsilon. Recommended: 3 for the classic refinement pass."
        ),
    )

    # Concurrency
    conc = parser.add_argument_group("Concurrency")
    conc.add_argument("--gemini-concurrency", type=int, default=50, help="Max concurrent Gemini API calls")
    conc.add_argument("--eval-concurrency", type=int, default=4, help="Max concurrent eval threads")

    # Zero-step bootstrap captioner (one-time captioning of the fixed reference set)
    bootstrap = parser.add_argument_group("Bootstrap Captioner")
    bootstrap.add_argument(
        "--bootstrap-captioner",
        choices=["gemini", "claude"],
        default="gemini",
        help=(
            "Provider for the one-time zero-step captioning (default gemini reuses the "
            "per-iteration captioner). 'claude' routes via the Anthropic reasoning client at "
            "--bootstrap-caption-model; requires ANTHROPIC_API_KEY."
        ),
    )
    bootstrap.add_argument(
        "--bootstrap-caption-model",
        default="claude-opus-4-7",
        help="Anthropic model used when --bootstrap-captioner claude (default claude-opus-4-7).",
    )

    # Gemini extended-thinking (trades latency + tokens for quality)
    thinking = parser.add_argument_group("Gemini Thinking")
    thinking.add_argument(
        "--caption-thinking-level",
        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        default="MINIMAL",
        help=(
            "Gemini Pro captioner extended-thinking level (default MINIMAL — current behavior). "
            "MEDIUM materially improves medium identification + proportion precision at 2-3x latency."
        ),
    )
    thinking.add_argument(
        "--generation-thinking-level",
        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        default="MINIMAL",
        help="Gemini Flash image-generation extended-thinking level (default MINIMAL).",
    )

    # API keys
    keys = parser.add_argument_group("API Keys")
    keys.add_argument("--anthropic-api-key", default=None, help="Anthropic API key (env: ANTHROPIC_API_KEY)")
    keys.add_argument("--google-api-key", default=None, help="Google API key (env: GOOGLE_API_KEY)")
    keys.add_argument("--zai-api-key", default=None, help="Z.AI API key (env: ZAI_API_KEY)")
    keys.add_argument("--openai-api-key", default=None, help="OpenAI API key (env: OPENAI_API_KEY)")
    keys.add_argument("--xai-api-key", default=None, help="xAI API key (env: XAI_API_KEY)")

    args = parser.parse_args(argv)
    return _validate_and_build_config(args, parser)


def _validate_and_build_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Config:
    """Validate parsed args, resolve API keys, create directories, and build Config."""
    anthropic_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY", "")
    zai_key = args.zai_api_key or os.environ.get("ZAI_API_KEY", "")
    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    xai_key = args.xai_api_key or os.environ.get("XAI_API_KEY", "")

    provider = args.reasoning_provider
    if provider == "anthropic" and not anthropic_key:
        parser.error("ANTHROPIC_API_KEY must be set via --anthropic-api-key or environment variable")
    if args.bootstrap_captioner == "claude" and not anthropic_key:
        parser.error("ANTHROPIC_API_KEY must be set when --bootstrap-captioner claude (via --anthropic-api-key or env)")
    if provider == "zai" and not zai_key:
        parser.error("ZAI_API_KEY must be set via --zai-api-key or environment variable")
    if provider == "openai" and not openai_key:
        parser.error("OPENAI_API_KEY must be set via --openai-api-key or environment variable")
    if provider == "xai" and not xai_key:
        parser.error("XAI_API_KEY must be set via --xai-api-key or environment variable")
    if provider == "local" and not args.reasoning_base_url:
        parser.error("--reasoning-base-url is required when using --reasoning-provider local")
    if provider == "local" and not args.reasoning_model:
        parser.error("--reasoning-model is required when using --reasoning-provider local")
    if not google_key:
        parser.error("GOOGLE_API_KEY must be set via --google-api-key or environment variable")

    comparison_provider = args.comparison_provider
    if comparison_provider == "xai" and not xai_key:
        parser.error("XAI_API_KEY must be set via --xai-api-key or environment variable")
    if not 8 <= args.raw_proposals <= 12:
        parser.error("--raw-proposals must be between 8 and 12")
    if args.replicates < 1:
        parser.error("--replicates must be >= 1")

    # Default model based on provider
    default_models = {
        "anthropic": "claude-opus-4-7",
        "zai": "glm-5.1",
        "openai": "gpt-5.4",
        "xai": "grok-4.20-reasoning-latest",
        "local": "",
    }
    reasoning_model = args.reasoning_model or default_models[provider]
    default_comparison_models = {
        "gemini": args.caption_model,
        "xai": "grok-4.20-reasoning-latest",
    }
    comparison_model = args.comparison_model or default_comparison_models[comparison_provider]

    # Resolve seed: generate one if not provided
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    if not args.reference_dir.is_dir():
        parser.error(f"Reference directory does not exist: {args.reference_dir}")

    run_dir = resolve_run_dir(args.runs_dir, args.run_name, args.new)
    output_dir = run_dir / "outputs"
    log_dir = run_dir / "logs"
    state_file = run_dir / "state.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Phase-1 foundation pass is hard-capped at 3 iterations (plan §6). Clamp regardless of CLI
    # so ``--protocol short --max-iterations 50`` doesn't silently balloon the budget.
    max_iterations = 3 if args.protocol == "short" else args.max_iterations

    return Config(
        reference_dir=args.reference_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        state_file=state_file,
        run_dir=run_dir,
        max_iterations=max_iterations,
        plateau_window=args.plateau_window,
        num_branches=args.num_branches,
        aspect_ratio=args.aspect_ratio,
        num_fixed_refs=args.num_fixed_refs,
        caption_model=args.caption_model,
        generator_model=args.generator_model,
        reasoning_model=reasoning_model,
        reasoning_provider=provider,
        reasoning_base_url=args.reasoning_base_url or "",
        gemini_concurrency=args.gemini_concurrency,
        eval_concurrency=args.eval_concurrency,
        seed=seed,
        protocol=args.protocol,
        replicates=args.replicates,
        anthropic_api_key=anthropic_key,
        google_api_key=google_key,
        zai_api_key=zai_key,
        openai_api_key=openai_key,
        xai_api_key=xai_key,
        comparison_provider=comparison_provider,
        comparison_model=comparison_model,
        raw_proposals=args.raw_proposals,
        caption_thinking_level=args.caption_thinking_level,
        generation_thinking_level=args.generation_thinking_level,
        reasoning_effort=args.reasoning_effort,
        bootstrap_captioner=args.bootstrap_captioner,
        bootstrap_caption_model=args.bootstrap_caption_model,
    )
