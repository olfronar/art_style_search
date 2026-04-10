"""CLI argument parsing and configuration."""

from __future__ import annotations

import argparse
import os
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
    reasoning_provider: str  # "anthropic", "zai", "openai", or "local"
    reasoning_base_url: str  # custom base URL for local/remote OpenAI-compatible servers

    # Concurrency
    gemini_concurrency: int
    eval_concurrency: int

    # API keys
    anthropic_api_key: str
    google_api_key: str
    zai_api_key: str
    openai_api_key: str


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
    loop.add_argument("--max-iterations", type=int, default=20, help="Maximum optimization iterations")
    loop.add_argument(
        "--plateau-window", type=int, default=5, help="Iterations without improvement before branch stops"
    )
    loop.add_argument("--num-branches", type=int, default=5, help="Number of parallel population branches")

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
        choices=["anthropic", "zai", "openai", "local"],
        default="anthropic",
        help="Reasoning model provider: anthropic (Claude), zai (GLM-5.1), openai (GPT-5.4), or local (OpenAI-compatible)",
    )
    models.add_argument(
        "--reasoning-model",
        default=None,
        help="Reasoning model name (default: claude-sonnet-4-6 / glm-5.1 / gpt-5.4)",
    )
    models.add_argument(
        "--reasoning-base-url",
        default="",
        help="Base URL for local/remote OpenAI-compatible server (e.g. http://gpu:8000/v1)",
    )

    # Concurrency
    conc = parser.add_argument_group("Concurrency")
    conc.add_argument("--gemini-concurrency", type=int, default=50, help="Max concurrent Gemini API calls")
    conc.add_argument("--eval-concurrency", type=int, default=4, help="Max concurrent eval threads")

    # API keys
    keys = parser.add_argument_group("API Keys")
    keys.add_argument("--anthropic-api-key", default=None, help="Anthropic API key (env: ANTHROPIC_API_KEY)")
    keys.add_argument("--google-api-key", default=None, help="Google API key (env: GOOGLE_API_KEY)")
    keys.add_argument("--zai-api-key", default=None, help="Z.AI API key (env: ZAI_API_KEY)")
    keys.add_argument("--openai-api-key", default=None, help="OpenAI API key (env: OPENAI_API_KEY)")

    args = parser.parse_args(argv)

    anthropic_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY", "")
    zai_key = args.zai_api_key or os.environ.get("ZAI_API_KEY", "")
    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    provider = args.reasoning_provider
    if provider == "anthropic" and not anthropic_key:
        parser.error("ANTHROPIC_API_KEY must be set via --anthropic-api-key or environment variable")
    if provider == "zai" and not zai_key:
        parser.error("ZAI_API_KEY must be set via --zai-api-key or environment variable")
    if provider == "openai" and not openai_key:
        parser.error("OPENAI_API_KEY must be set via --openai-api-key or environment variable")
    if provider == "local" and not args.reasoning_base_url:
        parser.error("--reasoning-base-url is required when using --reasoning-provider local")
    if provider == "local" and not args.reasoning_model:
        parser.error("--reasoning-model is required when using --reasoning-provider local")
    if not google_key:
        parser.error("GOOGLE_API_KEY must be set via --google-api-key or environment variable")

    # Default model based on provider
    default_models = {"anthropic": "claude-sonnet-4-6", "zai": "glm-5.1", "openai": "gpt-5.4", "local": ""}
    reasoning_model = args.reasoning_model or default_models[provider]

    if not args.reference_dir.is_dir():
        parser.error(f"Reference directory does not exist: {args.reference_dir}")

    run_dir = resolve_run_dir(args.runs_dir, args.run_name, args.new)
    output_dir = run_dir / "outputs"
    log_dir = run_dir / "logs"
    state_file = run_dir / "state.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        reference_dir=args.reference_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        state_file=state_file,
        run_dir=run_dir,
        max_iterations=args.max_iterations,
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
        anthropic_api_key=anthropic_key,
        google_api_key=google_key,
        zai_api_key=zai_key,
        openai_api_key=openai_key,
    )
