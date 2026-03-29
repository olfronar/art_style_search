"""CLI argument parsing and configuration."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """All configuration, built from CLI args + defaults."""

    # Paths
    reference_dir: Path
    output_dir: Path
    log_dir: Path
    state_file: Path

    # Loop
    max_iterations: int
    plateau_window: int
    num_branches: int

    # Generation
    num_images: int
    aspect_ratio: str

    # Sampling
    max_analysis_images: int
    max_eval_images: int

    # Models
    caption_model: str
    generator_model: str
    reasoning_model: str
    reasoning_provider: str  # "anthropic" or "zai"

    # Concurrency
    gemini_concurrency: int
    eval_concurrency: int

    # API keys
    anthropic_api_key: str
    google_api_key: str
    zai_api_key: str


def parse_args(argv: list[str] | None = None) -> Config:
    """Parse CLI arguments into a Config."""
    parser = argparse.ArgumentParser(
        prog="art_style_search",
        description="Self-improving loop for art style prompt optimization",
    )

    # Paths
    paths = parser.add_argument_group("Paths")
    paths.add_argument("--reference-dir", type=Path, default=Path("reference_images"), help="Reference art directory")
    paths.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Generated images output directory")
    paths.add_argument("--log-dir", type=Path, default=Path("logs"), help="Iteration logs directory")
    paths.add_argument("--state-file", type=Path, default=Path("state.json"), help="State file for resume")

    # Loop control
    loop = parser.add_argument_group("Loop Control")
    loop.add_argument("--max-iterations", type=int, default=20, help="Maximum optimization iterations")
    loop.add_argument(
        "--plateau-window", type=int, default=5, help="Iterations without improvement before branch stops"
    )
    loop.add_argument("--num-branches", type=int, default=5, help="Number of parallel population branches")

    # Generation
    gen = parser.add_argument_group("Generation")
    gen.add_argument("--num-images", type=int, default=4, help="Images per iteration per branch")
    gen.add_argument("--aspect-ratio", default="1:1", help="Aspect ratio for generated images")

    # Sampling
    samp = parser.add_argument_group("Sampling")
    samp.add_argument("--max-analysis-images", type=int, default=10, help="Max reference images for zero-step analysis")
    samp.add_argument("--max-eval-images", type=int, default=10, help="Max reference images per evaluation iteration")

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
        choices=["anthropic", "zai"],
        default="anthropic",
        help="Reasoning model provider: anthropic (Claude) or zai (GLM-5.1)",
    )
    models.add_argument(
        "--reasoning-model",
        default=None,
        help="Reasoning model name (default: claude-opus-4-6 for anthropic, glm-5 for zai)",
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

    args = parser.parse_args(argv)

    anthropic_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY", "")
    zai_key = args.zai_api_key or os.environ.get("ZAI_API_KEY", "")

    provider = args.reasoning_provider
    if provider == "anthropic" and not anthropic_key:
        parser.error("ANTHROPIC_API_KEY must be set via --anthropic-api-key or environment variable")
    if provider == "zai" and not zai_key:
        parser.error("ZAI_API_KEY must be set via --zai-api-key or environment variable")
    if not google_key:
        parser.error("GOOGLE_API_KEY must be set via --google-api-key or environment variable")

    # Default model based on provider
    default_models = {"anthropic": "claude-opus-4-6", "zai": "glm-5"}
    reasoning_model = args.reasoning_model or default_models[provider]

    if not args.reference_dir.is_dir():
        parser.error(f"Reference directory does not exist: {args.reference_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        state_file=args.state_file,
        max_iterations=args.max_iterations,
        plateau_window=args.plateau_window,
        num_branches=args.num_branches,
        num_images=args.num_images,
        aspect_ratio=args.aspect_ratio,
        max_analysis_images=args.max_analysis_images,
        max_eval_images=args.max_eval_images,
        caption_model=args.caption_model,
        generator_model=args.generator_model,
        reasoning_model=reasoning_model,
        reasoning_provider=provider,
        gemini_concurrency=args.gemini_concurrency,
        eval_concurrency=args.eval_concurrency,
        anthropic_api_key=anthropic_key,
        google_api_key=google_key,
        zai_api_key=zai_key,
    )
