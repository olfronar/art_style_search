# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Self-improving loop that finds the optimal prompt to define and reproduce an art style from reference images. Inspired by karpathy/autoresearch.

## Architecture

- **Claude (Anthropic)**: Brain/analyzer. Evaluates metrics, reasons about style, proposes prompt improvements.
- **Gemini 3.1 Pro Preview**: Captioner. Describes reference images in detail (superior vision).
- **Gemini 3.1 Flash Image Preview**: Generator. Produces images from style prompts.

## Loop Flow

1. Caption reference images (once, cached) via Gemini 3.1 Pro Preview
2. Claude proposes/refines a style prompt based on captions + prior metrics
3. Gemini 3.1 Flash Image Preview generates images from the prompt
4. Evaluate generated vs reference images (DINO, LPIPS, HPS, aesthetics)
5. Claude analyzes results, decides to keep/discard, proposes next iteration
6. Repeat until convergence or iteration limit

## Commands

```bash
uv sync                                  # Install all dependencies
uv run ruff check .                      # Lint
uv run ruff format .                     # Format
uv run python -m art_style_search.loop   # Run the optimization loop
```

## Environment Variables

- `ANTHROPIC_API_KEY` - Anthropic API key for Claude
- `GOOGLE_API_KEY` - Google API key for Gemini models

## Directory Conventions

- `src/art_style_search/` - All source code
- `reference_images/` - User-provided reference art (not committed)
- `outputs/` - Generated images by iteration (not committed)
- `logs/` - Iteration logs with metrics history (not committed)

## Evaluation Metrics

All metrics compare generated images against reference images:
- **DINO cosine similarity**: Semantic/structural similarity via DINOv2 embeddings. Higher = better.
- **LPIPS**: Perceptual distance. Lower = better.
- **HPS v2**: Human preference score for text-to-image quality. Higher = better.
- **LAION Aesthetics**: Aesthetic quality predictor (1-10 scale). Higher = better.

## Code Style

- Ruff handles linting and formatting (config in pyproject.toml)
- Line length: 120
- Format-on-edit hook is active — do not manually run ruff after edits
