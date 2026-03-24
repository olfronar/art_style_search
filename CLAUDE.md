# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Self-improving loop that optimizes a meta-prompt for precise image recreation. The meta-prompt instructs a captioner (Gemini Pro) how to describe images so a generator (Gemini Flash) can recreate them from the captions. Inspired by karpathy/autoresearch.

## Architecture

- **Claude (Anthropic)**: Brain/optimizer. Analyzes reproduction quality, refines the meta-prompt.
- **Gemini 3.1 Pro Preview**: Captioner. Describes reference images using the meta-prompt instructions.
- **Gemini 3.1 Flash Image Preview**: Generator. Produces images from per-image captions.

## Loop Flow

0. **Zero-step**: Fix 10 reference images. Caption them, analyze style → `StyleProfile` + initial meta-prompt (`PromptTemplate`).
1. Meta-prompt + each reference image → Gemini Pro generates per-image captions
2. Each caption → Gemini Flash generates an image
3. Compare each (original, generated) pair: metrics (DINO, LPIPS, HPS, aesthetics) + Gemini vision comparison
4. Claude analyzes per-image gaps + metrics, refines the meta-prompt
5. Cross-pollinate: share global best meta-prompt across population branches
6. Repeat until convergence (max iterations / plateau / Claude stop)

## Commands

```bash
uv sync                                  # Install all dependencies
uv run ruff check .                      # Lint
uv run ruff format .                     # Format
uv run pytest tests/                     # Run tests
uv run python -m art_style_search        # Run the optimization loop
uv run python -m art_style_search --help # Show all CLI options
```

## Environment Variables

- `ANTHROPIC_API_KEY` - Anthropic API key for Claude
- `GOOGLE_API_KEY` - Google API key for Gemini models

## Module Map

- `types.py` - Shared dataclasses (Caption, MetricScores, StyleProfile, PromptTemplate, BranchState, LoopState, etc.)
- `config.py` - CLI argument parsing → Config dataclass
- `analyze.py` - Zero-step: parallel Gemini+Claude style analysis → StyleProfile + initial PromptTemplate
- `caption.py` - Gemini Pro captioning with disk cache
- `prompt.py` - Claude meta-prompt proposal/refinement (template structure + values)
- `generate.py` - Gemini Flash image generation with semaphore + retry
- `models.py` - ModelRegistry: lazy-load DINO/LPIPS/HPS/Aesthetics with per-model locks
- `evaluate.py` - Dispatches 4 metrics per image via asyncio.to_thread + Gemini vision comparison
- `utils.py` - Shared helpers: Anthropic streaming/text extraction, Gemini image part builder, MIME map
- `state.py` - JSON persistence (state.json + per-iteration logs)
- `loop.py` - BSP orchestration loop (zero-step → population branches → convergence)
- `__main__.py` - Entry point

## Directory Conventions

- `src/art_style_search/` - All source code
- `reference_images/` - User-provided reference art (not committed)
- `outputs/` - Generated images by iteration/branch (not committed)
- `logs/` - Iteration logs, captions cache, style profile (not committed)
- `state.json` - Resume state (not committed)

## Evaluation Metrics

All metrics compare generated images against reference images:
- **DINO cosine similarity**: Semantic/structural similarity via DINOv2 embeddings. Higher = better.
- **LPIPS**: Perceptual distance. Lower = better.
- **HPS v2**: Human preference score for text-to-image quality. Higher = better.
- **LAION Aesthetics**: Aesthetic quality predictor (1-10 scale). Higher = better.

## Code Conventions

- Helpers used by 2+ modules belong in `utils.py` — do not duplicate locally (e.g. MIME maps, API call wrappers, response extractors)
- Data fed to Claude in `refine_template` must appear via exactly one path — if the history formatter includes a field, don't also add a dedicated section for it (or vice versa)

## Code Style

- Ruff handles linting and formatting (config in pyproject.toml)
- Line length: 120
- Format-on-edit hook is active — do not manually run ruff after edits
