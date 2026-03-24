# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Self-improving loop that optimizes a meta-prompt for precise image recreation. The meta-prompt instructs a captioner (Gemini Pro) how to describe images so a generator (Gemini Flash) can recreate them from the captions. Inspired by karpathy/autoresearch.

## Architecture

- **Claude (Anthropic)**: Brain/optimizer. Analyzes reproduction quality, refines the meta-prompt.
- **Gemini 3.1 Pro Preview**: Captioner. Describes reference images using the meta-prompt instructions.
- **Gemini 3.1 Flash Image Preview**: Generator. Produces images from per-image captions.

## Loop Flow

0. **Zero-step**: Fix 10 reference images. Caption them, analyze style → `StyleProfile` + N diverse initial meta-prompts. Evaluate all, pick best.
1. Claude proposes N experiments (hypothesis-driven template variants) from shared KB, each with a different hypothesis
2. Each experiment in parallel: meta-prompt + reference → caption → generate → evaluate
3. Compare each (original, generated) pair: per-image paired metrics (DINO, LPIPS, HPS, aesthetics) + Gemini vision comparison (style + subject fidelity)
4. Best experiment updates the current template; all results feed into shared KnowledgeBase
5. Repeat until convergence (max iterations / plateau / Claude stop)

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

- `types.py` - Shared dataclasses (Caption, MetricScores, StyleProfile, PromptTemplate, LoopState, KnowledgeBase, Hypothesis, etc.) + category classification helpers
- `config.py` - CLI argument parsing → Config dataclass
- `analyze.py` - Zero-step: parallel Gemini+Claude style analysis → StyleProfile + initial PromptTemplate
- `caption.py` - Gemini Pro captioning with disk cache
- `prompt.py` - Claude meta-prompt proposal/refinement (template structure + values)
- `generate.py` - Gemini Flash image generation with semaphore + retry
- `models.py` - ModelRegistry: lazy-load DINO/LPIPS/HPS/Aesthetics with per-model locks
- `evaluate.py` - Dispatches 4 metrics per image via asyncio.to_thread + Gemini vision comparison
- `utils.py` - Shared helpers: Anthropic streaming/text extraction, Gemini image part builder, MIME map
- `state.py` - JSON persistence (state.json + per-iteration logs)
- `loop.py` - Experiment-based orchestration loop (zero-step → N parallel experiments per iteration → shared KB → convergence)
- `__main__.py` - Entry point

## Directory Conventions

- `src/art_style_search/` - All source code
- `reference_images/` - User-provided reference art (not committed)
- `outputs/` - Generated images by iteration/branch (not committed)
- `logs/` - Iteration logs, captions cache, style profile (not committed)
- `state.json` - Resume state (not committed)

## Evaluation Metrics

Each metric compares a generated image against its specific paired original (not all references):
- **DINO cosine similarity**: Semantic/structural match per image pair. Higher = better.
- **LPIPS**: Perceptual distance per image pair. Lower = better.
- **HPS v2**: How well the generated image matches its caption. Higher = better.
- **LAION Aesthetics**: Aesthetic quality predictor (1-10 scale). Higher = better.

## Code Conventions

- Helpers used by 2+ modules belong in `utils.py`; within a module, extract helpers when the same logic appears in both zero-step and main loop paths (e.g. `_apply_best_result`, `_collect_experiment_results`)
- Data fed to Claude in `refine_template` must appear via exactly one path — if the history formatter includes a field, don't also add a dedicated section for it (or vice versa)
- Evaluation metrics must receive inputs matching their documented semantics — DINO/LPIPS compare against the specific paired reference (not a set centroid), HPS scores against the per-image caption (the generation prompt, not the meta-prompt)
- Persisted collections (`experiment_history`, etc.) must be bounded — cap and drop oldest entries rather than growing without limit in state.json
- Iteration-to-iteration learning uses a shared `KnowledgeBase` on `LoopState` — no persistent branches, just per-iteration experiments feeding one KB
- `BranchState` is legacy (kept for backward compat deserialization of old state.json)
- Hypothesis classification uses keyword matching in `classify_hypothesis()` with `_CATEGORY_SYNONYMS` — extend the synonym map when adding new categories

## Code Style

- Ruff handles linting and formatting (config in pyproject.toml)
- Line length: 120
- Format-on-edit hook is active — do not manually run ruff after edits
