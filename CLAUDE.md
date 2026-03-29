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
1. Claude proposes N experiments in a single batched call (hypothesis-driven template variants, `<branch>` tags) from shared KB, each with a different hypothesis
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
uv run python -m art_style_search --help  # Show all CLI options
uv run python -m art_style_search clean  # Remove outputs, logs, and state
```

## Environment Variables

- `ANTHROPIC_API_KEY` - Anthropic API key for Claude (when using `--reasoning-provider anthropic`)
- `GOOGLE_API_KEY` - Google API key for Gemini models (always required)
- `ZAI_API_KEY` - Z.AI API key for GLM-5 (when using `--reasoning-provider zai`)

## Module Map

- `types.py` - Shared dataclasses (Caption, MetricScores, StyleProfile, PromptTemplate, LoopState, KnowledgeBase, Hypothesis, etc.) + category classification helpers
- `config.py` - CLI argument parsing → Config dataclass
- `analyze.py` - Zero-step: parallel Gemini+Claude style analysis → StyleProfile + initial PromptTemplate
- `caption.py` - Gemini Pro captioning with disk cache
- `prompt.py` - Claude meta-prompt proposal/refinement (template structure + values); `propose_experiments` batches N proposals in one call using `<branch>` tags; `refine_template` exists for single-experiment use; `RefinementResult` dataclass for structured returns
- `generate.py` - Gemini Flash image generation with semaphore + retry
- `experiment.py` - Single-experiment execution (caption + generate + evaluate), `ExperimentProposal` dataclass, result collection helpers
- `knowledge.py` - Knowledge Base maintenance (hypothesis tracking, open problems, caption diffs)
- `models.py` - ModelRegistry: lazy-load DINO/LPIPS/HPS/Aesthetics/Texture/SSIM with per-model locks
- `evaluate.py` - Dispatches metrics per image via asyncio.to_thread + Gemini vision comparison
- `utils.py` - Shared helpers: Anthropic streaming/text extraction, Gemini image part builder, MIME map
- `state.py` - JSON persistence (state.json + per-iteration logs)
- `loop.py` - Experiment-based orchestration loop (zero-step → N parallel experiments per iteration → shared KB → convergence)
- `__main__.py` - Entry point

## Directory Conventions

- `src/art_style_search/` - All source code
- `reference_images/` - User-provided reference art (not committed)
- `outputs/` - Generated images by iteration/experiment (not committed)
- `logs/` - Iteration logs, captions cache, style profile (not committed)
- `state.json` - Resume state (not committed)

## Evaluation Metrics

Each metric compares a generated image against its specific paired original (not all references):
- **DINO cosine similarity** (31%): Semantic/structural match per image pair. Higher = better.
- **LPIPS** (-14%): Perceptual distance per image pair (normalized: raw / 0.7, clamped to 1.0). Lower = better.
- **Color histogram** (14%): HSV histogram intersection. Higher = better.
- **Texture** (10%): Gabor filter energy cosine similarity. Higher = better.
- **SSIM** (8%): Structural similarity index for pixel-level comparison. Higher = better.
- **HPS v2** (5%): Caption-image alignment (normalized: raw / 0.35, clamped to 1.0). Higher = better.
- **LAION Aesthetics** (6%): Aesthetic quality predictor (1-10 scale, normalized /10). Higher = better.
- **Vision scores (style/subject/composition)** (4% each = 12%): Per-image Gemini ternary comparison (MATCH=1.0, PARTIAL=0.5, MISS=0.0). Higher = better.

## Code Conventions

- Helpers used by 2+ modules belong in `utils.py`; within a module, extract helpers when the same logic appears in both zero-step and main loop paths (e.g. `_apply_best_result`, `_collect_experiment_results`)
- Data fed to Claude in `propose_experiments` / `refine_template` must appear via exactly one path — if the history formatter includes a field, don't also add a dedicated section for it (or vice versa)
- Evaluation metrics must receive inputs matching their documented semantics — DINO/LPIPS compare against the specific paired reference (not a set centroid), HPS scores against the per-image caption (the generation prompt, not the meta-prompt)
- Persisted collections (`experiment_history`, etc.) must be bounded — cap and drop oldest entries rather than growing without limit in state.json
- Iteration-to-iteration learning uses a shared `KnowledgeBase` on `LoopState` — no persistent branches, just per-iteration experiments feeding one KB
- `BranchState` is legacy (kept for backward compat deserialization of old state.json)
- Hypothesis classification uses keyword matching in `classify_hypothesis()` with `_CATEGORY_SYNONYMS` — extend the synonym map when adding new categories
- Scoring: `adaptive_composite_score` ranks experiments against each other (relative); `composite_score` is used for improvement checks against baseline (absolute, same scale, with `IMPROVEMENT_EPSILON` threshold to filter generation noise) — never compare values from different scoring functions
- `composite_score` includes a consistency penalty (0.30 weight) based on per-image std of DINO, LPIPS, color histogram, and texture — experiments with high variance across images are penalized
- All metrics in `composite_score` are normalized to [0, 1] before weighting — LPIPS via `_normalize_lpips` (ceiling 0.7), HPS via `_normalize_hps` (ceiling 0.35), aesthetics /10, vision /10
- KB metric deltas must be computed against the pre-update baseline — `update_knowledge_base` runs BEFORE `_apply_best_result` mutates `state.best_metrics`
- Caption quality is validated after Gemini returns — empty or too-short captions (<50 chars) raise RuntimeError
- Open problems in KB are merged across experiments (deduplicated by text, capped at 10), not replaced — earlier experiments' problems survive
- Vision comparison is per-image (one Gemini call per image pair) with ternary verdicts (MATCH/PARTIAL/MISS → 1.0/0.5/0.0); failures degrade to PARTIAL (0.5) neutral defaults
- Exploration mechanism: on even plateau counts (2, 4, 6, ...), the loop adopts the second-best experiment (ranked by `adaptive_composite_score`) to escape local optima; odd counts stay greedy (alternating exploration/exploitation). Requires >= 2 experiments to trigger.

## Code Style

- Ruff handles linting and formatting (config in pyproject.toml)
- Line length: 120
- Format-on-edit hook is active — do not manually run ruff after edits
