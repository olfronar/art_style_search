# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Self-improving loop that optimizes a meta-prompt for art-style capture and image recreation. The meta-prompt (1200-1800 words) instructs a captioner (Gemini Pro) how to describe images with labeled style-guidance sections so that: (a) a generator (Gemini Flash) can recreate them from the captions, and (b) the style guidance can be reused to generate new art in the same style. Inspired by karpathy/autoresearch.

## Architecture

- **Claude (Anthropic)**: Brain/optimizer. Analyzes reproduction quality, refines the meta-prompt.
- **Gemini 3.1 Pro Preview**: Captioner. Describes reference images using the meta-prompt instructions.
- **Gemini 3.1 Flash Image Preview**: Generator. Produces images from per-image captions.

## Loop Flow

0. **Zero-step**: Fix 20 reference images. Caption them, analyze style → `StyleProfile` + N diverse initial meta-prompts. Evaluate all, pick best.
1. Claude proposes N experiments in a single batched call (hypothesis-driven template variants, `<branch>` tags) from shared KB, each with a different hypothesis
2. Each experiment in parallel: meta-prompt + reference → caption → generate → evaluate
3. Compare each (original, generated) pair: per-image paired metrics (DreamSim, HPS, aesthetics) + Gemini vision comparison (style + subject fidelity)
3.5. Top experiments synthesized into merged template
3.9. Independent review (CycleResearcher-inspired): skeptical assessment of metric movements, noise vs signal, strategic guidance for next iteration
4. Best experiment updates the current template; all results feed into shared KnowledgeBase
5. Repeat until convergence (max iterations / plateau / Claude stop)

## Commands

This project uses [**uv**](https://docs.astral.sh/uv/) as its package and tool manager. Every Python command goes through `uv run`, dependencies are installed with `uv sync`, and external CLI tools (e.g. `pre-commit`) should be installed via `uv tool install <name>` — not `pipx` or `brew`.

```bash
uv sync                                              # Install all dependencies
uv run ruff check .                                  # Lint
uv run ruff format .                                 # Format
uv run pytest tests/                                 # Run tests
uv run python -m art_style_search                    # New auto-named run (runs/run_001/)
uv run python -m art_style_search --run my-test      # Resume or create named run
uv run python -m art_style_search --run my-test --new  # Force new (error if exists)
uv run python -m art_style_search --help             # Show all CLI options
uv run python -m art_style_search list               # List all runs with status
uv run python -m art_style_search clean --run NAME   # Remove a specific run
uv run python -m art_style_search clean --all        # Remove all runs
uv run python -m art_style_search report --run NAME  # Generate runs/<NAME>/report.html
uv run python -m art_style_search report --run NAME --open  # Generate and open in browser
uv run python -m art_style_search report --all       # Regenerate reports for all runs
uv tool install pre-commit                           # Install the pre-commit CLI (optional)
```

## Environment Variables

- `ANTHROPIC_API_KEY` - Anthropic API key for Claude (when using `--reasoning-provider anthropic`)
- `GOOGLE_API_KEY` - Google API key for Gemini models (always required)
- `ZAI_API_KEY` - Z.AI API key for GLM-5.1 (when using `--reasoning-provider zai`)
- `OPENAI_API_KEY` - OpenAI API key for GPT-5.4 (when using `--reasoning-provider openai`)

## Module Map

- `types.py` - Shared dataclasses (Caption, MetricScores, StyleProfile, PromptTemplate, LoopState, KnowledgeBase, Hypothesis, ReviewResult, etc.) + category classification helpers. PromptTemplate includes `caption_sections` (ordered labeled output sections for captions) and `caption_length_target` (target word count for captions).
- `config.py` - CLI argument parsing → Config dataclass
- `analyze.py` - Zero-step: parallel Gemini+Claude style analysis → StyleProfile + initial PromptTemplate
- `caption.py` - Gemini Pro captioning with disk cache
- `prompt.py` - Claude meta-prompt proposal/refinement (template structure + values); `propose_experiments` batches N proposals in one call using `<branch>` tags; `RefinementResult` dataclass for structured returns; `review_iteration` provides independent CycleResearcher-inspired review of experiment outcomes
- `generate.py` - Gemini Flash image generation with semaphore + retry
- `experiment.py` - Single-experiment execution (caption + generate + evaluate), `ExperimentProposal` dataclass, result collection helpers
- `knowledge.py` - Knowledge Base maintenance (hypothesis tracking, open problems); `build_caption_diffs` compares consecutive iterations' best captions for drift detection
- `models.py` - ModelRegistry: lazy-load DreamSim/HPS/Aesthetics/SSIM with per-model locks
- `evaluate.py` - Dispatches metrics per image via asyncio.to_thread + Gemini vision comparison; also `pairwise_compare_experiments` (SPO-inspired head-to-head), `check_caption_compliance` (keyword/section/length checks), `compute_style_consistency` (Jaccard overlap of [Art Style] blocks)
- `utils.py` - Shared helpers: Anthropic streaming/text extraction, Gemini image part builder, MIME map, XML tag extraction, async retry, `build_ref_gen_pairs` (reference/generated pairing from caption-index filenames, used by `loop.py` and `report.py`), `CATEGORY_SYNONYMS` (canonical hypothesis-category synonym map shared by `scoring.classify_hypothesis`, `types.get_category_names`, and `loop._should_honor_stop`)
- `runs.py` - Run directory management: resolve/create/list/clean isolated run directories under `runs/`
- `state.py` - JSON persistence (state.json + per-iteration logs); `load_iteration_log` is the public reader for one log file (inverse of `save_iteration_log`)
- `loop.py` - Experiment-based orchestration loop (zero-step → N parallel experiments per iteration → shared KB → convergence)
- `report.py` - Post-run HTML report generator. `build_report(run_dir)` writes `runs/<name>/report.html` (Plotly metric trajectories, per-iteration drill-down, image comparison grid, hypothesis tree + KB state). Plotly is lazy-imported so the `list`/`clean` commands stay fast. Images use relative paths via `_rel(target, report_dir)`.
- `__main__.py` - Entry point

## Directory Conventions

- `src/art_style_search/` - All source code
- `reference_images/` - User-provided reference art (not committed)
- `runs/` - All run data, each run in its own subdirectory (not committed):
  - `runs/<name>/outputs/` - Generated images by iteration/experiment
  - `runs/<name>/logs/` - Iteration logs (`iter_NNN_branch_M.json`), captions cache, style profile, `best_prompt.txt`
  - `runs/<name>/state.json` - Resume state
  - `runs/<name>/report.html` - Post-run HTML report (generated on demand via `report` subcommand)

## Evaluation Metrics

Each metric compares a generated image against its specific paired original (not all references):
- **DreamSim** (40%): Human-aligned perceptual similarity capturing semantic content, layout, color, pose (replaces DINO + LPIPS). Higher = better.
- **Color histogram** (22%): HSV histogram intersection. Higher = better.
- **SSIM** (11%): Structural similarity index for pixel-level comparison. Higher = better.
- **HPS v2** (5%): Caption-image alignment (normalized: raw / 0.35, clamped to 1.0). Higher = better.
- **LAION Aesthetics** (6%): Aesthetic quality predictor (1-10 scale, normalized /10). Higher = better.
- **Style consistency** (4%): Jaccard word-overlap of [Art Style] blocks across captions. Higher = more consistent shared style guidance.
- **Vision scores (style/subject/composition)** (4% each = 12%): Per-image Gemini ternary comparison (MATCH=1.0, PARTIAL=0.5, MISS=0.0). Higher = better.

## Code Conventions

- Helpers and constants used by 2+ modules belong in `utils.py` (derive from existing data where possible, e.g. `IMAGE_EXTENSIONS = frozenset(MIME_MAP)`); within a module, extract helpers when the same logic appears in both zero-step and main loop paths (e.g. `_apply_best_result`, `_collect_experiment_results`)
- Data fed to Claude in `propose_experiments` / `refine_template` must appear via exactly one path — if the history formatter includes a field, don't also add a dedicated section for it (or vice versa)
- Evaluation metrics must receive inputs matching their documented semantics — DreamSim compares against the specific paired reference (not a set centroid), HPS scores against the per-image caption (the generation prompt, not the meta-prompt)
- Persisted collections (`experiment_history`, etc.) must be bounded — cap and drop oldest entries rather than growing without limit in state.json
- Iteration-to-iteration learning uses a shared `KnowledgeBase` on `LoopState` — no persistent branches, just per-iteration experiments feeding one KB
- When removing or renaming a metric, field, or function, update all references across source *and* tests — search the entire codebase, not just the file being changed
- Hypothesis classification uses keyword matching in `classify_hypothesis()` against `CATEGORY_SYNONYMS` (defined in `utils.py`, re-imported as `_CATEGORY_SYNONYMS` inside `scoring.py`) — extend the synonym map when adding new categories. `get_category_names()` in `types.py` unions the synonym-map keys into the ranked category pool so `KnowledgeBase.suggest_target_categories()` can surface unexplored synonym categories (`lighting`, `texture`, `background`, `caption_structure`, …) at priority 1.0 even in fresh runs.
- Scoring: `adaptive_composite_score` ranks experiments against each other (relative); `composite_score` is used for improvement checks against baseline (absolute, same scale, with `improvement_epsilon(baseline)` adaptive threshold to filter generation noise) — never compare values from different scoring functions
- `improvement_epsilon(baseline)` returns `max(IMPROVEMENT_EPSILON * (1 - max(baseline, 0)), 0.001)` — threshold shrinks as score climbs with a floor of 0.001 to prevent false-positive improvements at high baselines
- `composite_score` includes a consistency penalty (0.30 weight) based on per-image std of DreamSim and color histogram — experiments with high variance across images are penalized
- All metrics in `composite_score` are normalized to [0, 1] before weighting — HPS via `_normalize_hps` (ceiling 0.35), aesthetics /10, vision /10
- KB metric deltas must be computed against the pre-update baseline — `update_knowledge_base` runs BEFORE `_apply_best_result` mutates `state.best_metrics`
- Caption diffs compare iteration N-1's best captions against N-2's via `state.prev_best_captions` (stored before overwriting `last_iteration_results`). `build_caption_diffs` in `knowledge.py` accepts `prev_captions: list[Caption]` directly.
- Caption quality is validated after Gemini returns — empty or too-short captions (<150 chars) raise RuntimeError
- Open problems in KB are merged across experiments (deduplicated by text, capped at 10), not replaced — earlier experiments' problems survive
- Vision comparison is per-image (one Gemini call per image pair) with ternary verdicts (MATCH/PARTIAL/MISS → 1.0/0.5/0.0); failures degrade to PARTIAL (0.5) neutral defaults
- Synthesis always runs when >= 2 experiments exist — top 2-3 by `adaptive_composite_score` are merged regardless of whether they individually beat baseline. This allows cherry-picking best sections from experiments that failed overall but improved different aspects.
- Exploration mechanism: on even plateau counts, the loop adopts the second-best experiment via `_apply_exploration_result` (updates `current_template`/`best_template` only — does NOT touch `best_metrics` or `global_best_*`) and resets `plateau_counter` to 1 to give exploration runway. `_apply_best_result` is used only for genuine improvements and guards `global_best_*` with a score comparison so it never regresses.
- `ConvergenceReason.REASONING_STOP` (Claude emitted `[CONVERGED]` or the parser returned zero proposals) is gated behind `_should_honor_stop()` in `loop.py`, which requires ALL three: (1) iteration ≥ `max_iterations * _MIN_ITER_FRACTION_FOR_STOP` (0.5), (2) `plateau_counter ≥ max(plateau_window - 1, 2)`, (3) every `CATEGORY_SYNONYMS` key has at least one hypothesis in the KB. If the guard rejects, the `should_stop` flag is dropped, the refinement is kept as a real proposal, and the zero-proposals path returns `([], False)` so the outer loop bumps the plateau counter naturally. The softened `[CONVERGED]` instruction in `prompt/experiments.py` also pushes Claude to self-check the same conditions before emitting the token.
- Meta-prompt is 1200-1800 words with 8-15 sections (4-8 sentences each). The FIRST section must be `style_foundation` — a mandatory, non-removable section with fixed style rules from StyleProfile. The first caption output label must be `[Art Style]`.
- Style consistency is measured via Jaccard word-overlap of [Art Style] blocks across captions and included in composite_score (4% weight).
- Captions have labeled output sections (e.g. `[Art Style]`, `[Color Palette]`). The set of section names, their ordering, and the caption length target are all part of the optimization surface — Claude experiments with these via `caption_sections` and `caption_length_target` on `PromptTemplate`.
- Caption compliance checking verifies keyword coverage (meta-prompt section topics), labeled section marker presence (`[Section Name]` in caption text), section ordering (markers appear in expected order), and section length balance (no single section dominates >50% of words).
- Experiments must change exactly 1 section per experiment (not multiple) for clean attribution. Each experiment declares a `<changed_section>` tag identifying which section was modified.
- The worst discarded experiment's details (hypothesis, caption, vision feedback) are shown to Claude for negative learning — helps avoid repeating failures.
- `propose_experiments` system prompt includes PE2-inspired "Optimization dynamics" section with three principles: **Momentum** (double down on confirmed KB insights), **Step size** (adapt change magnitude to current composite score regime: LOW <0.35 = bold, MODERATE 0.35-0.50 = targeted, HIGH >0.50 = surgical), **Diversity pressure** (deprioritize categories with 3+ rejections and no confirmed insights). The user message also includes the current composite score with regime label so Claude can calibrate.
- Hypothesis variability is enforced at 3 layers: (1) **Prompt-level** — each `<branch>` requires a `<target_category>` tag unique across branches; (2) **Post-parse dedup** — `enforce_hypothesis_diversity()` in `prompt.py` uses the parsed `target_category` from `RefinementResult` (falling back to `classify_hypothesis()` keyword matching if the tag is missing) to drop duplicate-category experiments; (3) **KB-guided targeting** — `KnowledgeBase.suggest_target_categories()` ranks categories by improvement potential (unexplored=1.0, partial success=0.7, diminishing returns=0.1) and injects the ranked list into the user message for Claude. The dedup filter is called in `loop.py` after `propose_experiments()` returns.
- Number of fixed reference images is configurable via `--num-fixed-refs` (default 20). On resume, existing refs from state.json are used regardless.
- Independent review loop (CycleResearcher-inspired): after synthesis (Phase 3.9), `review_iteration` in `prompt.py` sends all experiment results to the reasoning model as a skeptical reviewer. The reviewer assesses each experiment as SIGNAL/NOISE/MIXED, identifies which metric movements are real vs noise, and provides strategic guidance. The `strategic_guidance` is stored in `LoopState.review_feedback` (persisted in state.json) and prepended to `roundtrip_fb` at the start of the next iteration so `propose_experiments` can incorporate the reviewer's recommendations.
- Pairwise experiment comparison (SPO-inspired): after synthesis (Phase 3.7), `pairwise_compare_experiments` in `evaluate.py` sends sampled image trios (original, set A reproduction, set B reproduction) from the top 2 experiments to Gemini vision for a head-to-head comparison. Returns a winner (A/B/TIE) with rationale. The rationale is stored in `LoopState.pairwise_feedback` (persisted in state.json) and prepended to `vision_fb` at the start of the next iteration so `propose_experiments` can learn which experiment's approach was visually superior. `build_ref_gen_pairs` in `utils.py` reconstructs (ref, gen) pairs from `IterationResult` by parsing the caption index from generated filenames; both `loop.py` (for pairwise comparison) and `report.py` (for the image grid) use it.
- HTML report metric trajectories must use `composite_score` only — `adaptive_composite_score` is min-max normalized within a single batch and is meaningless across iterations. Within an iteration's experiment table, `adaptive_composite_score` is fine (and useful for ranking) because it's recomputed per-batch.

## Code Style

- Ruff handles linting and formatting (config in pyproject.toml)
- Line length: 120
- Format-on-edit hook is active — do not manually run ruff after edits
