# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Self-improving loop that optimizes a meta-prompt for art-style capture and image recreation. The meta-prompt (1200-1800 words) instructs a captioner (Gemini Pro) how to describe images with labeled style-guidance sections so that: (a) a generator (Gemini Flash) can recreate them from the captions, and (b) the style guidance can be reused to generate new art in the same style. Inspired by karpathy/autoresearch.

## Architecture

- **Claude (Anthropic) / GPT / GLM / local model**: Brain/optimizer. Analyzes reproduction quality, refines the meta-prompt. Supports `--reasoning-provider local` with `--reasoning-base-url` for OpenAI-compatible servers (vLLM, SGLang, Ollama).
- **Gemini 3.1 Pro Preview**: Captioner. Describes reference images using the meta-prompt instructions.
- **Gemini 3.1 Flash Image Preview**: Generator. Produces images from per-image captions.

## Loop Flow

0. **Zero-step**: Fix 20 reference images. Caption them, analyze style → `StyleProfile` + N diverse initial meta-prompts. Evaluate all, pick best.
1. Claude proposes N experiments in a single batched call (hypothesis-driven template variants, `<branch>` tags) from shared KB, each with a different hypothesis
2. Each experiment in parallel: meta-prompt + reference → caption → generate → evaluate
3. Compare each (original, generated) pair: per-image paired metrics (DreamSim, HPS, aesthetics) + Gemini vision comparison (style + subject fidelity)
3.5/3.7/3.9. In parallel: synthesis reasoning + pairwise comparison + independent review. Then synthesis experiment runs on the merged template.

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
- `--reasoning-base-url` - Base URL for local/remote OpenAI-compatible server (when using `--reasoning-provider local`)

## Module Map

- `types.py` - Shared dataclasses (Caption, MetricScores, StyleProfile, PromptTemplate, LoopState, KnowledgeBase, Hypothesis, ReviewResult, RunManifest, PromotionDecision, PromotionTestResult, ReplicatedEvaluation, etc.) + category classification helpers. PromptTemplate includes `caption_sections` (ordered labeled output sections for captions) and `caption_length_target` (target word count for captions).
- `config.py` - CLI argument parsing → Config dataclass. Includes `--seed` (RNG seed), `--protocol {classic, rigorous}` for scientific rigor mode.
- `analyze.py` - Zero-step: parallel Gemini+Claude style analysis → StyleProfile + initial PromptTemplate
- `caption.py` - Gemini Pro captioning with disk cache
- `prompt.py` - Claude meta-prompt proposal/refinement (template structure + values); `propose_experiments` batches N proposals in one call using `<branch>` tags; `RefinementResult` dataclass for structured returns; `review_iteration` provides independent CycleResearcher-inspired review of experiment outcomes
- `generate.py` - Gemini Flash image generation with semaphore + retry + disk cache (skips API on resume). Public API: `generate_single`.
- `experiment.py` - Single-experiment execution (pipelined caption→generate per-image + evaluate), `ExperimentProposal` dataclass (carries `analysis`, `template_changes`, `changed_section`, `target_category` from parsing through to `IterationResult`), result collection helpers, `replicate_experiment` for confirmatory validation (rigorous mode)
- `knowledge.py` - Knowledge Base maintenance (hypothesis tracking, open problems); `build_caption_diffs` compares consecutive iterations' best captions for drift detection
- `models.py` - ModelRegistry: lazy-load DreamSim/HPS/Aesthetics/SSIM with per-model locks
- `evaluate.py` - Dispatches metrics per image via asyncio.to_thread + Gemini vision comparison; also `pairwise_compare_experiments` (SPO-inspired head-to-head), `check_caption_compliance` (keyword/section/length checks), `compute_style_consistency` (Jaccard overlap of [Art Style] blocks). `evaluate_images` returns 2-tuple `(scores, n_eval_failed)` with zero-score sentinels for failed evaluations; callers aggregate with their own `completion_rate`. `aggregate` accepts optional `completion_rate` kwarg.
- `utils.py` - Shared helpers: Anthropic streaming/text extraction, Gemini image part builder, MIME map, XML tag extraction, async retry, `build_ref_gen_pairs` (reference/generated pairing from caption-index filenames, used by `loop.py` and `report.py`), `CATEGORY_SYNONYMS` (canonical hypothesis-category synonym map shared by `scoring.classify_hypothesis`, `types.get_category_names`, and `loop._should_honor_stop`)
- `runs.py` - Run directory management: resolve/create/list/clean isolated run directories under `runs/`
- `state.py` - JSON persistence (state.json + per-iteration logs); `load_iteration_log` is the public reader for one log file (inverse of `save_iteration_log`). Public serialization helpers: `to_dict`, `prompt_template_from_dict`, `style_profile_from_dict` (used by `analyze.py` for style cache). Also `save_manifest`/`load_manifest` (run provenance) and `append_promotion_log`/`load_promotion_log` (promotion decision JSONL).
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
- **Style consistency** (6%): Jaccard word-overlap of [Art Style] blocks across captions. Higher = more consistent shared style guidance.
- **Vision style** (5%): Per-image Gemini ternary comparison of style fidelity (MATCH=1.0, PARTIAL=0.5, MISS=0.0). Higher = better.
- **Vision subject** (1%): Per-image Gemini ternary comparison of subject fidelity. Deliberately low weight — subject reproduction is structurally limited by text-to-image generation; the system optimizes for style capture. Higher = better.
- **Vision composition** (4%): Per-image Gemini ternary comparison of spatial layout. Higher = better.

## Code Conventions

- Helpers and constants used by 2+ modules belong in `utils.py` (derive from existing data where possible, e.g. `IMAGE_EXTENSIONS = frozenset(MIME_MAP)`); within a module, extract helpers when the same logic appears in both zero-step and main loop paths (e.g. `_apply_best_result`, `_collect_experiment_results`)
- Data fed to Claude in `propose_experiments` / `refine_template` must appear via exactly one path — if the history formatter includes a field, don't also add a dedicated section for it (or vice versa)
- Evaluation metrics must receive inputs matching their documented semantics — DreamSim compares against the specific paired reference (not a set centroid), HPS scores against the per-image caption (the generation prompt, not the meta-prompt)
- `IterationResult` alignment invariant: `image_paths[i]`, `per_image_scores[i]`, and `iteration_captions[i]` all refer to the same image. `per_image_scores` is stored in original (generation) order, never sorted. Sorting for feedback display uses local variables only.
- `evaluate_images` returns zero-score sentinel `MetricScores` (all fields 0.0, `is_fallback=True`) for failed evaluations to preserve index alignment — the returned list always has the same length as input. The second return value `n_eval_failed` counts substitutions. Callers compute their own `completion_rate` and pass it to `aggregate()`. `aggregate()` filters out `is_fallback=True` scores before computing mean/std so sentinels don't contaminate aggregated metrics.
- Completion tracking: `AggregatedMetrics.completion_rate` (succeeded/attempted) penalizes experiments with partial failures in `composite_score` (0.15 weight). `IterationResult.n_images_attempted/n_images_succeeded` record generation-phase completeness. Experiments below 50% generation success (`_MIN_COMPLETION_RATE`) are rejected outright.
- Persisted collections (`experiment_history`, etc.) must be bounded — cap and drop oldest entries rather than growing without limit in state.json
- Iteration-to-iteration learning uses a shared `KnowledgeBase` on `LoopState` — no persistent branches, just per-iteration experiments feeding one KB
- When removing or renaming a metric, field, or function, update all references across source *and* tests — search the entire codebase, not just the file being changed
- Hypothesis classification uses keyword matching in `classify_hypothesis()` against `CATEGORY_SYNONYMS` (defined in `utils.py`, re-imported as `_CATEGORY_SYNONYMS` inside `scoring.py`) — extend the synonym map when adding new categories. `get_category_names()` in `types.py` unions the synonym-map keys into the ranked category pool so `suggest_target_categories()` (in `prompt/_format.py`) can surface unexplored synonym categories (`lighting`, `texture`, `background`, `caption_structure`, …) at priority 1.0 even in fresh runs.
- Scoring: `adaptive_composite_score` ranks experiments against each other (relative); `composite_score` is used for improvement checks against baseline (absolute, same scale, with `improvement_epsilon(baseline)` adaptive threshold to filter generation noise) — never compare values from different scoring functions
- `improvement_epsilon(baseline)` returns `max(IMPROVEMENT_EPSILON * (1 - max(baseline, 0)), 0.001)` — threshold shrinks as score climbs with a floor of 0.001 to prevent false-positive improvements at high baselines
- `composite_score` includes a consistency penalty (0.30 weight) based on per-image std of DreamSim and color histogram, plus a completion penalty (0.15 weight) based on `completion_rate` — experiments with high variance or missing images are penalized. The final score is floor-clamped to 0.0 (never negative) to prevent `adaptive_composite_score` min-max normalization from producing nonsensical rankings
- All metrics in `composite_score` are normalized to [0, 1] before weighting — HPS via `_normalize_hps` (ceiling 0.35), aesthetics /10, vision /10. Weights are shared constants (`_W_DREAMSIM`, `_W_HPS`, etc.) in `scoring.py` — both `composite_score` (aggregated) and `per_image_composite` (single image) derive from them
- KB metric deltas must be computed against the pre-update baseline — `update_knowledge_base` runs BEFORE `_apply_best_result` mutates `state.best_metrics`
- Caption diffs compare iteration N-1's best captions against N-2's via `state.prev_best_captions` (stored before overwriting `last_iteration_results`). `build_caption_diffs` in `knowledge.py` accepts `prev_captions: list[Caption]` directly.
- Caption quality is validated after Gemini returns — empty or too-short captions (<150 chars) raise RuntimeError
- `_caption_and_generate` returns 3-tuple `(captions, generated_paths, pairs)`. Failed images are dropped from the surviving lists; counts are derived by the caller from `len(fixed_refs)` and `len(generated_paths)`. `run_experiment` rejects experiments below `_MIN_COMPLETION_RATE` (50%) generation success.
- Open problems in KB are merged across experiments (deduplicated by text, capped at 10), not replaced — earlier experiments' problems survive
- Vision comparison is per-image (one Gemini call per image pair) with ternary verdicts (MATCH/PARTIAL/MISS → 1.0/0.5/0.0); failures degrade to PARTIAL (0.5) neutral defaults
- Synthesis always runs when >= 2 experiments exist — top 2-3 by `adaptive_composite_score` are merged regardless of whether they individually beat baseline. This allows cherry-picking best sections from experiments that failed overall but improved different aspects. Synthesis reasoning runs in parallel with pairwise comparison and independent review; only the synthesis experiment (caption+generate+eval) runs after.
- Retry and reliability: all Gemini API calls use jittered exponential backoff (`delay * (0.5 + random())`), per-request `asyncio.wait_for` timeouts (180s generation, 90s caption/vision, 120s pairwise), 429 rate-limit detection with 30s base delay, and a shared `gemini_circuit_breaker` that pauses all calls for 60s after 15 consecutive failures. Anthropic streaming retries transient errors by exception class (`httpx.RemoteProtocolError`, `httpcore.ReadError`) first, with string-based fallback for unrecognized transient errors.
- Generated images are disk-cached: `generate_single` skips the API call if `output_path` already exists with >0 bytes. This makes crash+resume skip the entire generation phase.
- Caption+generation runs as a per-image pipeline in `_caption_and_generate` — each image's caption→generate is a single chained task so generation starts as soon as each individual caption completes (no serial boundary between phases). Failures are per-image via `return_exceptions=True`.
- Exploration mechanism: on even plateau counts, the loop adopts the second-best experiment via `_apply_exploration_result` (updates `current_template` only — does NOT touch `best_template`, `best_metrics`, or `global_best_*`) and resets `plateau_counter` to 1 to give exploration runway. `best_template` must stay in sync with `best_metrics` so that rigorous-mode confirmatory validation replicates the correct incumbent. `_apply_best_result` is used only for genuine improvements and guards `global_best_*` with a score comparison so it never regresses.
- `ConvergenceReason.REASONING_STOP` (Claude emitted `[CONVERGED]` or the parser returned zero proposals) is gated behind `_should_honor_stop()` in `loop.py`, which requires ALL three: (1) iteration ≥ `max_iterations * _MIN_ITER_FRACTION_FOR_STOP` (0.5), (2) `plateau_counter ≥ max(plateau_window - 1, 2)`, (3) every `CATEGORY_SYNONYMS` key has at least one hypothesis in the KB. If the guard rejects, the `should_stop` flag is dropped, the refinement is kept as a real proposal, and the zero-proposals path returns `([], False)` so the outer loop bumps the plateau counter naturally. The softened `[CONVERGED]` instruction in `prompt/experiments.py` also pushes Claude to self-check the same conditions before emitting the token.
- Style analysis cache is shared across runs via `runs/.cache/style_{hash}.json` keyed by sorted reference paths+mtimes. New runs with the same ref images skip the 3 API calls entirely. The cache is also copied into each run's `log_dir` for provenance.
- Resume safety: `run()` checks `state.converged` after `load_state` and returns early if the previous run already converged. The `continue` path (empty experiment results) advances `state.iteration` and calls `save_state` before continuing so that a crash+resume does not replay the same doomed iteration forever.
- Meta-prompt is 1200-1800 words with 8-15 sections (4-8 sentences each). The FIRST section must be `style_foundation` — a mandatory, non-removable section with fixed style rules from StyleProfile. The first caption output label must be `[Art Style]`.
- Style consistency is measured via Jaccard word-overlap of [Art Style] blocks across captions and included in composite_score (6% weight).
- Captions have labeled output sections (e.g. `[Art Style]`, `[Color Palette]`). The set of section names, their ordering, and the caption length target are all part of the optimization surface — Claude experiments with these via `caption_sections` and `caption_length_target` on `PromptTemplate`.
- Caption compliance checking verifies keyword coverage (meta-prompt section topics), labeled section marker presence (`[Section Name]` in caption text), section ordering (markers appear in expected order), and section length balance (no single section dominates >50% of words).
- Experiments must change exactly 1 section per experiment (not multiple) for clean attribution. Each experiment declares a `<changed_section>` tag identifying which section was modified.
- The worst discarded experiment's details (hypothesis, caption, vision feedback) are shown to Claude for negative learning — helps avoid repeating failures.
- `propose_experiments` system prompt includes PE2-inspired "Optimization dynamics" section with three principles: **Momentum** (double down on confirmed KB insights), **Step size** (adapt change magnitude to current composite score regime: LOW <0.35 = bold, MODERATE 0.35-0.50 = targeted, HIGH >0.50 = surgical), **Diversity pressure** (deprioritize categories with 3+ rejections and no confirmed insights). The user message also includes the current composite score with regime label so Claude can calibrate.
- Hypothesis variability is enforced at 3 layers: (1) **Prompt-level** — each `<branch>` requires a `<target_category>` tag unique across branches; (2) **Post-parse dedup** — `enforce_hypothesis_diversity()` in `prompt.py` uses the parsed `target_category` from `RefinementResult` (falling back to `classify_hypothesis()` keyword matching if the tag is missing) to drop duplicate-category experiments; (3) **KB-guided targeting** — `suggest_target_categories()` (in `prompt/_format.py`) ranks categories by improvement potential (unexplored=1.0, partial success=0.7, diminishing returns=0.1) and injects the ranked list into the user message for Claude. The dedup filter is called in `loop.py` after `propose_experiments()` returns.
- Number of fixed reference images is configurable via `--num-fixed-refs` (default 20). On resume, existing refs from state.json are used regardless.
- Independent review loop (CycleResearcher-inspired): in parallel with synthesis reasoning (Phase 3.9), `review_iteration` in `prompt.py` sends all experiment results to the reasoning model as a skeptical reviewer. The reviewer assesses each experiment as SIGNAL/NOISE/MIXED, identifies which metric movements are real vs noise, and provides strategic guidance. The `strategic_guidance` is stored in `LoopState.review_feedback` (persisted in state.json) and prepended to `roundtrip_fb` at the start of the next iteration so `propose_experiments` can incorporate the reviewer's recommendations.
- Pairwise experiment comparison (SPO-inspired): in parallel with synthesis reasoning (Phase 3.7), `pairwise_compare_experiments` in `evaluate.py` sends sampled image trios (original, set A reproduction, set B reproduction) from the top 2 experiments to Gemini vision for a head-to-head comparison. Returns a winner (A/B/TIE) with rationale. The rationale is stored in `LoopState.pairwise_feedback` (persisted in state.json) and prepended to `vision_fb` at the start of the next iteration so `propose_experiments` can learn which experiment's approach was visually superior. `build_ref_gen_pairs` in `utils.py` reconstructs (ref, gen) pairs from `IterationResult` by parsing the caption index from generated filenames; both `loop.py` (for pairwise comparison) and `report.py` (for the image grid) use it.
- HTML report metric trajectories must use `composite_score` only — `adaptive_composite_score` is min-max normalized within a single batch and is meaningless across iterations. Within an iteration's experiment table, `adaptive_composite_score` is fine (and useful for ranking) because it's recomputed per-batch.
- Scientific rigor is controlled by `--protocol {classic, rigorous}`. Classic mode preserves current behavior exactly. Rigorous mode enables: (1) information barrier (14 feedback + 6 silent images), (2) confirmatory replication (top-2 + incumbent × 3 replicates), (3) Wilcoxon signed-rank statistical test for promotion (p < 0.10).
- `--seed` provides deterministic reference selection via a `random.Random(seed)` instance on `RunContext`. Retry jitter in `utils.py` stays unseeded (timing, not experiment logic). Seed is persisted in `LoopState.seed` and `run_manifest.json`.
- Information barrier: `_split_information_barrier()` in `loop.py` splits `fixed_refs` into `feedback_refs` (shown to reasoning model in per-image feedback) and `silent_refs` (evaluated but hidden). All images go through captioning + generation + evaluation. Only feedback contexts (`_build_iteration_context`, `_run_pairwise_comparison`) are filtered. Scoring uses ALL images.
- Promotion decisions are logged to `{run_dir}/promotion_log.jsonl` via `append_promotion_log()` — every iteration records decision/score/delta/epsilon regardless of protocol.
- Run provenance: `run_manifest.json` written at run start with seed, CLI args, model names, git SHA, reference image hashes, platform. On resume in rigorous mode, mismatches abort; in classic mode, they warn.
- In rigorous mode, caption cache keys include prompt text hash (`_p{hash[:8]}`) to invalidate when the meta-prompt changes. Classic mode uses the old format for backward compat.
- `per_image_composite()` in `scoring.py` applies the same weights as `composite_score` but to a single `MetricScores` (no variance penalty). Used for paired statistical testing.
- `paired_promotion_test()` in `scoring.py` uses Wilcoxon signed-rank (scipy.stats.wilcoxon, one-sided) with bootstrap 95% CI (2000 resamples). Falls back to sign test on ties. Threshold: p < 0.10 AND effect_size > 0.
- `replicate_experiment()` in `experiment.py` runs N caption+generate+eval cycles per template, computing per-image median scores. Captions cache-hit (same prompt), generation is stochastic (different output paths per replicate).

## Code Style

- Ruff handles linting and formatting (config in pyproject.toml)
- Line length: 120
- Format-on-edit hook is active — do not manually run ruff after edits
