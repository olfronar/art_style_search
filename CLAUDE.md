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
1. Claude emits a single JSON batch of 8-12 raw proposals grouped into 3 mechanism-tagged directions (`D1`/`D2`/`D3`); each direction carries 1 targeted proposal + 1-3 bold proposals. The workflow dedups on `(category, failure_mechanism, intervention_type)` and runs `select_experiment_portfolio` to pick up to `--num-branches` experiments to evaluate
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
- `XAI_API_KEY` - xAI API key for Grok (when using `--reasoning-provider xai` and/or `--comparison-provider xai`)
- `--reasoning-base-url` - Base URL for local/remote OpenAI-compatible server (when using `--reasoning-provider local`)

## Module Map

### Entry + orchestration
- `__main__.py` - Entry point with `list`, `clean`, `report` subcommands; default runs `loop.run(config)`.
- `loop.py` - Thin façade re-exporting `run` from the `workflow` package (kept for a stable import surface).
- `workflow/` - Internal orchestration package:
  - `context.py` - `RunContext`, `_setup_run_context`, `_finalize_run`, image discovery, manifest handling, logging.
  - `iteration.py` - Façade re-exporting the per-iteration phase modules below.
  - `iteration_context.py` - Phase 0: builds `(vision_fb, roundtrip_fb, caption_diffs)` from the previous iteration; `_filter_feedback_by_refs` honors the information barrier.
  - `iteration_proposals.py` - Phase 1: `_propose_iteration_experiments` — calls the reasoner, dedups by `target_category`, produces `ExperimentProposal` list.
  - `iteration_execution.py` - Phases 2–3: `IterationRanking`, parallel experiment execution, pairwise compare, independent review, synthesis, confirmatory validation, final ranking.
  - `iteration_persistence.py` - Phase 4: writes iteration state/logs; `_update_knowledge_base_for_iteration`.
  - `policy.py` - Promotion/convergence/exploration: `_apply_best_result`, `_apply_exploration_result`, `_should_honor_stop`, `_candidate_results_for_validation`.
  - `services.py` - Provider-adapter dataclasses: `CaptioningService`, `EvaluationService`, `GenerationService`, `ReasoningService`, `RunServices`.
  - `zero_step.py` - Zero-step bootstrap: initial templates, sanitization, first baseline experiments.

### Data + types
- `types.py` - Shared dataclasses (Caption, MetricScores, AggregatedMetrics, CaptionComplianceStats, StyleProfile, PromptTemplate, LoopState, KnowledgeBase, Hypothesis, ReviewResult, RunManifest, PromotionDecision, PromotionTestResult, ReplicatedEvaluation, etc.). PromptTemplate includes `caption_sections` (ordered labeled output sections) and `caption_length_target`. `compliance_components_mean()` averages `CaptionComplianceStats` components with a divisor derived from `len(fields(CaptionComplianceStats))` — adding a sixth compliance metric updates the divisor automatically. `IterationResult`, `Hypothesis`, `ExperimentProposal`, and `RefinementResult` carry directional-search metadata: `direction_id` (`D1`/`D2`/`D3`), `direction_summary`, `failure_mechanism`, `intervention_type`, `risk_level` (`"targeted"` or `"bold"`), `expected_primary_metric`, `expected_tradeoff`, plus `changed_sections: list[str]` alongside the singular `changed_section`. `CategoryProgress` adds `last_mechanism_tried` / `last_confirmed_mechanism`.
- `contracts.py` - Transient (non-persisted) workflow dataclasses: `Lessons`, `RefinementResult`, `ExperimentProposal`.
- `taxonomy.py` - `CATEGORY_SYNONYMS` — canonical hypothesis-category synonym map (re-exported from `utils.py` for back-compat).
- `config.py` - CLI argument parsing → `Config` dataclass. Includes `--seed`, `--protocol {classic, rigorous}`, `--reasoning-provider`, `--comparison-provider`, etc.

### Reasoning-model interface (`prompt/` package)
- `prompt/__init__.py` - Public API re-exporting every symbol (`propose_experiments`, `propose_initial_templates`, `synthesize_templates`, `review_iteration`, `validate_template`, `enforce_hypothesis_diversity`, format helpers, etc.).
- `prompt/_format.py` - Dataclass-to-text rendering: `_format_style_profile`, `_format_template`, `_format_metrics`, `format_knowledge_base`, `suggest_target_categories` (PE2-inspired category-gap ranker).
- `prompt/_parse.py` - Response parsing + `validate_template`. Required-anchor validation is table-driven via `_REQUIRED_SECTION_ANCHORS` / `_REQUIRED_CAPTION_ANCHORS` + `_check_anchors` helper (adding a third required anchor is a one-line edit).
- `prompt/json_contracts.py` - JSON schema hints and `validate_*_payload` helpers per reasoner exchange type.
- `prompt/initial.py` - Zero-step: `propose_initial_templates` (N diverse initial meta-prompts).
- `prompt/experiments.py` - Per-iteration `propose_experiments` (single JSON batch of 8-12 raw proposals grouped into 3 directions) + `enforce_hypothesis_diversity` (drops duplicates keyed on `(category, failure_mechanism, intervention_type)`) + `select_experiment_portfolio` (picks 1 targeted per direction in D1→D2→D3 order, then fills remaining slots with bolds up to `num_branches`). System prompt uses tiered priority (TIER 1 critical rules → TIER 2 core task → TIER 3 execution → TIER 4 diagnostics) + execution checklist before response format.
- `prompt/synthesis.py` - `synthesize_templates`: merges top 2-3 experiments' best sections into a single template. Decision criteria: DreamSim > vision_subject > color_histogram for section selection.
- `prompt/review.py` - CycleResearcher-inspired `review_iteration` (skeptical reviewer → `strategic_guidance`). Phased into assess → synthesize → advise.

### Pipeline: caption → generate → evaluate
- `analyze.py` - Zero-step: parallel Gemini vision + reasoning-model style analysis → `StyleProfile` + initial `PromptTemplate` seeds. Cached via `runs/.cache/style_{hash}.json`. Bootstrap captions use a generic section format; the compilation prompt explicitly notes this so the template defines its own section names.
- `caption.py` - Gemini Pro captioning with disk cache. Validates captions are non-empty and ≥ 150 chars. Zero-step `CAPTION_PROMPT` uses fixed sections with Subjects as most important, specific color names, and 200-400 word target.
- `caption_sections.py` - `parse_labeled_sections` (single-pass `[Section]` dict parse, used by compliance and style-consistency checks) + `build_generation_prompt` (reorders caption so `[Subject]` leads the generator prompt, falling back to raw caption when either anchor is missing).
- `generate.py` - Gemini Flash image generation with semaphore + retry + disk cache (skips API on resume). Atomic temp-file + rename writes prevent partial files on crash.
- `experiment.py` - Single-experiment pipeline (per-image caption→generate chain, then parallel evaluate + vision comparison) via `RunServices`. Combined `compute_caption_compliance` call replaces the previous double-scan. `replicate_experiment` runs N confirmatory replicates for rigorous mode; `_format_experiment_feedback` builds vision/roundtrip feedback strings.
- `evaluate.py` - Per-image paired metrics dispatched via `asyncio.to_thread`, Gemini/xAI vision comparison, and caption compliance. `compute_caption_compliance` parses each caption once and returns `(CaptionComplianceStats, prose)` — `check_caption_compliance` and `compute_caption_compliance_stats` are thin wrappers. `compute_style_consistency` uses `parse_labeled_sections` for the `[Art Style]` block. `evaluate_images` returns `(scores, n_eval_failed)` with zero-score sentinels preserving index alignment; callers derive their own `completion_rate`.
- `scoring.py` - `composite_score`, `adaptive_composite_score`, `per_image_composite`, `classify_hypothesis`, `paired_promotion_test` (Wilcoxon + bootstrap CI). Weight constants + penalty constants live at module top. `_VISION_SUBJECT_FLOOR=0.35` / `_W_VISION_SUBJECT_FLOOR_PENALTY=0.05` form a floor-penalty on the subject dimension. `compliance_components_mean` is used for the compliance term.
- `knowledge.py` - Knowledge Base maintenance (hypothesis tracking, open problems, category progress). `build_caption_diffs` compares consecutive iterations' best captions for drift. Open problem text has `[HIGH]/[MED]/[LOW]` prefix stripped at creation time; parsed priority overrides the code heuristic. Near-duplicate problems are merged via token Jaccard similarity (threshold 0.6).
- `models.py` - `ModelRegistry`: lazy-load DreamSim / HPS / Aesthetics / SSIM with per-model locks; auto-detects MPS / CUDA / CPU.

### Providers, retry, shared helpers
- `reasoning_client.py` - Provider-agnostic `ReasoningClient` supporting Anthropic / OpenAI / xAI / Z.AI / local (OpenAI-compatible) backends; streaming + text-extraction helpers.
- `retry.py` - `async_retry` with jittered exponential backoff, 429/ResourceExhausted detection, and per-surface circuit breakers: `caption_circuit_breaker`, `generation_circuit_breaker`, `vision_circuit_breaker` (60s pause after 15 consecutive failures). `gemini_circuit_breaker` is a deprecated alias for `caption_circuit_breaker`.
- `media.py` - `MIME_MAP`, `IMAGE_EXTENSIONS`, Gemini image-part builder, xAI data-URL helpers + size limits.
- `utils.py` - Shared helpers re-exporting `CATEGORY_SYNONYMS` from `taxonomy.py`; `build_ref_gen_pairs` (reference/generated pairing from caption-index filenames, used by `loop.py`/`workflow/` and `report.py`); `extract_xml_tag`; async-retry and image-part re-exports.

### Persistence + runs
- `runs.py` - Run directory management: resolve/create/list/clean isolated run directories under `runs/`.
- `state.py` - Public JSON persistence: `save_state`/`load_state`, per-iteration logs (`save_iteration_log`/`load_iteration_log`), `save_manifest`/`load_manifest`, `append_promotion_log`/`load_promotion_log`. Transient fields (`review_feedback`, `pairwise_feedback`) are excluded from serialization — cleared each iteration.
- `state_codec.py` - Low-level encoders/decoders: `_Encoder` (handles Path, dataclasses, enums), plus `*_from_dict` helpers used by `analyze.py` for the style cache.
- `state_migrations.py` - Schema version constants (`_SCHEMA_VERSION=4`, manifest=3, iteration-log=1, promotion-log=1) and payload migrators. v3→v4 backfills direction/mechanism fields on `experiment_history` and `last_iteration_results` (`direction_id`, `risk_level`, `failure_mechanism`, `intervention_type`, `changed_sections`); the `dino_similarity` → `dreamsim_similarity` rename is also handled here.

### Reporting
- `report.py` - Thin façade re-exporting `build_report`, `build_all_reports`, `load_report_data`, `ReportData`.
- `report_data.py` - `ReportData` dataclass centralizing data loading (state, iteration logs, manifest, promotion decisions, holdout summary).
- `reporting/render.py` - HTML section renderers: header, summary (score trajectory + hypothesis outcomes + top open problems), trajectories, iteration drilldown, KB, protocol, promotion decisions, holdout. Vision feedback XML tags parsed into styled verdict cards. Prompt diffs via `difflib.unified_diff`.
- `reporting/charts.py` - Plotly chart builders (composite trajectory + per-metric subplots); lazy-imports Plotly.
- `reporting/document.py` - HTML5 document assembly; CSS lazy-loaded via `functools.cache`. `--offline` embeds Plotly JS inline.
- `reporting/report.css` - Editorial dark-theme design system (CSS custom properties, responsive at 880px).

## Directory Conventions

- `src/art_style_search/` - All source code
- `reference_images/` - User-provided reference art (not committed)
- `runs/` - All run data, each run in its own subdirectory (not committed):
  - `runs/<name>/outputs/` - Generated images by iteration/experiment
  - `runs/<name>/logs/` - Iteration logs (`iter_NNN_branch_M.json`), captions cache, style profile, `best_prompt.txt`
  - `runs/<name>/state.json` - Resume state
  - `runs/<name>/report.html` - Post-run HTML report (generated on demand via `report` subcommand)

## Evaluation Metrics

Each metric compares a generated image against its specific paired original (not all references). Base weights sum to 1.00:

- **DreamSim** (34%): Human-aligned perceptual similarity capturing semantic content, layout, color, pose (replaces DINO + LPIPS). Higher = better.
- **Color histogram** (17%): HSV histogram intersection. Higher = better.
- **Vision subject** (10%): Per-image Gemini ternary comparison of subject fidelity. Subject fidelity was up-weighted alongside the new `subject_anchor` section; paired with a floor penalty (see below). Higher = better.
- **SSIM** (10%): Structural similarity index for pixel-level comparison. Higher = better.
- **Vision style** (8%): Per-image Gemini ternary comparison of style fidelity (MATCH=1.0, PARTIAL=0.5, MISS=0.0). Higher = better.
- **HPS v2** (7%): Caption-image alignment (normalized: raw / 0.35, clamped to 1.0). Higher = better.
- **LAION Aesthetics** (6%): Aesthetic quality predictor (1-10 scale, normalized /10). Higher = better.
- **Vision composition** (4%): Per-image Gemini ternary comparison of spatial layout. Higher = better.
- **Style consistency** (4%): Jaccard word-overlap of [Art Style] blocks across captions (experiment-level, omitted from `per_image_composite`). Higher = more consistent shared style guidance.

Penalties subtracted from the weighted sum (result floor-clamped to 0.0):
- **Variance penalty** (×0.30): mean of per-image DreamSim and color-histogram std.
- **Completion penalty** (×0.15): `(1 - completion_rate)`.
- **Compliance penalty** (×0.08): `(1 - compliance_components_mean(...))` over topic coverage, marker coverage, section ordering, section balance, subject specificity.
- **Ref-shortfall penalty** (×0.04): `max(requested - actual, 0) / requested`.
- **Subject-floor penalty** (×0.05): active only when `vision_subject < 0.35`; scales linearly toward the floor.

`per_image_composite` (used in `paired_promotion_test`) uses the same base weights minus `_W_STYLE_CON` and applies no penalties — max output 0.96.

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
- Retry and reliability: all Gemini API calls use jittered exponential backoff (`delay * (0.5 + random())`), per-request `asyncio.wait_for` timeouts (180s generation, 90s caption/vision, 120s pairwise), 429 rate-limit detection with 30s base delay, and per-surface circuit breakers (`caption_circuit_breaker`, `generation_circuit_breaker`, `vision_circuit_breaker` in `retry.py`) that pause calls for 60s after 15 consecutive failures. `gemini_circuit_breaker` is a deprecated alias for `caption_circuit_breaker`. Anthropic streaming retries transient errors by exception class (`httpx.RemoteProtocolError`, `httpcore.ReadError`, `ConnectionResetError`, `BrokenPipeError`) first, with string-based fallback for unrecognized transient errors.
- Generated images are disk-cached: `generate_single` skips the API call if `output_path` already exists with >0 bytes. This makes crash+resume skip the entire generation phase.
- Caption+generation runs as a per-image pipeline in `_caption_and_generate` — each image's caption→generate is a single chained task so generation starts as soon as each individual caption completes (no serial boundary between phases). Failures are per-image via `return_exceptions=True`.
- Exploration mechanism: on even plateau counts, the loop adopts the second-best experiment via `_apply_exploration_result` (updates `current_template` only — does NOT touch `best_template`, `best_metrics`, or `global_best_*`) and resets `plateau_counter` to 1 to give exploration runway. `best_template` must stay in sync with `best_metrics` so that rigorous-mode confirmatory validation replicates the correct incumbent. `_apply_best_result` is used only for genuine improvements and guards `global_best_*` with a score comparison so it never regresses.
- `ConvergenceReason.REASONING_STOP` (the reasoning model emitted `[CONVERGED]` or the parser returned zero proposals) is gated behind `_should_honor_stop()` in `workflow/policy.py`, which requires ALL three: (1) iteration ≥ `max_iterations * _MIN_ITER_FRACTION_FOR_STOP` (0.5), (2) `plateau_counter ≥ max(plateau_window - 1, 2)`, (3) every `CATEGORY_SYNONYMS` key has at least one hypothesis in the KB. If the guard rejects, the `should_stop` flag is dropped, the refinement is kept as a real proposal, and the zero-proposals path returns `([], False)` so the outer loop bumps the plateau counter naturally. The softened `[CONVERGED]` instruction in `prompt/experiments.py` also pushes the reasoning model to self-check the same conditions before emitting the token.
- Style analysis cache is shared across runs via `runs/.cache/style_{hash}.json` keyed by sorted reference paths+mtimes. New runs with the same ref images skip the 3 API calls entirely. The cache is also copied into each run's `log_dir` for provenance.
- Resume safety: `run()` checks `state.converged` after `load_state` and returns early if the previous run already converged. The `continue` path (empty experiment results) advances `state.iteration` and calls `save_state` before continuing so that a crash+resume does not replay the same doomed iteration forever.
- Meta-prompt is 1200-1800 words with 8-15 sections (4-8 sentences each). The FIRST section must be `style_foundation` (fixed style rules from StyleProfile) and the SECOND section must be `subject_anchor` (detailed subject-fidelity instructions). The first two caption output labels must be `[Art Style]` and `[Subject]`. These required anchors are enforced by `_REQUIRED_SECTION_ANCHORS` / `_REQUIRED_CAPTION_ANCHORS` tables in `prompt/_parse.py`.
- Style consistency is measured via Jaccard word-overlap of [Art Style] blocks across captions and included in composite_score (4% weight).
- Captions have labeled output sections (e.g. `[Art Style]`, `[Subject]`, `[Color Palette]`). Beyond the two required anchors, the set of section names, their ordering, and the caption length target are all part of the optimization surface — the reasoning model experiments with these via `caption_sections` and `caption_length_target` on `PromptTemplate`. Before handing the caption to the generator, `build_generation_prompt` in `caption_sections.py` reorders the blocks so `[Subject]` leads, then `[Art Style]` as the style anchor, then the remaining sections — this subject-first rendering is the key to reliable subject fidelity from text-to-image generation.
- Caption compliance checking verifies keyword coverage (meta-prompt section topics), labeled section marker presence (`[Section Name]` in caption text), section ordering (markers appear in expected order), section length balance (no single section dominates >50% of words), and subject specificity (when `[Subject]` is present: min word count, facet coverage via `_SUBJECT_FACET_KEYWORDS`, and generic-vs-specific-word ratio). All five components flow through `compliance_components_mean` in `types.py` with a divisor derived from `len(fields(CaptionComplianceStats))`. `compute_caption_compliance` parses each caption once (via `parse_labeled_sections`) and returns `(stats, prose)` in a single pass so callers don't double-scan.
- Experiment section-count rules depend on `risk_level`: targeted experiments (`risk_level="targeted"`, the default) change exactly 1 section for clean attribution; bold experiments (`risk_level="bold"`) may change up to 3 related sections for broader mechanism tests. Each proposal declares both `changed_section` (singular, primary) and `changed_sections` (list) in the JSON contract — `_normalize_changed_sections` in `prompt/json_contracts.py` backfills whichever is missing and enforces `changed_sections[0] == changed_section`. `validate_template` in `prompt/_parse.py` enforces the per-risk-level count bounds.
- The worst discarded experiment's details (hypothesis, caption, vision feedback) are shown to Claude for negative learning — helps avoid repeating failures.
- `propose_experiments` system prompt includes PE2-inspired "Optimization dynamics" section with three principles: **Momentum** (double down on confirmed KB insights), **Step size** (adapt change magnitude to current composite score regime: LOW <0.35 = bold, MODERATE 0.35-0.50 = targeted, HIGH >0.50 = surgical), **Diversity pressure** (deprioritize categories with 3+ rejections and no confirmed insights). The user message also includes the current composite score with regime label so Claude can calibrate.
- Hypothesis variability is enforced at 4 layers: (1) **Prompt-level** — the system prompt asks for 3 mechanism-tagged directions, and each proposal declares `target_category`, `failure_mechanism`, and `intervention_type`; multiple directions may touch the same category as long as mechanisms or interventions differ; (2) **Post-parse dedup** — `enforce_hypothesis_diversity()` in `prompt/experiments.py` keys on the triple `(target_category, failure_mechanism, intervention_type)` (falling back to `classify_hypothesis()` keyword matching when `target_category` is empty) and drops duplicates; (3) **KB-guided targeting** — `suggest_target_categories()` (in `prompt/_format.py`) ranks categories by improvement potential (unexplored=1.0, partial success=0.7, diminishing returns=0.1) and injects the ranked list into the user message; (4) **Portfolio selection** — `select_experiment_portfolio()` in `prompt/experiments.py` trims the raw batch to `num_branches` by taking one targeted proposal per direction (preserving D1→D2→D3 priority order) and then filling remaining slots with bolds. Both the dedup filter and the portfolio selector are called in `workflow/iteration_proposals.py` after `propose_experiments()` returns.
- Number of fixed reference images is configurable via `--num-fixed-refs` (default 20). On resume, existing refs from state.json are used regardless.
- Independent review loop (CycleResearcher-inspired): in parallel with synthesis reasoning (Phase 3.9), `review_iteration` in `prompt/review.py` sends all experiment results to the reasoning model as a skeptical reviewer. The reviewer assesses each experiment as SIGNAL/NOISE/MIXED, identifies which metric movements are real vs noise, and provides strategic guidance. The `strategic_guidance` is stored in `LoopState.review_feedback` (persisted in state.json) and prepended to `roundtrip_fb` at the start of the next iteration so `propose_experiments` can incorporate the reviewer's recommendations.
- Pairwise experiment comparison (SPO-inspired): in parallel with synthesis reasoning (Phase 3.7), `pairwise_compare_experiments` in `evaluate.py` sends sampled image trios (original, set A reproduction, set B reproduction) from the top 2 experiments to the configured comparison provider (Gemini or xAI) for a head-to-head comparison. Returns a winner (A/B/TIE) with rationale. The rationale is stored in `LoopState.pairwise_feedback` (persisted in state.json) and prepended to `vision_fb` at the start of the next iteration so `propose_experiments` can learn which experiment's approach was visually superior. `build_ref_gen_pairs` in `utils.py` reconstructs (ref, gen) pairs from `IterationResult` by parsing the caption index from generated filenames; both `workflow/iteration_execution.py` (for pairwise comparison) and `report.py` (for the image grid) use it.
- HTML report metric trajectories must use `composite_score` only — `adaptive_composite_score` is min-max normalized within a single batch and is meaningless across iterations. Within an iteration's experiment table, `adaptive_composite_score` is fine (and useful for ranking) because it's recomputed per-batch.
- Scientific rigor is controlled by `--protocol {classic, rigorous}`. Classic mode preserves current behavior exactly. Rigorous mode enables: (1) information barrier — 70/30 split via `math.ceil(0.7 * len(fixed_refs))` with a floor of 2 silent refs (yields 14 feedback / 6 silent for the default 20-ref run; see `workflow/context.py:_split_information_barrier`), (2) confirmatory replication (top-2 + incumbent × 3 replicates), (3) Wilcoxon signed-rank statistical test for promotion (p < 0.10).
- `--seed` provides deterministic reference selection via a `random.Random(seed)` instance on `RunContext`. Retry jitter in `utils.py` stays unseeded (timing, not experiment logic). Seed is persisted in `LoopState.seed` and `run_manifest.json`.
- Information barrier: `_split_information_barrier()` in `workflow/context.py` splits `fixed_refs` into `feedback_refs` (shown to reasoning model in per-image feedback) and `silent_refs` (evaluated but hidden). All images go through captioning + generation + evaluation. Only feedback contexts (`workflow.iteration_context._build_iteration_context`, `workflow.iteration_execution._run_pairwise_comparison`) are filtered. Scoring uses ALL images.
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
