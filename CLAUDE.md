# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Self-improving loop that optimizes a meta-prompt for art-style capture and image recreation. The meta-prompt (2000-8000 words) instructs a captioner (Gemini Pro) how to describe images with labeled style-guidance sections so that: (a) a generator (Gemini Flash) can recreate them from the captions, and (b) the style guidance can be reused to generate new art in the same style. Inspired by karpathy/autoresearch.

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
- `--caption-thinking-level` - Gemini Pro captioner extended-thinking level (MINIMAL/LOW/MEDIUM/HIGH, default MINIMAL). MEDIUM materially improves medium-class + proportion precision at 2-3x latency.
- `--generation-thinking-level` - Gemini Flash generator extended-thinking level (MINIMAL/LOW/MEDIUM/HIGH, default MINIMAL).

## Module Map

Top-level modules in `src/art_style_search/`:

- `__main__.py` - Entry point with `list`, `clean`, `report` subcommands; default runs `loop.run(config)`.
- `loop.py` - Thin façade re-exporting `run` from the `workflow` package (kept for a stable import surface).
- `config.py` - CLI argument parsing → `Config` dataclass. Includes `--seed`, `--protocol {classic, rigorous}`, `--reasoning-provider`, `--comparison-provider`, etc.
- `types.py` / `contracts.py` / `taxonomy.py` - Shared dataclasses, transient workflow contracts, `CATEGORY_SYNONYMS` map. See `src/art_style_search/CLAUDE.md` for the full data shape notes.
- `analyze.py`, `caption.py`, `caption_sections.py`, `generate.py`, `experiment.py`, `evaluate.py`, `scoring.py`, `knowledge.py`, `models.py` - Pipeline: caption → generate → evaluate. See `src/art_style_search/CLAUDE.md`.
- `reasoning_client.py`, `retry.py`, `media.py`, `utils.py` - Providers, retry, shared helpers. See `src/art_style_search/CLAUDE.md`.
- `runs.py`, `state.py`, `state_codec.py`, `state_migrations.py` - Persistence + run directory management.
- `report.py`, `report_data.py` - Reporting façade + data loader.

Sub-packages have their own CLAUDE.md:

- `src/art_style_search/CLAUDE.md` - Pipeline conventions (caption/evaluate/scoring/knowledge/state/retry).
- `src/art_style_search/prompt/CLAUDE.md` - Reasoning-model interface (propose_experiments, synthesis, review, initial-template brainstorm→rank→expand).
- `src/art_style_search/workflow/CLAUDE.md` - Orchestration package (iteration phases, policy, information barrier).
- `src/art_style_search/reporting/CLAUDE.md` - HTML report (render, charts, document, CSS).

## Directory Conventions

- `src/art_style_search/` - All source code
- `reference_images/` - User-provided reference art (not committed)
- `runs/` - All run data, each run in its own subdirectory (not committed):
  - `runs/<name>/outputs/` - Generated images by iteration/experiment
  - `runs/<name>/logs/` - Iteration logs (`iter_NNN_branch_M.json`), captions cache, style profile, and the best meta-prompt in three parallel forms: `best_prompt.txt` (legacy flat string), `best_prompt.md` (structured markdown with YAML front-matter — iter/score/seed/sha), `best_prompt.json` (full `PromptTemplate` dataclass dump, re-ingestable as seed)
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
- **Compliance penalty** (×0.08): `(1 - compliance_components_mean(...))` over topic coverage, marker coverage, section ordering, section balance, subject specificity, and style-DNA purity (trigram overlap between caption `[Art Style]` and meta-prompt — 1.0 = captioner paraphrases in own voice, 0.0 = near-verbatim copy).
- **Ref-shortfall penalty** (×0.04): `max(requested - actual, 0) / requested`.
- **Subject-floor penalty** (×0.05): active only when `vision_subject < 0.35`; scales linearly toward the floor.

`per_image_composite` (used in `paired_promotion_test`) uses the same base weights minus `_W_STYLE_CON` and applies no penalties — max output 0.96.

## Cross-cutting Conventions

- Helpers and constants used by 2+ modules belong in `utils.py` (derive from existing data where possible, e.g. `IMAGE_EXTENSIONS = frozenset(MIME_MAP)`); within a module, extract helpers when the same logic appears in both zero-step and main loop paths (e.g. `_apply_best_result`, `_collect_experiment_results`).
- `IterationResult` alignment invariant: `image_paths[i]`, `per_image_scores[i]`, and `iteration_captions[i]` all refer to the same image. `per_image_scores` is stored in original (generation) order, never sorted. Sorting for feedback display uses local variables only. Generated filenames (`00.png`, `11.png`, …) encode the original `fixed_refs` slot index (see `experiment.py:_caption_and_generate`), so on drops the filename stem diverges from the list position — `int(gen_path.stem)` must never be used as a position index into any of these lists.
- Completion tracking: `AggregatedMetrics.completion_rate` (succeeded/attempted) penalizes experiments with partial failures in `composite_score` (0.15 weight). `IterationResult.n_images_attempted/n_images_succeeded` record generation-phase completeness. Experiments below 50% generation success (`_MIN_COMPLETION_RATE`) are rejected outright.
- Persisted collections (`experiment_history`, etc.) must be bounded — cap and drop oldest entries rather than growing without limit in state.json.
- When removing or renaming a metric, field, or function, update all references across source *and* tests — search the entire codebase, not just the file being changed.
- Scoring: `adaptive_composite_score` ranks experiments against each other (relative); `composite_score` is used for improvement checks against baseline (absolute, same scale, with `improvement_epsilon(baseline)` adaptive threshold to filter generation noise) — never compare values from different scoring functions. `improvement_epsilon(baseline)` returns `max(IMPROVEMENT_EPSILON * (1 - max(baseline, 0)), 0.001)` — threshold shrinks as score climbs with a floor of 0.001 to prevent false-positive improvements at high baselines.
- Caption cache keys are content-derived (`p{sha256(meta_prompt)[:12]}`) in both protocol modes — identical meta-prompts share cache entries across experiments and iterations.
- Style analysis cache is shared across runs via `runs/.cache/style_{hash}.json` keyed by sorted reference paths+mtimes. New runs with the same ref images skip the 3 API calls entirely. The cache is also copied into each run's `log_dir` for provenance.
- Scientific rigor is controlled by `--protocol {classic, rigorous}`. Classic preserves current behavior exactly. Rigorous enables: (1) information barrier (70/30 feedback/silent split), (2) confirmatory replication (top-2 + incumbent × 3 replicates), (3) Wilcoxon signed-rank statistical test for promotion (p < 0.10). See `src/art_style_search/workflow/CLAUDE.md` for details.
- `--seed` provides deterministic reference selection via a `random.Random(seed)` instance on `RunContext`. Retry jitter in `utils.py` stays unseeded (timing, not experiment logic). Seed is persisted in `LoopState.seed` and `run_manifest.json`.
- Number of fixed reference images is configurable via `--num-fixed-refs` (default 20). On resume, existing refs from state.json are used regardless.

## Meta-prompt structural contract

- Meta-prompt is 2000-8000 words with 8-20 sections (the reasoning-model target). Validator floor is **1000 words / 5 sections** — a laconic safety net so compact experiments can validate without requiring the reasoner to produce short output. `[Art Style]` and `[Subject]` caption blocks are expected to run roughly 1000-2000 words each; ancillary caption sections typically 150-400 words.
- The FIRST section must be `style_foundation` (fixed style rules from StyleProfile) and the SECOND section must be `subject_anchor` (detailed subject-fidelity instructions). The first two caption output labels must be `[Art Style]` and `[Subject]`. These required anchors are enforced by `_REQUIRED_SECTION_ANCHORS` / `_REQUIRED_CAPTION_ANCHORS` tables in `prompt/_parse.py`.
- `style_foundation.value` must contain a `How to Draw:` sub-block (silhouette primitives, construction order, line policy, shading layers, signature quirk) and `subject_anchor.value` must contain a `Proportions:` sub-block with at least one archetype token (`heads tall`, `chibi`, `heroic`, `realistic-adult`, `elongated`, …). Enforced by `_check_anchor_sub_blocks` in `prompt/_parse.py`. These two sub-blocks operationalize the "way of drawing" + forced proportions discipline the captioner carries into every caption.
- Captions have labeled output sections (e.g. `[Art Style]`, `[Subject]`, `[Color Palette]`). Beyond the two required anchors, the set of section names, their ordering, and the caption length target are all part of the optimization surface — the reasoning model experiments with these via `caption_sections` and `caption_length_target` on `PromptTemplate`. Before handing the caption to the generator, `build_generation_prompt` in `caption_sections.py` reorders the blocks so `[Subject]` leads, then `[Art Style]` as the style anchor, then the remaining sections.
- `PromptTemplate.render()` emits **section-delimited markdown** (not flat prose): `## <section.name>` header, italic `_<description>_` line, then the `value` body — per section. Trailing `## Negative Prompt`, `## Caption Sections (in order)`, `## Caption Length Target` blocks preserve the structural surface. The captioner sees this structure as its user turn, and the cache key `p{sha256(meta_prompt)[:12]}` hashes it, so structural changes to caption_sections/length also invalidate stale cache entries.
- `CAPTION_SYSTEM` (`caption.py`) additionally enforces an *audit-block discipline* — in downstream audit blocks (Medium Class Verification, Ambient Occlusion Map, Rim Light Policy, Specular Allocation, Surface Cleanliness, etc.) the captioner must cite meta-prompt rule names by reference (e.g. "Bevel Rule applies") rather than re-stating rule content — and an *output-format discipline* that forbids markdown bolding of `[Section]` labels.
- Style consistency is measured via Jaccard word-overlap of [Art Style] blocks across captions and included in `composite_score` (4% weight). Prompt-copying is measured separately via `compute_prompt_copying_score` (trigram overlap of the caption's first 500 `[Art Style]` tokens vs the meta-prompt) and contributes to the compliance penalty.
- Medium-class discipline: the captioner classifies every image as one of five classes (A hand-drawn 2D, B vector/flat 2D, C stylized 3D CGI, D photoreal 3D, E mixed/2.5D) and uses class-appropriate vocabulary. Class cues are seeded by `_GEMINI_ANALYSIS_PROMPT` (`analyze.py`) and enforced in `CAPTION_SYSTEM` (`caption.py`); the vision judge uses matching calibration examples in `_VISION_SYSTEM` (`evaluate.py`). `CATEGORY_SYNONYMS` (`taxonomy.py`) carries `rendering_dimensionality` and `proportions` categories so the optimizer can target these failures directly.

## Code Style

- Ruff handles linting and formatting (config in pyproject.toml)
- Line length: 120
- Format-on-edit hook is active — do not manually run ruff after edits
