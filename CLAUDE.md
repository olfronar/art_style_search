# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Self-improving loop that optimizes a meta-prompt for art-style capture and image recreation. The meta-prompt (2000-8000 words) instructs a captioner (Gemini Pro) how to describe images with labeled style-guidance sections so that: (a) a generator (Gemini Flash) can recreate them from the captions, and (b) the style guidance can be reused to generate new art in the same style. Inspired by karpathy/autoresearch.

## Architecture

- **Claude (Anthropic) / GPT / GLM / local model**: Brain/optimizer. Analyzes reproduction quality, refines the meta-prompt. Supports `--reasoning-provider local` with `--reasoning-base-url` for OpenAI-compatible servers (vLLM, SGLang, Ollama).
- **Gemini 3.1 Pro Preview**: Captioner. Describes reference images using the meta-prompt instructions.
- **Gemini 3.1 Flash Image Preview**: Generator. Produces images from per-image captions.
- **Zero-step bootstrap captioner** (optional): the one-time captioning of the 20 fixed references *and* the parallel visual analysis of those images that together seed `StyleProfile` + the initial meta-prompt can both be routed through Anthropic Claude via `--bootstrap-captioner claude` (default `gemini`). Per-iteration captioning stays on Gemini regardless of this flag.

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
uv run python -m art_style_search verify-metrics     # Sanity-check metrics at both extremes — ref vs. ref (identity, "1" baseline) AND ref vs. black square ("0" baseline)
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
- `--reasoning-effort` - Reasoning-model effort level (low/medium/high, default medium). Anthropic: low→thinking disabled, medium→adaptive, high→enabled with 16k budget. OpenAI: mapped to `reasoning.effort`. Z.AI/xAI/local: dropped with a one-time warning.
- `--bootstrap-captioner` - Zero-step provider (`gemini` default, or `claude`). Controls BOTH the one-time 20-ref captioning AND the parallel visual analysis of those images that feeds `_reasoning_compile`. Per-iteration captioning and the vision judge are unaffected. `claude` requires `ANTHROPIC_API_KEY` and is enforced at config parse time.
- `--bootstrap-caption-model` - Anthropic model used for both zero-step surfaces when `--bootstrap-captioner claude` (default `claude-opus-4-7`). `--caption-thinking-level` maps to Anthropic `reasoning_effort` for this path (MINIMAL/LOW → low, MEDIUM → medium, HIGH → high).

## Module Map

Top-level modules in `src/art_style_search/`:

- `__main__.py` - Entry point with `list`, `clean`, `report` subcommands; default runs `loop.run(config)`.
- `loop.py` - Main orchestration loop: zero-step setup, per-iteration phase calls (`_build_iteration_context` → `_propose_iteration_experiments` → `_run_experiments_parallel` → `_score_and_rank` → `asyncio.gather` over synthesis/pairwise/review → `_run_synthesis_experiment` → `_run_replicate_gate` (A1, active when `--replicates ≥ 2`) → `_apply_iteration_result` → `append_canon_edit_ledger` → `_update_knowledge_base_for_iteration` → `_record_iteration_state`), plus plateau-convergence check. Phase helpers live in `workflow/`.
- `config.py` - CLI argument parsing → `Config` dataclass. Includes `--seed`, `--protocol {short, classic}`, `--reasoning-provider`, `--comparison-provider`, etc.
- `types.py` / `contracts.py` / `taxonomy.py` - Shared dataclasses, transient workflow contracts, `CATEGORY_SYNONYMS` map. See `src/art_style_search/CLAUDE.md` for the full data shape notes.
- `analyze.py`, `caption.py`, `caption_sections.py`, `generate.py`, `experiment.py`, `evaluate.py`, `scoring.py`, `knowledge.py`, `models.py` - Pipeline: caption → generate → evaluate. See `src/art_style_search/CLAUDE.md`.
- `reasoning_client.py`, `retry.py`, `media.py`, `utils.py` - Providers, retry, shared helpers. See `src/art_style_search/CLAUDE.md`.
- `runs.py`, `state.py`, `state_codec.py`, `state_migrations.py` - Persistence + run directory management.
- `report.py`, `report_data.py` - Reporting façade + data loader.
- `verify_metrics.py` - `verify-metrics` subcommand: runs the full evaluation stack on a randomly-chosen reference from an existing run in two cases — **identity** (ref vs. ref, "1" baseline: paired metrics hit max, vision judge returns MATCH) and **zero** (ref vs. same-size black square, "0" baseline: vision judge returns MISS, paired metric values reported as INFO since floors vary with content). Prints a per-case sanity report plus run-level caption-compliance rows. Exit 0 when both cases' hard invariants hold (identity paired >= tolerance, identity vision == MATCH, zero vision == MISS), 1 on failure, 2 on setup errors.

Sub-packages have their own CLAUDE.md:

- `src/art_style_search/CLAUDE.md` - Pipeline conventions (caption/evaluate/scoring/knowledge/state/retry).
- `src/art_style_search/prompt/CLAUDE.md` - Reasoning-model interface (propose_experiments, synthesis, review, initial-template brainstorm→rank→expand).
- `src/art_style_search/workflow/CLAUDE.md` - Orchestration package (iteration phases, policy, protocol dispatch).
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
- **HPS v2** (7%): Caption-image alignment (normalized: raw / 0.35, clamped to 1.0). Higher = better.
- **MegaStyle** (8%): Cosine similarity between ref and gen in a SigLIP SoViT-400M embedding fine-tuned on 1.4M style-paired images ([MegaStyle, Gao et al. 2026, arXiv:2604.08364](https://arxiv.org/abs/2604.08364)). Continuous, content-disentangled style-space signal — Spearman ~0 with DreamSim (content) and the ternary vision judge (Gemini) on a 754-pair calibration sweep, so it adds a new independent axis. Higher = better. Entered the composite at 5% funded by demoting `style_consistency` (see below), then promoted 5% → 8% after the homescapes run showed the ternary `vision_style` judge systematically demoting branches with the highest MegaStyle — two nominal "style" signals anti-correlating in practice. MegaStyle is continuous and cheap, `vision_style` is coarse and costly, so MegaStyle is now the primary style-similarity weight with `vision_style` in a secondary regression-alarm role.
- **Vision subject** (7%): Per-image Gemini ternary comparison of subject fidelity. Paired with a floor penalty (see below). Higher = better.
- **LAION Aesthetics** (6%): Aesthetic quality predictor (1-10 scale, normalized /10). Higher = better.
- **SSIM** (6%): Structural similarity index for pixel-level comparison. Higher = better.
- **Vision composition** (4%): Per-image Gemini ternary comparison of spatial layout. Higher = better.
- **Vision style** (3%): Per-image Gemini ternary comparison of style fidelity (MATCH=1.0, PARTIAL=0.5, MISS=0.0). Higher = better. Demoted 6% → 3% when MegaStyle was promoted (see MegaStyle above); at this weight it behaves as a regression alarm alongside `style_consistency`.
- **Style consistency** (3%): Jaccard word-overlap of [Art Style] blocks across captions (experiment-level, omitted from `per_image_composite`). Under the canon-first contract the [Art Style] block is copied verbatim from the meta-prompt's `style_foundation`; this metric is now a small regression alarm for canon-pull-through, demoted from 8% when the MegaStyle metric was added (token-overlap on captions is a poor proxy for image-space style similarity — the 754-pair calibration sweep measured Spearman ≈−0.006). Higher = better.
- **Vision proportions** (3%): Per-image Gemini ternary comparison of head-heights + character archetype. Higher = better.
- **Vision medium** (2%): Per-image Gemini ternary agreement on the rendering medium — described in plain observable vocabulary that matches the specific surface behavior each image exhibits; no letter buckets, no implicit menu. A medium mismatch is a style MISS in the vision judge, so this dim catches that failure explicitly. Higher = better.

Penalties subtracted from the weighted sum (result floor-clamped to 0.0):
- **Variance penalty** (×0.30): mean of per-image DreamSim and color-histogram std.
- **Completion penalty** (×0.15): `(1 - completion_rate)`.
- **Compliance penalty** (×0.08): `(1 - compliance_components_mean(...))` over topic coverage, marker coverage, section ordering, section balance, subject specificity, **style-canon fidelity** (caption `[Art Style]` block reproduces the meta-prompt's `style_foundation` canon — 1.0 = verbatim copy, the desired outcome), and **observation-boilerplate purity** (observation blocks `[Subject]` / `[Color Palette]` / `[Composition]` / `[Lighting & Atmosphere]` stay free of canon text — 1.0 = clean observations, 0.0 = canon pasted into per-image sections).
- **Ref-shortfall penalty** (×0.04): `max(requested - actual, 0) / requested`.
- **Subject-floor penalty** (×0.05): active only when `vision_subject < 0.35`; scales linearly toward the floor.

`per_image_composite` uses the same base weights minus `_W_STYLE_CON` (experiment-level only) and applies no penalties — max output 0.97 (sum of every weight except the 0.03 style-consistency axis).

## Cross-cutting Conventions

- Helpers and constants used by 2+ modules belong in `utils.py` (derive from existing data where possible, e.g. `IMAGE_EXTENSIONS = frozenset(MIME_MAP)`); within a module, extract helpers when the same logic appears in both zero-step and main loop paths (e.g. `_apply_best_result`, `_collect_experiment_results`).
- `IterationResult` alignment invariant: `image_paths[i]`, `per_image_scores[i]`, and `iteration_captions[i]` all refer to the same image. `per_image_scores` is stored in original (generation) order, never sorted. Sorting for feedback display uses local variables only. Generated filenames (`00.png`, `11.png`, …) encode the original `fixed_refs` slot index (see `experiment.py:_caption_and_generate`), so on drops the filename stem diverges from the list position — `int(gen_path.stem)` must never be used as a position index into any of these lists.
- Completion tracking: `AggregatedMetrics.completion_rate` (succeeded/attempted) penalizes experiments with partial failures in `composite_score` (0.15 weight). `IterationResult.n_images_attempted/n_images_succeeded` record generation-phase completeness. Experiments below 50% generation success (`_MIN_COMPLETION_RATE`) are rejected outright.
- Persisted collections (`experiment_history`, etc.) must be bounded — cap and drop oldest entries rather than growing without limit in state.json.
- When removing or renaming a metric, field, or function, update all references across source *and* tests — search the entire codebase, not just the file being changed.
- Scoring: `adaptive_composite_score` ranks experiments against each other (relative); `composite_score` is used for improvement checks against baseline (absolute, same scale, with `improvement_epsilon(baseline)` adaptive threshold to filter generation noise) — never compare values from different scoring functions. `improvement_epsilon(baseline)` clamps `baseline` to `[0, 1]` then returns `max(IMPROVEMENT_EPSILON * (1 - clamped), 0.001)` — threshold shrinks as score climbs with a floor of 0.001 to prevent false-positive improvements at high baselines (and upper clamp prevents negative thresholds if a baseline ever exceeds 1.0).
- Caption cache keys are content-derived (`p{sha256(meta_prompt)[:12]}`) in both protocol modes — identical meta-prompts share cache entries across experiments and iterations.
- Style analysis cache is shared across runs via `runs/.cache/style_{hash}.json` keyed by sorted reference paths+mtimes. New runs with the same ref images skip the 3 API calls entirely. The cache is also copied into each run's `log_dir` for provenance.
- Protocol is controlled by `--protocol {short, classic}` (default `short`). `short` = 3-iter cheap foundation pass; `classic` = 5-iter refinement pass, typically resuming on a prior short run via `--run <name>`. Runtime behavior: paired-replicate gate (A1) activates when `--replicates ≥ 2` regardless of protocol; portfolio category quota (A4) runs for every protocol; **headroom-weighted scoring (A6) is the classic-protocol-only promotion-gate function** — `workflow/policy.py::_promotion_score` dispatches on `state.protocol` and selects `headroom_composite_score` for classic, plain `composite_score` for short. A1 composes with A6 under classic: `_run_replicate_gate` computes per-replicate scores via the same `_promotion_score` dispatch, so dominance + median-epsilon checks see headroom numbers end-to-end. **Diff-based canon editing (A2)** is wired via `prompt/json_contracts.py::validate_expansion_payload(prior_canon=...)` — when the reasoner's expansion payload contains a `canon_ops` array, ops are applied to the incumbent canon (bound by `expand_experiment_sketches` from `current_template.sections[0].value`) before template validation; ops are authoritative over the payload's `style_foundation.value`, with fallback to full rewrite on apply failure. Ops flow onto `RefinementResult` → `ExperimentProposal` → `IterationResult` → `CanonEditLedgerEntry.canon_ops` so the next iteration's reasoner sees its prior edits as structured data. See `src/art_style_search/workflow/CLAUDE.md` for details.
- `--seed` provides deterministic reference selection via a `random.Random(seed)` instance on `RunContext`. Retry jitter in `utils.py` stays unseeded (timing, not experiment logic). Seed is persisted in `LoopState.seed` and `run_manifest.json`.
- Number of fixed reference images is configurable via `--num-fixed-refs` (default 20). On resume, existing refs from state.json are used regardless.

## Shipped vs scaffolded

A feature listed as *shipped* in CLAUDE.md MUST have a production call site — a non-test file under `src/` that invokes its entry point. When documenting a feature, name the entry point in the same sentence (`via workflow/X.py::fn_name`). Scaffolded code that isn't wired yet belongs in a dedicated "Scaffolded (not yet wired)" subsection with a TODO — never mixed into main protocol descriptions.

Guarded by:
- `tests/test_architecture_invariants.py::TestShippedFeaturesHaveProductionCallers` — fails when `apply_canon_ops` or `headroom_composite_score` loses its production caller. Extend `_grep_non_test_call_sites` usage when adding a new shipped feature.
- `tests/test_architecture_invariants.py::TestProtocolDivergesInPromotionGate` — fails when `--protocol classic` and `--protocol short` produce identical promotion scores.
- `tests/test_architecture_invariants.py::TestCodecReflection` — every `dataclasses.fields()` entry on `MetricScores`/`AggregatedMetrics`/`IterationResult` must survive a serialize→deserialize round-trip with distinctive sentinel values. When adding a dataclass field, update the `_*_from_dict` decoder in `state_codec.py`, the migration entry in `state_migrations.py`, and the `tests/conftest.py` factory with a non-default value. The factory meta-check (`test_factory_seeds_every_*_with_non_default`) catches defaulted-on-both-sides bugs where the round-trip passes trivially.

**Protocol-branch rule**: any runtime behavior that depends on `config.protocol` or `state.protocol` MUST be observable from outside the workflow (scoring-function label in `promotion_log.jsonl`, different promotion outcome on a saturated-axes fixture, etc.) — silent internal branches are forbidden.

## Meta-prompt structural contract

- Meta-prompt is 2000-8000 words with 8-20 sections (the reasoning-model target). Validator floor is **1000 words / 5 sections** — a laconic safety net so compact experiments can validate without requiring the reasoner to produce short output. `[Subject]` caption blocks run roughly 800-2000 words (they carry the per-image weight including character proportions); `[Art Style]` targets 400-800 words of generic RULES only, across the 5-slot skeleton below; ancillary caption sections typically 150-400 words. Default `caption_sections` is `['Art Style', 'Subject', 'Color Palette', 'Composition', 'Lighting & Atmosphere']` — the reasoner may add others as experiments but `'Technique'` / `'Textures'` are discouraged since they tend to duplicate `[Art Style]`.
- The FIRST section must be `style_foundation` (fixed style rules from StyleProfile) and the SECOND section must be `subject_anchor` (detailed subject-fidelity instructions). The first two caption output labels must be `[Art Style]` and `[Subject]`. These required anchors are enforced by `_REQUIRED_SECTION_ANCHORS` / `_REQUIRED_CAPTION_ANCHORS` tables in `prompt/_parse.py`.
- `style_foundation.value` **IS the STYLE CANON** — the concrete assertive description of this specific reference art style, written as **third-person declarative assertions** ("This style renders as…", "Edges are…", "The palette pulls from…"). The captioner copies its content verbatim into every caption's `[Art Style]` block AND the image generator reads it as the style descriptor. It covers five facets in order, each 2-4 declarative sentences: (1) **How to Draw** (medium identification in plain observable vocabulary matching the specific surface behavior of the analyzed references — no implicit menu, no letter buckets — plus line/edge policy describing the **finished surface**; drawing procedure, primitives, and fabrication steps are NOT expected content here because they don't help a text-to-image generator — the slot describes how the image *reads*, not how an artist would build it) — carries the literal `How to Draw:` marker enforced by the validator; (2) **Shading & Light** (layer stack base→AO→midtones→rim→specular + edge softness + key-fill-rim direction); (3) **Color Principle** (generic palette families + value + saturation policy; NO actual colors named); (4) **Surface & Texture** (grain + material vocabulary that matches the observed medium; NO specific objects); (5) **Style Invariants** (3-5 generative MUST/NEVER rules every image in this style obeys). The canon must be **content, not methodology** — the validator (`_CANON_METHODOLOGY_PATTERNS` in `prompt/_parse.py`) rejects five drift shapes in the value: audit scaffolding (`SLOT N`, `- [ ]`, `MANDATORY`), specific imperative phrases (`Write the block/canon`, `Declare the medium`, `Begin the block`), sentence-initial imperative verbs addressed to a reader (`Begin`, `Write`, `Declare`, `Target`, `Avoid`, `Follow`, `Before`, `After`, `Do not`, `Use`, `Include` — `never`/`always`/`ensure` deliberately excluded so MUST/NEVER invariants pass), direct address + meta-references (`the caption(er)`, `this block/section/slot/canon/skeleton`, `each slot`), and reproduction vocabulary + word-count targets (`verbatim`, `paraphrase`, `REUSABLE DNA`, `N-M words`). Canon is the **primary optimization target**: hypotheses and experiments propose canon edits, evaluations measure whether those edits shrink the gap between generated and reference images, and synthesis merges canon wins. Two preambles govern the whole block: the **observations-vs-rules rule** (per-image observations — specific body parts, named objects, proper nouns, actual colors, pose details — belong in `[Subject]`, `[Color Palette]`, `[Composition]`, or `[Lighting & Atmosphere]`, NEVER in `[Art Style]`; a sentence inside `[Art Style]` is well-formed only if it would still be true of a DIFFERENT image in the same style) and the **anti-name preamble** (forbid genre labels like `3D CGI of X`, `cel-shaded anime`, `{Artist}-style`, `watercolor illustration`). `subject_anchor.value` must contain a `Proportions:` sub-block with at least one archetype token (`heads tall`, `chibi`, `heroic`, `realistic-adult`, `elongated`, …). The `how to draw:` and `proportions:` markers + archetype tokens + methodology patterns are enforced by `_check_anchor_sub_blocks` in `prompt/_parse.py`; the other four skeleton slots, the observations-vs-rules rule, and the anti-name preamble are soft-guided via `CAPTION_SYSTEM` + all meta-prompt seeders.
- Captions have labeled output sections (e.g. `[Art Style]`, `[Subject]`, `[Color Palette]`). Beyond the two required anchors, the set of section names, their ordering, and the caption length target are all part of the optimization surface — the reasoning model experiments with these via `caption_sections` and `caption_length_target` on `PromptTemplate`. Before handing the caption to the generator, `build_generation_prompt` in `caption_sections.py` reorders the blocks so `[Subject]` leads, then `[Art Style]` as the style anchor, then the remaining sections. When the caption's `[Art Style]` block is missing or below the 100-word floor, `build_generation_prompt` falls back to injecting `style_foundation.value` (the canon) directly so the generator always sees the canonical style assertions.
- `PromptTemplate.render()` emits **section-delimited markdown** (not flat prose): `## <section.name>` header, italic `_<description>_` line, then the `value` body — per section. Trailing `## Negative Prompt`, `## Caption Sections (in order)`, `## Caption Length Target` blocks preserve the structural surface. The captioner sees this structure as its user turn, and the cache key `p{sha256(meta_prompt)[:12]}` hashes it, so structural changes to caption_sections/length also invalidate stale cache entries.
- `CAPTION_SYSTEM` (`caption.py`) additionally enforces an *output-format discipline* — labels are plain `[Section]`, never markdown-bolded or wrapped in backticks — and an unconditional **forbidden-terms list** (`cartoon`, `stylised`, `beautiful`, `epic`, `cinematic`, …; no movement-naming exception). The one-line citation rule ("cite upstream rule names by reference; don't restate rule content") replaces the prior enumerated audit-block discipline.
- Style consistency is measured via Jaccard word-overlap of [Art Style] blocks across captions and included in `composite_score` at 3% weight. Canon pull-through is measured separately via two directional metrics in the compliance penalty: `compute_canon_fidelity` — **longest-common-substring ratio** (char-level via `difflib.SequenceMatcher.find_longest_match` with whitespace-normalized inputs; score = longest contiguous match length / len(canon)) between the caption's `[Art Style]` block and the meta-prompt's `style_foundation`. 1.0 = canon copied verbatim (the desired state); paraphrase breaks character-level contiguity so it scores <0.2; scrambled vocabulary that happens to share tokens can't fake a high score because no single contiguous run is long. Returns neutral 1.0 when canon <200 chars or caption `[Art Style]` block <50 chars. `compute_observation_boilerplate_purity` uses **asymmetric token-trigram containment** `|A∩B|/|A|` (fraction of observation-block trigrams present in the canon) — NOT Jaccard: the shape matches contamination detection (we want to know what fraction of the observation block came from the canon, independent of canon length): 1.0 = observations stay clean, 0.0 = canon pasted into `[Subject]`/`[Color Palette]`/etc. `extract_style_canon` in `evaluate.py` pulls `style_foundation.value` from a rendered meta-prompt for the metric.
- Vision judge style-gap observations: the per-pair Gemini vision comparison emits a `<style_gap>` block (2-4 sentences of concrete canon-actionable deltas between reference and generated — parsed by `STYLE_GAP_PATTERN` in `evaluate.py` and threaded onto `VisionScores.style_gap` / `MetricScores.style_gap`). `aggregate_style_gap_notes` in `knowledge.py` deduplicates per-experiment notes via Jaccard similarity; `append_kb_style_gap_observations` feeds them into `KnowledgeBase.style_gap_observations` (bounded ring buffer, 20 entries). The reasoner sees this buffer in its user-message context via `format_knowledge_base`, and is instructed to name which observation each proposed canon edit addresses.
- Medium identification discipline: the captioner describes the rendering medium of every image in plain, observable vocabulary that matches the specific surface behavior each image exhibits. No letter buckets, no implicit menu, no enumerated example list — observation drives vocabulary, not the other way around. The analyzer, captioner, and vision judge all use free-form medium language; the constraint is that technique words stay self-consistent with the surface just named and don't mix vocabulary across incompatible media. Directive-only framing is enforced in `_GEMINI_ANALYSIS_PROMPT` (`analyze.py`), `CAPTION_SYSTEM` (`caption.py`), and `_VISION_SYSTEM` (`evaluate.py`); the vision judge's tier anchors (MATCH/PARTIAL/MISS across style/subject/composition/medium/proportions) are media-agnostic — tiers describe what varies, not specific media vocabulary. `CATEGORY_SYNONYMS` (`taxonomy.py`) carries `rendering_dimensionality` and `proportions` categories so the optimizer can target these failures directly.

## Code Style

- Ruff handles linting and formatting (config in pyproject.toml)
- Line length: 120
- Format-on-edit hook is active — do not manually run ruff after edits

## Diagnostics

- Pyright is the type checker. Config at `pyrightconfig.json` (basic mode, pinned to the uv-managed `.venv`). Run with `uv run --with pyright pyright [paths]`.
- **Pyright must be clean project-wide** — `uv run --with pyright pyright` must report `0 errors, 0 warnings, 0 informations` before any change is considered done. This is stricter than "clean on files you touched": cross-file refactors and shared-type changes can introduce errors in files you never opened, and those must be fixed too. The in-editor "new-diagnostics" harness sometimes surfaces stale/cached warnings that the CLI disagrees with — when they conflict, the CLI is authoritative; rerun `uv run --with pyright pyright` to confirm before believing any warning.
- **Fix diagnostic issues in any file you touch** — whether the error is from your change or pre-existed. "Pre-existing" is not a reason to leave a file with errors after editing it. If a pre-existing error in a file you're editing is out of scope for the current task, fix it anyway (same edit pass) or explicitly flag it to the user before moving on. This includes: unresolved imports, type mismatches, undefined-variable warnings, and unused-import errors.
- Acceptable exceptions: third-party libraries without stubs (guard with `useLibraryCodeForTypes` — already configured) and `reportUnusedVariable` on intentional-unused params (rename to `_name` or use `# noqa` with a reason).
- Ruff lint + format (`uv run ruff check .` / `uv run ruff format .`) must also be clean before considering work done.
