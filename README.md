# Art Style Search

Self-improving loop that finds the best prompt to define and follow an art style from reference images. A meta-prompt instructs a captioner (Gemini Pro) how to describe images so a generator (Gemini Flash) can recreate them from the captions. A reasoning model (Claude, GLM-5.1, GPT-5.4, or Grok 4.20 — swappable via `--reasoning-provider`) optimizes the meta-prompt through hypothesis-driven experiments, and image comparison can run on either Gemini or Grok via `--comparison-provider`.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## How It Works

```
Reference Image + Meta-Prompt
        |
        v
  Gemini Pro (captioner)
        |
        v
   Detailed Caption
        |
        v
  Gemini Flash (generator)
        |
        v
   Generated Image
        |
        v
  Compare with Original
  (DreamSim, Color, SSIM, HPS, Aesthetics, Gemini ternary vision, caption compliance)
        |
        v
  Reasoning model (optimizer)
  Refines the meta-prompt
```

The meta-prompt is the only thing being optimized. It tells the captioner *how* to describe images -- what visual details to capture, how precise to be about colors, technique, characters, composition, etc. Better meta-prompts produce captions that lead to more faithful recreations.

### Optimization Loop

0. **Zero-step**: Fix the reference images. Caption them, run parallel style analysis (Gemini vision + reasoning model) to build a `StyleProfile` and N diverse initial meta-prompts.
1. **Propose**: the reasoning model emits a single JSON batch of 8-12 raw hypotheses grouped into 3 mechanism-tagged directions (`D1`/`D2`/`D3`), each carrying 1 targeted proposal (one section changed) + 1-3 bold proposals (up to 3 related sections). The workflow dedups duplicates on `(category, failure_mechanism, intervention_type)` and selects up to `--num-branches` experiments to evaluate (one targeted per direction first, then bolds fill remaining slots).
2. **Run**: each experiment runs in parallel -- caption all references with the proposed meta-prompt, generate images from captions, evaluate against originals.
3. **Rank & synthesize**: experiments are ranked by an adaptive composite score; the top 2-3 are merged into a synthesized template (picks the best section from each) even if none individually beat the baseline.
4. **Pairwise vision comparison** (SPO-inspired): the top two experiments' reproductions are sent to Gemini vision for a head-to-head verdict, and the rationale is fed back into the next iteration.
5. **Independent review** (CycleResearcher-inspired): a skeptical reviewer assesses every experiment as SIGNAL/NOISE/MIXED and writes strategic guidance that is prepended to the next iteration's proposal prompt.
6. **Apply**: best experiment becomes the new baseline (or, on even plateau counts, the second-best is adopted as an exploration move). All results feed a shared Knowledge Base that tracks hypothesis chains, per-category progress, confirmed insights, rejected approaches, and open problems.
7. **Repeat** until convergence (max iterations, plateau window, or the reasoning model signals stop).

## Cost & resources

Running this loop is not free. Know the order-of-magnitude before you start:

- **API calls**. A default run is `--max-iterations 10 × --num-branches 9 × --num-fixed-refs 20` -- on the order of 1800 Gemini Pro captions, 1800 Gemini Flash generations, 1800 image-comparison calls (Gemini by default, optionally Grok), and 30-50 reasoning-model calls (Claude, GLM, GPT, or Grok). Expect several US dollars per full run at current 2026 prices. Costs scale roughly linearly with `max_iterations × num_branches × num_fixed_refs`.
- **First-run ML model downloads**. The first invocation pulls ~2 GB of weights from Hugging Face Hub: DreamSim `dino_vitb16` (~870 MB), LAION-Aesthetics CLIP-L, and HPSv2 CLIP-H. These are cached under `~/.cache/huggingface/` plus the local package caches used by DreamSim/OpenCLIP.
- **GPU is optional**. CPU works but is slow. Apple Silicon uses MPS automatically. NVIDIA CUDA users have to pick a matching `torch` wheel (see Troubleshooting).
- **Smoke-test recipe** (~1% of the cost of a default run):

  ```bash
  uv run python -m art_style_search \
    --max-iterations 1 --num-branches 1 --raw-proposals 8 --num-fixed-refs 3 \
    --run smoke-test --new
  ```

  This runs one iteration with one experiment against three images -- enough to validate the full pipeline end-to-end (downloads, captioning, generation, evaluation, state persistence) without burning budget.

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Google API key](https://aistudio.google.com/apikey) (for Gemini models -- always required)
- One of:
  - [Anthropic API key](https://console.anthropic.com/) (for Claude -- default)
  - [Z.AI API key](https://z.ai/) (for GLM-5.1 -- alternative)
  - [OpenAI API key](https://platform.openai.com/) (for GPT-5.4 -- alternative)
  - [xAI API key](https://console.x.ai/) (for Grok 4.20 reasoning and/or comparison)

## Quick Start

```bash
# Install dependencies
uv sync

# Configure API keys
cp .env.sample .env
# Edit .env with your keys

# Add reference images
# Drop at least 20 images of the target art style in reference_images/,
# or pass --num-fixed-refs N to match however many you have (minimum 3 for the smoke test).

# Run the optimization loop (creates runs/run_001/)
uv run python -m art_style_search

# Resume or create a named run
uv run python -m art_style_search --run my-experiment

# View all options
uv run python -m art_style_search --help

# List all runs with status
uv run python -m art_style_search list

# Clean a specific run or all runs
uv run python -m art_style_search clean --run run_001
uv run python -m art_style_search clean --all
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--reference-dir` | `reference_images` | Directory containing reference art |
| `--runs-dir` | `runs` | Base directory for all runs |
| `--run` | auto | Run name (auto-incremented if omitted) |
| `--new` | | Force new run (error if name exists) |
| `--max-iterations` | `10` | Maximum optimization iterations |
| `--plateau-window` | `5` | Iterations without improvement before stop |
| `--num-branches` | `5` | Parallel experiments per iteration (portfolio size after selection) |
| `--raw-proposals` | `9` | Raw proposals per iteration before portfolio selection (range 8-12) |
| `--num-fixed-refs` | `20` | Fixed reference images for optimization |
| `--protocol` | `classic` | `classic` (default) or `rigorous` (info barrier + replication + statistical testing) |
| `--seed` | random | RNG seed for reproducible reference selection |
| `--aspect-ratio` | `1:1` | Aspect ratio for generated images |
| `--caption-model` | `gemini-3.1-pro-preview` | Gemini model for captioning |
| `--generator-model` | `gemini-3.1-flash-image-preview` | Gemini model for generation |
| `--reasoning-provider` | `anthropic` | Reasoning provider: `anthropic`, `zai`, `openai`, `xai`, or `local` |
| `--reasoning-model` | auto | Model name (default: `claude-sonnet-4-6` / `glm-5.1` / `gpt-5.4` / `grok-4.20-reasoning-latest`) |
| `--comparison-provider` | `gemini` | Image comparison provider: `gemini` or `xai` |
| `--comparison-model` | auto | Comparison model name (default: caption model for `gemini`, `grok-4.20-reasoning-latest` for `xai`) |
| `--reasoning-base-url` | none | Required with `--reasoning-provider local`; base URL for an OpenAI-compatible server |
| `--gemini-concurrency` | `50` | Max concurrent Gemini API calls |
| `--eval-concurrency` | `4` | Max concurrent eval threads |

## Troubleshooting

- **`torch` wheel doesn't match CUDA**. `uv sync` pulls the CPU wheel by default. NVIDIA CUDA users need to override the index: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` (pick the channel that matches your CUDA version).
- **Apple Silicon**. Works out of the box -- the code detects MPS automatically via `torch.backends.mps.is_available()` and uses it for DreamSim / HPS / aesthetics inference.
- **Hugging Face download failures on first run**. The first invocation pulls ~2 GB of weights. If downloads fail with rate-limit errors, rerun once the limit resets. If the cache directory isn't writable, set `HF_HOME=/path/to/writable/cache` before running.
- **Missing API keys**. The CLI refuses to start and tells you exactly which env var to set. Mapping: `GOOGLE_API_KEY` (always required, Gemini captioning/generation and zero-step analysis), `ANTHROPIC_API_KEY` (for `--reasoning-provider anthropic`), `ZAI_API_KEY` (for `zai`), `OPENAI_API_KEY` (for `openai`), `XAI_API_KEY` (for `--reasoning-provider xai` and/or `--comparison-provider xai`).
- **Using a local reasoning model**. `--reasoning-provider local` skips third-party reasoning API keys, but it does require both `--reasoning-model` and `--reasoning-base-url`, for example `--reasoning-base-url http://localhost:8000/v1`.
- **Empty `reference_images/`**. The loop raises `FileNotFoundError` with an actionable message. Drop at least `--num-fixed-refs` images of a supported type (see `IMAGE_EXTENSIONS` in `utils.py`).
- **`KeyError: 'branches'` when resuming**. Your `state.json` predates the branch-based → shared-KB refactor. Delete the old state and start a new run with `--new`.

## Evaluation Metrics

Each metric compares a generated image against its specific paired original; weights sum to 1.00:

| Metric | Weight | Measures | Better |
|--------|--------|----------|--------|
| **DreamSim** | 34% | Human-aligned perceptual similarity | Higher |
| **Color histogram** | 17% | HSV histogram intersection | Higher |
| **Vision (subject)** | 10% | Gemini ternary subject fidelity | Higher |
| **SSIM** | 10% | Structural similarity index | Higher |
| **Vision (style)** | 8% | Gemini ternary style fidelity | Higher |
| **HPS v2** | 7% | Caption-image alignment (normalized / 0.35) | Higher |
| **Aesthetics** | 6% | Visual quality (LAION predictor, 1-10) | Higher |
| **Vision (composition)** | 4% | Gemini ternary spatial layout | Higher |
| **Style consistency** | 4% | Jaccard overlap of [Art Style] blocks | Higher |

Additional penalties (subtracted from the weighted sum, floor-clamped to 0):

- **Variance penalty** (×0.30): mean of per-image DreamSim and color-histogram std — punishes inconsistent reproduction across images.
- **Completion penalty** (×0.15): `1 - completion_rate`, punishes experiments that drop images.
- **Compliance penalty** (×0.08): `1 - mean(compliance rates)` over topic coverage, marker coverage, section ordering, section balance, subject specificity.
- **Ref-shortfall penalty** (×0.04): fraction of requested reference images that were skipped.
- **Subject-floor penalty** (×0.05): triggers only when `vision_subject < 0.35`, scaling linearly to the floor.

`per_image_composite` (used for paired statistical testing in rigorous mode) applies the same base weights minus `style_consistency` (experiment-level) and without any penalties — max output 0.96.

## Scientific Rigor Mode

Pass `--protocol rigorous` (optionally with `--seed 42` for reproducibility) to upgrade from the default exploratory loop to a controlled experimental protocol:

- **Information barrier**: reference images are split into *feedback* (70%, shown to the reasoning model in per-image feedback) and *silent* (30%, evaluated but hidden from the optimizer). Improvements on silent images indicate genuine generalization, not overfitting to feedback signals.
- **Confirmatory replication**: the top-2 candidates and the current incumbent are each regenerated 3 times. Per-image medians replace single-pass scores, reducing generation noise.
- **Statistical testing**: promotion requires a Wilcoxon signed-rank test (p < 0.10, one-sided) on per-image composite-score differences, plus a bootstrap 95% CI on the mean delta. This prevents lucky single generations from being mistaken for real improvements.
- **Run provenance**: a `run_manifest.json` records the seed, git SHA, model versions, reference-image hashes, and platform. On resume in rigorous mode, mismatches abort to prevent silent state drift.
- **Promotion log**: every iteration's promotion decision (score, delta, epsilon, p-value, verdict) is appended to `promotion_log.jsonl` for post-hoc audit.

Classic mode (`--protocol classic`, the default) is unchanged — no replication, no barrier, no statistical gating. The only always-on additions are the manifest and promotion log (observability, zero cost).

Cost: rigorous mode is ~2.8x more expensive per iteration (confirmatory replication of 3 templates x 3 replicates x N images).

## Project Structure

```
src/art_style_search/
  __main__.py             Entry point + list/clean/report subcommands
  loop.py                 Façade re-exporting workflow.run
  workflow/               Internal orchestration (context, iteration phases, policy, services, zero-step)
  prompt/                 Reasoning-model interface (proposals, synthesis, review, JSON contracts, parsing)
  analyze.py              Zero-step: parallel Gemini vision + reasoning-model style analysis
  caption.py              Gemini Pro captioning with disk cache
  caption_sections.py     [Section] parser + subject-first generator prompt builder
  generate.py             Gemini Flash image generation with retry
  experiment.py           Single-experiment caption→generate→evaluate pipeline + replicated evaluation
  evaluate.py             Per-image paired metrics + Gemini vision comparison + caption compliance
  scoring.py              Composite scoring, per-image scoring, Wilcoxon promotion test
  knowledge.py            Knowledge Base maintenance (hypothesis tracking)
  models.py               Lazy-loaded DreamSim/HPS/Aesthetics/SSIM models
  taxonomy.py             CATEGORY_SYNONYMS — canonical hypothesis category synonyms
  contracts.py            Transient workflow dataclasses (Lessons, RefinementResult, ExperimentProposal)
  reasoning_client.py     Provider-agnostic ReasoningClient (Anthropic/OpenAI/xAI/Z.AI/local)
  retry.py                async_retry + rate-limit and circuit-breaker helpers
  media.py                MIME map, IMAGE_EXTENSIONS, xAI data-URL helpers
  runs.py                 Run directory management (isolation, listing, cleanup)
  state.py                Public JSON persistence API (loaders/writers, manifest, promotion log)
  state_codec.py          Low-level encoders/decoders for persisted dataclasses
  state_migrations.py     Schema versions + backward-compat payload migrations
  types.py                Shared dataclasses (MetricScores, AggregatedMetrics, KnowledgeBase, ...)
  config.py               CLI argument parsing → Config
  utils.py                Shared helpers (CATEGORY_SYNONYMS re-export, build_ref_gen_pairs, ...)
  report.py               HTML report façade
  report_data.py          ReportData — centralized data loading for reports
  reporting/              HTML rendering (charts, document assembly, section renderers, CSS)
```

## Development

This project uses [`uv`](https://docs.astral.sh/uv/) as the package and tool manager. Every Python command goes through `uv run`, and external CLI tools should be installed via `uv tool install <name>` so they stay isolated from your system Python.

```bash
uv sync                  # Install/update dependencies
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run pytest tests/     # Run tests
```

Ruff handles both linting and formatting (config in `pyproject.toml`, line length 120).

Optional — enable the [`gitleaks`](https://github.com/gitleaks/gitleaks) secret-scanning hook from `.pre-commit-config.yaml` so accidentally-staged API keys are caught before they reach history:

```bash
uv tool install pre-commit   # Install the pre-commit CLI (isolated via uv)
pre-commit install           # Wire it into .git/hooks/pre-commit
```

For a full git-history secret scan before publishing, run the dedicated GitHub Actions workflow in [`.github/workflows/gitleaks.yml`](.github/workflows/gitleaks.yml) or install the `gitleaks` CLI locally and run `gitleaks git .`.

## Contributing

Contributor workflow lives in [`CONTRIBUTING.md`](CONTRIBUTING.md). Security reporting guidance lives in [`SECURITY.md`](SECURITY.md).

## Acknowledgements

This project stands on the shoulders of:

- [**DreamSim**](https://github.com/ssundaram21/dreamsim) (Fu et al.) -- human-aligned perceptual similarity, the main reproduction metric.
- [**HPS v2**](https://github.com/tgxs002/HPSv2) (Wu et al.) -- human preference score for caption-image alignment.
- [**LAION-Aesthetics-Predictor-v2**](https://laion.ai/blog/laion-aesthetics/) via [`simple-aesthetics-predictor`](https://github.com/shunk031/simple-aesthetics-predictor).
- [**scikit-image**](https://scikit-image.org/) -- SSIM implementation.
- [**Google Gemini**](https://ai.google.dev/) -- captioner, image generator, and default vision comparator.
- [**Anthropic Claude**](https://www.anthropic.com/) / [**Z.AI GLM-5.1**](https://z.ai/) / [**OpenAI GPT**](https://openai.com/) / [**xAI Grok**](https://x.ai/) -- interchangeable reasoning backends, with Grok also available for image comparison.
- [**karpathy/autoresearch**](https://github.com/karpathy/autoresearch) -- the self-improving research-loop pattern that inspired this project.

## License

[MIT](LICENSE) -- see the `LICENSE` file for details.
