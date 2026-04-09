# Art Style Search

Self-improving loop that finds the best prompt to define and follow an art style from reference images. A meta-prompt instructs a captioner (Gemini Pro) how to describe images so a generator (Gemini Flash) can recreate them from the captions. A reasoning model (Claude, GLM-5.1, or GPT-5.4 — swappable via `--reasoning-provider`) optimizes the meta-prompt through hypothesis-driven experiments.

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
  (DreamSim, Color, SSIM, HPS, Aesthetics + Gemini vision)
        |
        v
  Reasoning model (optimizer)
  Refines the meta-prompt
```

The meta-prompt is the only thing being optimized. It tells the captioner *how* to describe images -- what visual details to capture, how precise to be about colors, technique, characters, composition, etc. Better meta-prompts produce captions that lead to more faithful recreations.

### Optimization Loop

0. **Zero-step**: Fix the reference images. Caption them, run parallel style analysis (Gemini vision + reasoning model) to build a `StyleProfile` and N diverse initial meta-prompts.
1. **Propose**: the reasoning model proposes N experiments per iteration, each testing a different hypothesis about what to change in the meta-prompt (one section per experiment for clean attribution).
2. **Run**: each experiment runs in parallel -- caption all references with the proposed meta-prompt, generate images from captions, evaluate against originals.
3. **Rank & synthesize**: experiments are ranked by an adaptive composite score; the top 2-3 are merged into a synthesized template (picks the best section from each) even if none individually beat the baseline.
4. **Pairwise vision comparison** (SPO-inspired): the top two experiments' reproductions are sent to Gemini vision for a head-to-head verdict, and the rationale is fed back into the next iteration.
5. **Independent review** (CycleResearcher-inspired): a skeptical reviewer assesses every experiment as SIGNAL/NOISE/MIXED and writes strategic guidance that is prepended to the next iteration's proposal prompt.
6. **Apply**: best experiment becomes the new baseline (or, on even plateau counts, the second-best is adopted as an exploration move). All results feed a shared Knowledge Base that tracks hypothesis chains, per-category progress, confirmed insights, rejected approaches, and open problems.
7. **Repeat** until convergence (max iterations, plateau window, or the reasoning model signals stop).

## Cost & resources

Running this loop is not free. Know the order-of-magnitude before you start:

- **API calls**. A default run is `--max-iterations 20 × --num-branches 5 × --num-fixed-refs 20` -- on the order of 2000 Gemini Pro captions, 2000 Gemini Flash generations, 2000 Gemini vision comparisons, and 60-100 reasoning-model calls (Claude, GLM, or GPT). Expect several US dollars per full run at current 2026 prices. Costs scale roughly linearly with `max_iterations × num_branches × num_fixed_refs`.
- **First-run ML model downloads**. The first invocation pulls ~2 GB of weights from Hugging Face Hub: DreamSim `dino_vitb16` (~870 MB), LAION-Aesthetics CLIP-L, and HPSv2 CLIP-H. These are cached under `~/.cache/huggingface/` and the `dreamsim` / `hpsv2` package cache dirs.
- **GPU is optional**. CPU works but is slow. Apple Silicon uses MPS automatically. NVIDIA CUDA users have to pick a matching `torch` wheel (see Troubleshooting).
- **Smoke-test recipe** (~1% of the cost of a default run):

  ```bash
  uv run python -m art_style_search \
    --max-iterations 1 --num-branches 1 --num-fixed-refs 3 \
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
| `--max-iterations` | `20` | Maximum optimization iterations |
| `--plateau-window` | `5` | Iterations without improvement before stop |
| `--num-branches` | `5` | Parallel experiments per iteration |
| `--num-fixed-refs` | `20` | Fixed reference images for optimization |
| `--aspect-ratio` | `1:1` | Aspect ratio for generated images |
| `--caption-model` | `gemini-3.1-pro-preview` | Gemini model for captioning |
| `--generator-model` | `gemini-3.1-flash-image-preview` | Gemini model for generation |
| `--reasoning-provider` | `anthropic` | Reasoning provider: `anthropic`, `zai`, or `openai` |
| `--reasoning-model` | auto | Model name (default: `claude-sonnet-4-6` / `glm-5.1` / `gpt-5.4`) |
| `--gemini-concurrency` | `50` | Max concurrent Gemini API calls |
| `--eval-concurrency` | `4` | Max concurrent eval threads |

## Troubleshooting

- **`torch` wheel doesn't match CUDA**. `uv sync` pulls the CPU wheel by default. NVIDIA CUDA users need to override the index: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` (pick the channel that matches your CUDA version).
- **Apple Silicon**. Works out of the box -- the code detects MPS automatically via `torch.backends.mps.is_available()` and uses it for DreamSim / HPS / aesthetics inference.
- **Hugging Face download failures on first run**. The first invocation pulls ~2 GB of weights. If downloads fail with rate-limit errors, rerun once the limit resets. If the cache directory isn't writable, set `HF_HOME=/path/to/writable/cache` before running.
- **Missing API keys**. The CLI refuses to start and tells you exactly which env var to set. Mapping: `GOOGLE_API_KEY` (always required, Gemini), `ANTHROPIC_API_KEY` (for `--reasoning-provider anthropic`), `ZAI_API_KEY` (for `zai`), `OPENAI_API_KEY` (for `openai`).
- **Empty `reference_images/`**. The loop raises `FileNotFoundError` with an actionable message. Drop at least `--num-fixed-refs` images of a supported type (see `IMAGE_EXTENSIONS` in `utils.py`).
- **`KeyError: 'branches'` when resuming**. Your `state.json` predates the branch-based → shared-KB refactor. Delete the old state and start a new run with `--new`.

## Evaluation Metrics

Each metric compares a generated image against its specific paired original:

| Metric | Weight | Measures | Better |
|--------|--------|----------|--------|
| **DreamSim** | 40% | Human-aligned perceptual similarity | Higher |
| **Color histogram** | 22% | HSV histogram intersection | Higher |
| **SSIM** | 11% | Structural similarity index | Higher |
| **Aesthetics** | 6% | Visual quality (LAION predictor, 1-10) | Higher |
| **HPS v2** | 5% | Caption-image alignment | Higher |
| **Style consistency** | 4% | Jaccard overlap of [Art Style] blocks | Higher |
| **Vision (style/subject/composition)** | 4% each | Gemini ternary comparison | Higher |

A consistency penalty (30% weight on per-image std of DreamSim and color) penalizes high variance across images.

## Project Structure

```
src/art_style_search/
  __main__.py    Entry point + list/clean subcommands
  loop.py        Experiment-based orchestration loop
  prompt/        Meta-prompt proposal/refinement package (reasoning-model calls)
  analyze.py     Zero-step: parallel Gemini vision + reasoning-model style analysis
  caption.py     Gemini Pro captioning with disk cache
  generate.py    Gemini Flash image generation with retry
  experiment.py  Single-experiment execution pipeline
  evaluate.py    Per-image paired metrics + Gemini vision comparison
  knowledge.py   Knowledge Base maintenance (hypothesis tracking)
  models.py      Lazy-loaded DreamSim/HPS/Aesthetics/SSIM models
  runs.py        Run directory management (isolation, listing, cleanup)
  types.py       Shared dataclasses + KnowledgeBase
  config.py      CLI argument parsing
  state.py       JSON persistence + backward compat migration
  utils.py       Shared helpers (API wrappers, MIME map)
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

## Acknowledgements

This project stands on the shoulders of:

- [**DreamSim**](https://github.com/ssundaram21/dreamsim) (Fu et al.) -- human-aligned perceptual similarity, the main reproduction metric.
- [**HPS v2**](https://github.com/tgxs002/HPSv2) (Wu et al.) -- human preference score for caption-image alignment.
- [**LAION-Aesthetics-Predictor-v2**](https://laion.ai/blog/laion-aesthetics/) via [`simple-aesthetics-predictor`](https://github.com/shunk031/simple-aesthetics-predictor).
- [**scikit-image**](https://scikit-image.org/) -- SSIM implementation.
- [**Google Gemini**](https://ai.google.dev/) -- captioner, image generator, and vision comparator.
- [**Anthropic Claude**](https://www.anthropic.com/) / [**Z.AI GLM-5.1**](https://z.ai/) / [**OpenAI GPT**](https://openai.com/) -- interchangeable reasoning / optimization backends.
- [**karpathy/autoresearch**](https://github.com/karpathy/autoresearch) -- the self-improving research-loop pattern that inspired this project.

## License

[MIT](LICENSE) -- see the `LICENSE` file for details.
