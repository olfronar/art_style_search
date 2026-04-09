# Art Style Search

Self-improving loop that finds the best prompt to define and follow an art style from reference images. A meta-prompt instructs a captioner (Gemini Pro) how to describe images so a generator (Gemini Flash) can recreate them from the captions. Claude optimizes the meta-prompt through hypothesis-driven experiments.

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
  Claude (optimizer)
  Refines the meta-prompt
```

The meta-prompt is the only thing being optimized. It tells the captioner *how* to describe images -- what visual details to capture, how precise to be about colors, technique, characters, composition, etc. Better meta-prompts produce captions that lead to more faithful recreations.

### Optimization Loop

0. **Zero-step**: Fix 20 reference images. Caption them, run parallel style analysis (Gemini vision + Claude reasoning) to build a `StyleProfile` and N diverse initial meta-prompts.
1. Claude proposes N experiments per iteration, each testing a different hypothesis about what to change in the meta-prompt.
2. Each experiment runs in parallel: caption all references with the proposed meta-prompt, generate images from captions, evaluate against originals.
3. The best experiment's template becomes the new baseline. All results feed into a shared Knowledge Base that tracks hypothesis chains, per-category progress, and open problems.
4. Repeat until convergence (max iterations, plateau, or Claude signals stop).

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Google API key](https://aistudio.google.com/apikey) (for Gemini models -- always required)
- One of:
  - [Anthropic API key](https://console.anthropic.com/) (for Claude -- default)
  - [Z.AI API key](https://z.ai/) (for GLM-5.1 -- alternative)

## Quick Start

```bash
# Install dependencies
uv sync

# Configure API keys
cp .env.sample .env
# Edit .env with your keys

# Add reference images
# Place 5-20 images of the target art style in reference_images/

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
| `--reasoning-provider` | `anthropic` | Reasoning provider: `anthropic` or `zai` |
| `--reasoning-model` | auto | Model name (default: `claude-sonnet-4-6` / `glm-5.1`) |
| `--gemini-concurrency` | `50` | Max concurrent Gemini API calls |
| `--eval-concurrency` | `4` | Max concurrent eval threads |

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
  prompt.py      Claude meta-prompt proposal/refinement
  analyze.py     Zero-step: parallel Gemini+Claude style analysis
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

```bash
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run pytest tests/     # Run tests
```

Ruff handles both linting and formatting (config in `pyproject.toml`, line length 120).
