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
  (DINO, LPIPS, HPS, Aesthetics + Gemini vision)
        |
        v
  Claude (optimizer)
  Refines the meta-prompt
```

The meta-prompt is the only thing being optimized. It tells the captioner *how* to describe images -- what visual details to capture, how precise to be about colors, technique, characters, composition, etc. Better meta-prompts produce captions that lead to more faithful recreations.

### Optimization Loop

0. **Zero-step**: Fix 10 reference images. Caption them, run parallel style analysis (Gemini vision + Claude reasoning) to build a `StyleProfile` and N diverse initial meta-prompts.
1. Claude proposes N experiments per iteration, each testing a different hypothesis about what to change in the meta-prompt.
2. Each experiment runs in parallel: caption all references with the proposed meta-prompt, generate images from captions, evaluate against originals.
3. The best experiment's template becomes the new baseline. All results feed into a shared Knowledge Base that tracks hypothesis chains, per-category progress, and open problems.
4. Repeat until convergence (max iterations, plateau, or Claude signals stop).

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Google API key](https://aistudio.google.com/apikey) (for Gemini models — always required)
- One of:
  - [Anthropic API key](https://console.anthropic.com/) (for Claude — default)
  - [Z.AI API key](https://z.ai/) (for GLM-5 — alternative)

## Quick Start

```bash
# Install dependencies
uv sync

# Configure API keys
cp .env.sample .env
# Edit .env with your keys

# Add reference images
# Place 5-20 images of the target art style in reference_images/

# Run the optimization loop
uv run python -m art_style_search

# View all options
uv run python -m art_style_search --help
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--reference-dir` | `reference_images` | Directory containing reference art |
| `--output-dir` | `outputs` | Generated images output |
| `--log-dir` | `logs` | Iteration logs, captions cache |
| `--state-file` | `state.json` | State file for resume |
| `--max-iterations` | `20` | Maximum optimization iterations |
| `--plateau-window` | `5` | Iterations without improvement before stop |
| `--num-branches` | `5` | Parallel experiments per iteration |
| `--num-images` | `4` | Images generated per experiment |
| `--aspect-ratio` | `1:1` | Aspect ratio for generated images |
| `--caption-model` | `gemini-3.1-pro-preview` | Gemini model for captioning |
| `--generator-model` | `gemini-3.1-flash-image-preview` | Gemini model for generation |
| `--reasoning-provider` | `anthropic` | Reasoning provider: `anthropic` or `zai` |
| `--reasoning-model` | auto | Model name (default: `claude-opus-4-6` / `glm-5`) |

```bash
# Clean all generated outputs, logs, and state
uv run python -m art_style_search clean
```

## Evaluation Metrics

Each metric compares a generated image against its specific paired original:

| Metric | Measures | Scale | Better |
|--------|----------|-------|--------|
| **DINO** | Semantic/structural similarity (DINOv2 embeddings) | -1 to 1 | Higher |
| **LPIPS** | Perceptual distance (AlexNet) | 0+ | Lower |
| **HPS v2** | Caption-image alignment (human preference) | ~0.2-0.3 | Higher |
| **Aesthetics** | Visual quality (LAION predictor) | 1-10 | Higher |

Composite score: `0.4*DINO - 0.2*LPIPS + 0.2*HPS + 0.2*(Aesthetics/10)`

## Project Structure

```
src/art_style_search/
  __main__.py    Entry point + clean subcommand
  loop.py        Experiment-based orchestration loop
  prompt.py      Claude meta-prompt proposal/refinement
  analyze.py     Zero-step: parallel Gemini+Claude style analysis
  caption.py     Gemini Pro captioning with disk cache
  generate.py    Gemini Flash image generation with retry
  evaluate.py    Per-image paired metrics + Gemini vision comparison
  models.py      Lazy-loaded DINO/LPIPS/HPS/Aesthetics models
  types.py       Shared dataclasses + KnowledgeBase
  config.py      CLI argument parsing
  state.py       JSON persistence + backward compat migration
  utils.py       Shared helpers (API wrappers, MIME map)
```

## Development

```bash
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run pytest tests/     # Run tests (172 tests)
```

Ruff handles both linting and formatting (config in `pyproject.toml`, line length 120).
