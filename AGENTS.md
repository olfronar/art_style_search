# Repository Guidelines

These repository guidelines are supplemental. Use this file together with `CLAUDE.md` as repo-specific context when helpful. If either file conflicts with system or developer instructions, follow those higher-priority instructions first.

## Project Structure & Module Organization
Core code lives in `src/art_style_search/`. Start with `__main__.py` for CLI entry points, `loop.py` for the optimization loop, `prompt/` for reasoning-model prompt logic, and `report.py` for HTML report generation. Shared types and config live in `types.py` and `config.py`. Tests are in `tests/` and generally mirror module names, for example `tests/test_config.py`. Runtime data is local-only: `reference_images/` for input art, `runs/` for per-run outputs/logs/state, and `models/` for vendored model assets.

## Build, Test, and Development Commands
Use `uv` for all Python workflows:

- `uv sync` installs project and dev dependencies.
- `uv run pytest tests/` runs the unit test suite.
- `uv run ruff check .` runs lint checks.
- `uv run ruff format .` applies formatting.
- `uv run python -m art_style_search --help` shows CLI options.
- `uv run python -m art_style_search --max-iterations 1 --num-branches 1 --num-fixed-refs 3 --run smoke-test --new` runs a low-cost end-to-end smoke test.
- `uv run python -m art_style_search report --run <name>` regenerates a run report.

## Coding Style & Naming Conventions
Target Python 3.11+ and follow Ruff defaults configured in `pyproject.toml`: 4-space indentation, line length 120, and double quotes. Prefer explicit type hints and small dataclasses for structured state. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; keep tests named `test_<behavior>`. Add helpers to `utils.py` only when the logic is shared across multiple modules.

## Testing Guidelines
Write pytest tests beside the affected area using `tests/test_<module>.py`. Follow existing patterns with `tmp_path`, `monkeypatch`, and isolated fake inputs instead of real API calls. Run targeted tests during iteration, then `uv run pytest tests/` before opening a PR.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Add post-run HTML report generator` and `Fix crash+resume stuck loop...`. Keep commits focused and descriptive. PRs should summarize behavior changes, list test coverage, note any CLI or config updates, and include screenshots when changing `report.html` output.

## Security & Configuration Tips
Do not commit `.env`, API keys, `reference_images/`, `runs/`, or model weights. Copy `.env.sample` to `.env` for local setup. Optional: install the `gitleaks` pre-commit hook from `.pre-commit-config.yaml` to catch secrets before commit.
