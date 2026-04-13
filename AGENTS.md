# Repository Guidelines

These repository guidelines are supplemental. Use this file together with `CLAUDE.md` as repo-specific context when helpful. If either file conflicts with system or developer instructions, follow those higher-priority instructions first.

## Project Structure & Module Organization
Core code lives in `src/art_style_search/`. `__main__.py` owns the CLI and the `list`, `clean`, and `report` subcommands. `loop.py` and `report.py` are thin public facades; most loop orchestration now lives under `workflow/`, and report loading/rendering is split across `report_data.py` and `reporting/`. `workflow/iteration_proposals.py` now requests a raw proposal batch and selects a 3-direction portfolio, while `workflow/policy.py` owns plateau escape and promotion decisions. Reasoning-model prompt logic lives in `prompt/`; `prompt/experiments.py` is the main seam for the 3-direction, 8-12 raw-proposal search policy and mechanism-aware dedup/selection. Shared dataclasses and scoring live in `types.py`, `contracts.py`, `scoring.py`, and `taxonomy.py`. Persistence is split across `state.py`, `state_codec.py`, and `state_migrations.py`; experiment and KB metadata now include direction/mechanism fields such as `direction_id`, `failure_mechanism`, `risk_level`, and `changed_sections`, so schema changes must update loaders, migrations, and round-trip tests together. Tests live in `tests/`; start with `tests/test_loop.py`, `tests/test_loop_e2e.py`, `tests/test_iteration_proposals.py`, `tests/test_state.py`, and `tests/test_scoring_rigor.py` for the highest-impact flows. Runtime data is local-only: `reference_images/` for inputs and `runs/` for outputs, logs, state, manifests, promotion logs, and generated reports. Model weights are downloaded into external caches, not committed under a repo `models/` directory.

## Build, Test, and Development Commands
Use `uv` for all Python workflows:

- `uv sync` installs project and dev dependencies.
- `uv run pytest tests/` runs the full unit/integration suite.
- `uv run pytest tests/test_<module>.py -k <expr>` runs targeted tests while iterating.
- `uv run ruff check .` runs lint checks.
- `uv run ruff format .` applies formatting when you need an explicit pass, but avoid unrelated format churn.
- `uv run python -m art_style_search --help` shows CLI options.
- `uv run python -m art_style_search list` lists runs with status.
- `uv run python -m art_style_search clean --run <name>` removes a specific run.
- `uv run python -m art_style_search clean --all` removes all runs.
- `uv run python -m art_style_search --max-iterations 1 --num-branches 1 --raw-proposals 8 --num-fixed-refs 3 --run smoke-test --new` runs a low-cost end-to-end smoke test.
- `uv run python -m art_style_search report --run <name>` regenerates one run report.
- `uv run python -m art_style_search report --run <name> --offline` embeds Plotly for portable offline viewing.
- `uv run python -m art_style_search report --all` regenerates reports for all runs with `state.json`.
Default search settings are now `--max-iterations 10`, `--num-branches 5`, and `--raw-proposals 9`; the reasoner generates 8-12 raw ideas grouped into 3 directions, then the workflow selects the evaluated portfolio.

## Coding Style & Naming Conventions
Target Python 3.11+ and follow Ruff defaults configured in `pyproject.toml`: 4-space indentation, line length 120, and double quotes. Prefer explicit type hints and dataclasses for structured state. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; keep tests named `test_<behavior>`. Preserve the public facade imports in `loop.py`, `report.py`, and `prompt/__init__.py` unless the change explicitly intends to break that surface. Keep edits narrowly scoped in large, high-traffic modules such as `evaluate.py`, `experiment.py`, `workflow/context.py`, and `reporting/render.py`.

## Testing Guidelines
Write pytest coverage beside the affected area using `tests/test_<module>.py`. Follow existing patterns with `tmp_path`, `monkeypatch`, and fake inputs instead of real API calls. When changing proposal generation, direction grouping, or portfolio selection, update `tests/test_prompt.py`, `tests/test_diversity.py`, `tests/test_iteration_proposals.py`, and the loop-level behavior in `tests/test_loop_e2e.py`. When changing persistence, hypothesis metadata, or historical run artifacts, update round-trip and migration coverage in `tests/test_state.py` and behavior coverage in `tests/test_knowledge.py`. When changing report structure or HTML rendering, update `tests/test_report.py`. When changing rigorous-mode promotion or scoring logic, cover the degenerate cases in `tests/test_scoring_rigor.py` and the policy-level behavior in `tests/test_loop.py`. Run targeted tests during iteration, then `uv run pytest tests/` and `uv run ruff check .` before wrapping up.

## Persistence & Compatibility Notes
Files under `runs/<name>/` are part of the product surface: `state.json`, `logs/iter_*.json`, `run_manifest.json`, `promotion_log.jsonl`, `holdout_summary.json`, and `report.html`. Changes to serialized dataclasses or logged fields should preserve backward compatibility for existing runs whenever practical. This is especially important for experiment and KB metadata fields such as `changed_section`/`changed_sections`, `target_category`, `direction_id`, `failure_mechanism`, `intervention_type`, and `risk_level`. If a format change is intentional, bump the relevant schema version in `state_migrations.py`, teach the loaders in `state.py` how to read older payloads, and extend the migration tests. Statistical outputs written to logs and reports should stay finite and JSON-safe.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Dedupe compliance checks and refresh docs` and `Add subject-first captioning and scoring`. Keep commits focused and descriptive. PRs should summarize behavior changes, list test coverage, note any CLI, config, or schema updates, and include screenshots when changing `report.html` output.

## Security & Configuration Tips
Do not commit `.env`, API keys, `reference_images/`, `runs/`, or downloaded model caches. Copy `.env.sample` to `.env` for local setup. Optional: install the `gitleaks` pre-commit hook from `.pre-commit-config.yaml` to catch secrets before commit.
