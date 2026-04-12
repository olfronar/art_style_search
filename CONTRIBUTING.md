# Contributing

## Development Setup

1. Fork the repository and create a feature branch from `main`.
2. Install dependencies with `uv sync`.
3. Copy `.env.sample` to `.env` only if you need to run the full model-backed loop locally.
4. Keep `reference_images/`, `runs/`, `models/`, and `.env` out of commits. They are local-only inputs and outputs.

## Verification

Run the repo checks before opening a pull request:

```bash
uv run ruff check .
uv run pytest tests/
uv build
```

If you touch CLI docs or packaging metadata, also confirm the help output still looks right:

```bash
uv run python -m art_style_search --help
```

## Secret Scanning

This repository treats secret scanning as part of the release surface.

- Install the local hook:

  ```bash
  uv tool install pre-commit
  pre-commit install
  ```

- Run the repository scan via the hook on demand:

  ```bash
  pre-commit run gitleaks --all-files
  ```

- For a full repository history scan before publishing, run the GitHub Actions workflow in `.github/workflows/gitleaks.yml` or install the `gitleaks` CLI locally and run:

  ```bash
  gitleaks git .
  ```

## Pull Requests

- Keep pull requests focused.
- Describe behavior changes, schema changes, and report output changes clearly.
- Call out any API-provider or cost implications.
- Add or update tests when behavior changes.

## Style Notes

- Use the existing repo patterns and keep edits narrow in high-traffic modules.
- Prefer `snake_case` names for Python symbols and `test_<behavior>` for tests.
- Avoid unrelated format churn.
