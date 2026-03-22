---
name: verify
description: Run ruff linting and format checks on the entire project. Use after making changes to verify code quality.
disable-model-invocation: true
---

Run the project verification checks. Execute both commands and report results:

```bash
uv run ruff check .
uv run ruff format --check .
```

If either command fails, report the specific errors and fix them.
