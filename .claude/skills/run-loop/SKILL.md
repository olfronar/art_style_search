---
name: run-loop
description: Execute the self-improving art style prompt optimization loop. Only invoke manually.
disable-model-invocation: true
---

Run the art style search optimization loop:

```bash
uv run python -m art_style_search.loop $ARGUMENTS
```

Before running, verify:
1. `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` are set in the environment
2. Reference images exist in `reference_images/`
3. Dependencies are installed (`uv sync`)

After running, summarize:
- Number of iterations completed
- Best prompt found and its metrics
- Metric trajectory (improving/plateauing/diverging)
