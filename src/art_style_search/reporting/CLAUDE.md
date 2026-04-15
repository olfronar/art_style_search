# reporting/ — HTML report

Note: the façade lives at `src/art_style_search/report.py` (re-exports `build_report`, `build_all_reports`, `load_report_data`, `ReportData`). Data loading lives in `report_data.py` (`ReportData` dataclass centralizes state, iteration logs, manifest, promotion decisions, holdout summary).

## Module map

- `render.py` - HTML section renderers: header, summary (score trajectory + hypothesis outcomes + top open problems), trajectories, iteration drilldown, KB, protocol, promotion decisions, holdout. Vision feedback XML tags parsed into styled verdict cards. Prompt diffs via `difflib.unified_diff`.
- `charts.py` - Plotly chart builders (composite trajectory + per-metric subplots); lazy-imports Plotly.
- `document.py` - HTML5 document assembly; CSS lazy-loaded via `functools.cache`. `--offline` embeds Plotly JS inline.
- `report.css` - Editorial dark-theme design system (CSS custom properties, responsive at 880px).

## Conventions

- HTML report metric trajectories must use `composite_score` only — `adaptive_composite_score` is min-max normalized within a single batch and is meaningless across iterations. Within an iteration's experiment table, `adaptive_composite_score` is fine (and useful for ranking) because it's recomputed per-batch.
- Vision comparison verdicts are ternary (MATCH/PARTIAL/MISS) — the canonical `VERDICT_PATTERN` lives in `evaluate.py` and is consumed here via `reporting/render.py` to parse vision feedback XML tags into styled cards.
- Open-problem text: the `[HIGH]/[MED]/[LOW]` priority prefix is stripped at render time via `strip_priority_prefix` (shared with `knowledge.py`); parsed priority overrides the code heuristic.
