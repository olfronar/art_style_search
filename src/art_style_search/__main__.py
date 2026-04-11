"""Entry point for ``python -m art_style_search``."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from art_style_search.runs import DEFAULT_RUNS_DIR


def _handle_clean(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="art_style_search clean", description="Remove run data")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Base directory for all runs")
    parser.add_argument("--run", type=str, default=None, help="Remove a specific run")
    parser.add_argument("--all", action="store_true", dest="remove_all", help="Remove all runs")
    args = parser.parse_args(argv)

    if not args.run and not args.remove_all:
        print("Error: specify --run <name> or --all.", file=sys.stderr)
        sys.exit(1)
    if args.run and args.remove_all:
        print("Error: --run and --all are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    from art_style_search.runs import remove_all_runs, remove_run

    if args.remove_all:
        remove_all_runs(args.runs_dir)
    else:
        remove_run(args.runs_dir, args.run)


def _handle_list(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="art_style_search list", description="Show all runs")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Base directory for all runs")
    args = parser.parse_args(argv)

    from art_style_search.runs import list_runs

    runs = list_runs(args.runs_dir)
    if not runs:
        print("No runs found.")
        return

    print(f"{'Name':<25} {'Status':<25} {'Iter':<6} {'Created':<17}")
    print("-" * 73)
    for r in runs:
        print(f"{r['name']:<25} {r['status']:<25} {r['iteration']:<6} {r['created']:<17}")


def _handle_report(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="art_style_search report", description="Generate a post-run HTML report")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Base directory for all runs")
    parser.add_argument("--run", type=str, default=None, help="Generate report for a specific run")
    parser.add_argument("--all", action="store_true", dest="report_all", help="Regenerate reports for all runs")
    parser.add_argument("--open", action="store_true", dest="open_browser", help="Open the report in a browser")
    parser.add_argument("--offline", action="store_true", help="Embed Plotly JS inline for offline viewing")
    args = parser.parse_args(argv)

    if not args.run and not args.report_all:
        print("Error: specify --run <name> or --all.", file=sys.stderr)
        sys.exit(1)
    if args.run and args.report_all:
        print("Error: --run and --all are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    from art_style_search.report import build_all_reports, build_report

    if args.report_all:
        generated = build_all_reports(args.runs_dir)
        if not generated:
            print("No runs with state.json found.")
            return
        for path in generated:
            print(path)
        return

    run_dir = args.runs_dir / args.run
    if not run_dir.is_dir():
        print(f"Error: run {args.run!r} not found at {run_dir}.", file=sys.stderr)
        sys.exit(1)
    try:
        path = build_report(run_dir, open_browser=args.open_browser, offline=args.offline)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(path)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        _handle_clean(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        _handle_list(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        _handle_report(sys.argv[2:])
        return

    from art_style_search.config import parse_args
    from art_style_search.loop import run

    config = parse_args()
    print(f"Run: {config.run_dir.name} ({config.run_dir})")

    state = asyncio.run(run(config))

    if state.converged:
        print(f"\nConverged: {state.convergence_reason}")
    print(f"Best prompt: {state.global_best_prompt}")


if __name__ == "__main__":
    main()
