"""Entry point for ``python -m art_style_search``."""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from art_style_search.config import parse_args


def clean(
    output_dir: Path = Path("outputs"), log_dir: Path = Path("logs"), state_file: Path = Path("state.json")
) -> None:
    """Remove generated outputs, logs, and state file."""
    for d in (output_dir, log_dir):
        if d.is_dir():
            shutil.rmtree(d)
            print(f"Removed {d}/")
    if state_file.is_file():
        state_file.unlink()
        print(f"Removed {state_file}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean()
        return

    from art_style_search.loop import run

    config = parse_args()
    state = asyncio.run(run(config))

    if state.converged:
        print(f"\nConverged: {state.convergence_reason}")
    print(f"Best prompt: {state.global_best_prompt}")


if __name__ == "__main__":
    main()
