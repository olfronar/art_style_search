"""Entry point for ``python -m art_style_search``."""

from __future__ import annotations

import asyncio

from art_style_search.config import parse_args
from art_style_search.loop import run


def main() -> None:
    config = parse_args()
    state = asyncio.run(run(config))

    if state.converged:
        print(f"\nConverged: {state.convergence_reason}")
    print(f"Best prompt: {state.global_best_prompt}")


if __name__ == "__main__":
    main()
