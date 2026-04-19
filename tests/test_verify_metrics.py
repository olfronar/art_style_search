"""Unit tests for pure helpers in art_style_search.verify_metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from art_style_search.verify_metrics import (
    _classify,
    _parse_section_names,
    find_kept_branch,
    find_newest_run,
    load_meta_prompt,
    pick_random_caption,
)


def _write_branch_log(
    log_dir: Path,
    *,
    iteration: int,
    branch_id: int,
    kept: bool,
    captions: list[tuple[str, str]],
    dreamsim: float = 0.0,
) -> Path:
    path = log_dir / f"iter_{iteration:03d}_branch_{branch_id}.json"
    payload = {
        "iteration": iteration,
        "branch_id": branch_id,
        "kept": kept,
        "iteration_captions": [{"image_path": p, "text": t} for p, t in captions],
        "aggregated": {"dreamsim_similarity_mean": dreamsim},
    }
    path.write_text(json.dumps(payload))
    return path


class TestLoadMetaPrompt:
    def test_prefers_json_over_txt(self, tmp_path: Path) -> None:
        (tmp_path / "best_prompt.txt").write_text("FALLBACK_TEXT")
        template_payload = {
            "template": {
                "sections": [{"name": "style_foundation", "description": "canon", "value": "CANON_BODY"}],
                "caption_sections": [],
                "caption_length_target": 0,
                "negative_prompt": None,
            }
        }
        (tmp_path / "best_prompt.json").write_text(json.dumps(template_payload))
        rendered, source = load_meta_prompt(tmp_path)
        assert source == "json"
        assert "CANON_BODY" in rendered
        assert "FALLBACK_TEXT" not in rendered

    def test_falls_back_to_txt(self, tmp_path: Path) -> None:
        (tmp_path / "best_prompt.txt").write_text("PLAIN_TEXT_PROMPT")
        rendered, source = load_meta_prompt(tmp_path)
        assert source == "txt"
        assert rendered == "PLAIN_TEXT_PROMPT"

    def test_hard_fail_when_neither_exists(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_meta_prompt(tmp_path)


class TestFindKeptBranch:
    def test_prefers_kept_branch_in_newest_iter(self, tmp_path: Path) -> None:
        _write_branch_log(tmp_path, iteration=0, branch_id=0, kept=True, captions=[("a.png", "old")])
        _write_branch_log(
            tmp_path, iteration=1, branch_id=0, kept=False, captions=[("b.png", "new-unkept")], dreamsim=0.9
        )
        _write_branch_log(tmp_path, iteration=1, branch_id=1, kept=True, captions=[("c.png", "new-kept")], dreamsim=0.2)
        branch = find_kept_branch(tmp_path)
        assert branch.iteration == 1
        assert branch.branch_id == 1
        assert branch.kept is True

    def test_falls_back_to_highest_score_when_nothing_kept(self, tmp_path: Path) -> None:
        _write_branch_log(tmp_path, iteration=2, branch_id=0, kept=False, captions=[("a.png", "low")], dreamsim=0.1)
        _write_branch_log(tmp_path, iteration=2, branch_id=1, kept=False, captions=[("b.png", "high")], dreamsim=0.8)
        _write_branch_log(tmp_path, iteration=2, branch_id=2, kept=False, captions=[("c.png", "mid")], dreamsim=0.4)
        branch = find_kept_branch(tmp_path)
        assert branch.branch_id == 1

    def test_ignores_branches_with_no_captions(self, tmp_path: Path) -> None:
        _write_branch_log(tmp_path, iteration=0, branch_id=0, kept=True, captions=[])
        _write_branch_log(tmp_path, iteration=0, branch_id=1, kept=False, captions=[("a.png", "ok")])
        branch = find_kept_branch(tmp_path)
        assert branch.branch_id == 1

    def test_hard_fail_when_no_loadable_logs(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_kept_branch(tmp_path)


class TestPickRandomCaption:
    def test_seed_determinism(self, tmp_path: Path) -> None:
        _write_branch_log(
            tmp_path,
            iteration=0,
            branch_id=0,
            kept=True,
            captions=[(f"img_{i}.png", f"cap_{i}") for i in range(10)],
        )
        branch = find_kept_branch(tmp_path)
        pick_a = pick_random_caption(branch, seed=42)
        pick_b = pick_random_caption(branch, seed=42)
        pick_c = pick_random_caption(branch, seed=43)
        assert pick_a == pick_b
        assert pick_a != pick_c or len(branch.captions) == 1

    def test_returns_matching_pair(self, tmp_path: Path) -> None:
        _write_branch_log(
            tmp_path,
            iteration=0,
            branch_id=0,
            kept=True,
            captions=[("only.png", "only-caption")],
        )
        branch = find_kept_branch(tmp_path)
        ref_path, caption = pick_random_caption(branch, seed=0)
        assert ref_path == Path("only.png")
        assert caption == "only-caption"


class TestClassify:
    def test_minimum_tolerance(self) -> None:
        assert _classify(1.0, minimum=0.999) == "OK"
        assert _classify(0.9995, minimum=0.999) == "OK"
        assert _classify(0.998, minimum=0.999) == "FAIL"

    def test_exact_match(self) -> None:
        assert _classify(1.0, exact=1.0) == "OK"
        assert _classify(0.5, exact=1.0) == "FAIL"
        assert _classify(0.99999, exact=1.0) == "FAIL"

    def test_informational(self) -> None:
        assert _classify(0.42) == "INFO"


class TestParseSectionNames:
    def test_extracts_section_headers(self) -> None:
        rendered = """## style_foundation
_canon_
body

## subject_anchor
_anchor_
body

## Negative Prompt
_neg_

## Caption Sections (in order)
- [Art Style]
"""
        names = _parse_section_names(rendered)
        assert names == ["style_foundation", "subject_anchor"]

    def test_empty_input(self) -> None:
        assert _parse_section_names("") == []


class TestFindNewestRun:
    def test_picks_newest_runnable_dir(self, tmp_path: Path) -> None:
        older = tmp_path / "older"
        (older / "logs").mkdir(parents=True)
        (older / "logs" / "best_prompt.txt").write_text("old")
        _write_branch_log(older / "logs", iteration=0, branch_id=0, kept=True, captions=[("a.png", "ok")])

        newer = tmp_path / "newer"
        (newer / "logs").mkdir(parents=True)
        (newer / "logs" / "best_prompt.txt").write_text("new")
        _write_branch_log(newer / "logs", iteration=0, branch_id=0, kept=True, captions=[("a.png", "ok")])

        # Bump newer's mtime to make it win
        import os
        import time

        time.sleep(0.01)
        os.utime(newer, None)

        result = find_newest_run(tmp_path)
        assert result == newer

    def test_skips_dir_without_prompt(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad"
        (bad / "logs").mkdir(parents=True)
        _write_branch_log(bad / "logs", iteration=0, branch_id=0, kept=True, captions=[("a.png", "ok")])

        good = tmp_path / "good"
        (good / "logs").mkdir(parents=True)
        (good / "logs" / "best_prompt.txt").write_text("prompt")
        _write_branch_log(good / "logs", iteration=0, branch_id=0, kept=True, captions=[("a.png", "ok")])

        assert find_newest_run(tmp_path) == good

    def test_hard_fail_when_no_runs(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_newest_run(tmp_path)
