"""Unit tests for art_style_search.report — HTML generation, data loading, helpers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from art_style_search.report import build_report
from art_style_search.report_data import ReportData, _load_iteration_logs, _rel, load_report_data
from art_style_search.reporting.render import (
    _count_descendants,
    _format_vision_feedback,
    _render_hypothesis_tree,
    _render_open_problems,
)
from art_style_search.state import append_promotion_log, save_iteration_log, save_manifest, save_state
from art_style_search.types import (
    Hypothesis,
    KnowledgeBase,
    OpenProblem,
    PromotionDecision,
    RunManifest,
)
from tests.conftest import (
    make_aggregated_metrics,
    make_iteration_result,
    make_loop_state,
)

# ---------------------------------------------------------------------------
# _rel
# ---------------------------------------------------------------------------


class TestRel:
    """Tests for the path-relativization helper used by image src attributes."""

    def test_same_dir_returns_filename(self, tmp_path: Path) -> None:
        report_dir = tmp_path / "runs" / "demo"
        report_dir.mkdir(parents=True)
        target = report_dir / "outputs" / "iter_001" / "exp_0" / "00.png"
        target.parent.mkdir(parents=True)
        target.touch()

        rel = _rel(target, report_dir)
        assert rel == "outputs/iter_001/exp_0/00.png"

    def test_parent_traversal(self, tmp_path: Path) -> None:
        report_dir = tmp_path / "runs" / "demo"
        report_dir.mkdir(parents=True)
        target = tmp_path / "reference_images" / "ref.png"
        target.parent.mkdir(parents=True)
        target.touch()

        rel = _rel(target, report_dir)
        assert rel == "../../reference_images/ref.png"

    def test_uses_forward_slashes(self, tmp_path: Path) -> None:
        report_dir = tmp_path / "a"
        report_dir.mkdir()
        target = tmp_path / "b" / "c" / "d.png"
        target.parent.mkdir(parents=True)
        target.touch()

        rel = _rel(target, report_dir)
        assert "\\" not in rel
        assert "/" in rel


# ---------------------------------------------------------------------------
# _load_iteration_logs
# ---------------------------------------------------------------------------


class TestLoadIterationLogs:
    """Tests for the iteration-log directory loader."""

    def test_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        assert _load_iteration_logs(log_dir) == {}

    def test_missing_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        assert _load_iteration_logs(tmp_path / "missing") == {}

    def test_loads_valid_logs_grouped_by_iteration(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        save_iteration_log(make_iteration_result(branch_id=0, iteration=1), log_dir)
        save_iteration_log(make_iteration_result(branch_id=1, iteration=1), log_dir)
        save_iteration_log(make_iteration_result(branch_id=0, iteration=2), log_dir)

        loaded = _load_iteration_logs(log_dir)
        assert sorted(loaded.keys()) == [1, 2]
        assert len(loaded[1]) == 2
        assert len(loaded[2]) == 1
        # Sorted by branch_id
        assert [r.branch_id for r in loaded[1]] == [0, 1]

    def test_skips_malformed_files_without_crashing(self, tmp_path: Path, caplog) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        save_iteration_log(make_iteration_result(branch_id=0, iteration=1), log_dir)
        # Hand-craft a half-written file
        (log_dir / "iter_001_branch_1.json").write_text("{not valid json", encoding="utf-8")
        # And one with valid JSON but missing required fields
        (log_dir / "iter_002_branch_0.json").write_text('{"iteration": 2}', encoding="utf-8")

        loaded = _load_iteration_logs(log_dir)
        # The valid log loaded; the malformed ones were dropped
        assert 1 in loaded
        assert len(loaded[1]) == 1
        assert 2 not in loaded


class TestReportDataSelection:
    def test_kept_and_top_scoring_results_can_differ(self) -> None:
        kept = make_iteration_result(branch_id=1, iteration=1)
        kept.hypothesis = "kept branch"
        kept.kept = True
        kept.aggregated = replace(
            kept.aggregated, dreamsim_similarity_mean=0.55, color_histogram_mean=0.45, ssim_mean=0.45
        )

        top_raw = make_iteration_result(branch_id=0, iteration=1)
        top_raw.hypothesis = "top raw branch"
        top_raw.kept = False
        top_raw.aggregated = replace(
            top_raw.aggregated,
            dreamsim_similarity_mean=0.75,
            color_histogram_mean=0.70,
            ssim_mean=0.70,
        )

        data = ReportData(
            run_name="demo",
            run_dir=Path("/tmp/demo"),
            state=make_loop_state(global_best_metrics=make_aggregated_metrics()),
            iteration_logs={1: [top_raw, kept]},
        )

        assert data.kept_of(1).branch_id == 1
        assert data.top_scoring_of(1).branch_id == 0


# ---------------------------------------------------------------------------
# _render_hypothesis_tree
# ---------------------------------------------------------------------------


def _make_hypothesis(
    hid: str,
    parent_id: str | None = None,
    *,
    iteration: int = 0,
    outcome: str = "confirmed",
    statement: str = "test",
    category: str = "color_palette",
) -> Hypothesis:
    return Hypothesis(
        id=hid,
        iteration=iteration,
        parent_id=parent_id,
        statement=statement,
        experiment="test experiment",
        category=category,
        outcome=outcome,
        metric_delta={},
        kept=outcome == "confirmed",
        lesson="",
    )


class TestRenderHypothesisTree:
    """Tests for the recursive hypothesis tree renderer."""

    def test_empty_tree_renders_placeholder(self) -> None:
        kb = KnowledgeBase()
        html = _render_hypothesis_tree(kb)
        assert "No hypotheses recorded" in html

    def test_single_root_renders_node(self) -> None:
        kb = KnowledgeBase(hypotheses=[_make_hypothesis("H1", statement="Add red accents")])
        html = _render_hypothesis_tree(kb)
        assert "H1" in html
        assert "Add red accents" in html
        assert "hyp-confirmed" in html

    def test_parent_child_uses_details_element(self) -> None:
        h1 = _make_hypothesis("H1", statement="Brighten palette")
        h2 = _make_hypothesis("H2", parent_id="H1", iteration=1, statement="Boost saturation")
        kb = KnowledgeBase(hypotheses=[h1, h2])
        html = _render_hypothesis_tree(kb)
        # Parent with children becomes a <details>
        assert "<details" in html
        assert "Brighten palette" in html
        assert "Boost saturation" in html

    def test_orphan_hypothesis_treated_as_root(self) -> None:
        # H2's parent_id points to a missing H99 — should be rendered as a root.
        h1 = _make_hypothesis("H1", statement="Root one")
        h2 = _make_hypothesis("H2", parent_id="H99", statement="Orphan")
        kb = KnowledgeBase(hypotheses=[h1, h2])
        html = _render_hypothesis_tree(kb)
        assert "Orphan" in html
        assert "Root one" in html

    def test_depth_cap_collapses_descendants(self) -> None:
        # Build a chain H1 -> H2 -> ... -> H10 (depth 10)
        chain: list[Hypothesis] = []
        for i in range(1, 11):
            parent = f"H{i - 1}" if i > 1 else None
            chain.append(_make_hypothesis(f"H{i}", parent_id=parent, iteration=i, statement=f"step {i}"))
        kb = KnowledgeBase(hypotheses=chain)
        html = _render_hypothesis_tree(kb)
        # All ten IDs are still mentioned (some inside the collapsed marker is OK,
        # but the visible ones definitely are)
        assert "H1" in html
        # The "(+N deeper — collapsed)" marker should appear
        assert "deeper" in html
        assert "collapsed" in html


class TestCountDescendants:
    """Tests for the recursive descendant counter helper."""

    def test_no_children(self) -> None:
        assert _count_descendants("H1", {}) == 0

    def test_linear_chain(self) -> None:
        children_map = {
            "H1": [_make_hypothesis("H2", parent_id="H1")],
            "H2": [_make_hypothesis("H3", parent_id="H2")],
            "H3": [_make_hypothesis("H4", parent_id="H3")],
        }
        assert _count_descendants("H1", children_map) == 3

    def test_branching(self) -> None:
        children_map = {
            "H1": [
                _make_hypothesis("H2", parent_id="H1"),
                _make_hypothesis("H3", parent_id="H1"),
            ],
            "H2": [_make_hypothesis("H4", parent_id="H2")],
        }
        assert _count_descendants("H1", children_map) == 3


# ---------------------------------------------------------------------------
# _render_open_problems
# ---------------------------------------------------------------------------


class TestRenderOpenProblems:
    """Tests for the open-problems list renderer."""

    def test_empty_renders_placeholder(self) -> None:
        assert "No open problems" in _render_open_problems([])

    def test_priority_chip_class(self) -> None:
        problems = [
            OpenProblem(text="Edges too sharp", category="technique", priority="HIGH", since_iteration=2),
            OpenProblem(text="Cooler tones missing", category="color_palette", priority="MED", since_iteration=4),
            OpenProblem(text="Minor texture issue", category="texture", priority="LOW", since_iteration=5),
        ]
        html = _render_open_problems(problems)
        assert "prio-high" in html
        assert "prio-med" in html
        assert "prio-low" in html
        assert "Edges too sharp" in html

    def test_metric_gap_appears_when_present(self) -> None:
        problems = [
            OpenProblem(
                text="Color drift",
                category="color_palette",
                priority="HIGH",
                metric_gap=-0.123,
                since_iteration=3,
            )
        ]
        html = _render_open_problems(problems)
        assert "-0.123" in html


class TestFormatVisionFeedback:
    def test_malformed_verdict_tags_do_not_leak_raw_xml(self) -> None:
        raw = '**img.png** [S=P Su=P Co=P]: <style verdict="PARTIAL">The linework is too heavy.'
        rendered = _format_vision_feedback(raw)
        assert "The linework is too heavy." in rendered
        assert "&lt;style verdict=" not in rendered


# ---------------------------------------------------------------------------
# load_report_data + build_report end-to-end
# ---------------------------------------------------------------------------


class TestBuildReport:
    """Smoke tests for the top-level report builder."""

    def test_missing_state_raises(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "ghost"
        run_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_report_data(run_dir)

    def test_state_only_no_logs_renders_empty_section(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "demo"
        run_dir.mkdir()
        state = make_loop_state(iteration=0)
        save_state(state, run_dir / "state.json")

        path = build_report(run_dir)
        assert path == run_dir / "report.html"
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert "No iteration logs available yet" in text
        assert "Art Style Search" in text or "demo" in text

    def test_full_report_contains_expected_substrings(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "full"
        run_dir.mkdir()
        log_dir = run_dir / "logs"
        log_dir.mkdir()

        # Persist a state.json with a populated KB.
        kb = KnowledgeBase()
        kb.add_hypothesis(
            iteration=1,
            parent_id=None,
            statement="Increase color saturation",
            experiment="Boost saturation 20%",
            category="color_palette",
            kept=True,
            metric_delta={"dreamsim": 0.012},
            lesson="More saturated palette improved alignment",
            confirmed="Saturation lift confirmed",
            rejected="",
        )
        kb.open_problems = [
            OpenProblem(
                text="Edge sharpness still off",
                category="technique",
                priority="HIGH",
                metric_gap=-0.04,
                since_iteration=2,
            ),
        ]

        state = make_loop_state(iteration=2)
        state.knowledge_base = kb
        save_state(state, run_dir / "state.json")

        # Persist two iteration logs (iter 1 with two experiments, iter 2 with one).
        save_iteration_log(make_iteration_result(branch_id=0, iteration=1), log_dir)
        save_iteration_log(make_iteration_result(branch_id=1, iteration=1), log_dir)
        save_iteration_log(make_iteration_result(branch_id=0, iteration=2), log_dir)

        path = build_report(run_dir)
        text = path.read_text(encoding="utf-8")

        # Header
        assert "full" in text  # run name
        assert "Iterations" in text
        # KB section
        assert "Knowledge Base" in text
        assert "Increase color saturation" in text
        assert "Edge sharpness still off" in text
        assert "prio-high" in text
        # Trajectories section + Plotly hooks
        assert "Plotly.newPlot" in text
        assert "composite-chart" in text
        assert "metrics-chart" in text
        # Iteration drill-down — iteration numbers are zero-padded in the editorial layout
        assert "Iteration 01" in text
        assert "Iteration 02" in text
        # Plotly figure JSON is embedded as <script type=application/json>
        composite_block = text.split('id="composite-data">', 1)[1].split("</script>", 1)[0]
        assert json.loads(composite_block)  # parses cleanly

    def test_rigorous_run_without_holdout_summary_reports_missing_artifact(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "rigorous-missing-holdout"
        run_dir.mkdir()
        state = make_loop_state(iteration=1)
        state.converged = True
        state.protocol = "rigorous"
        state.silent_refs = [Path("/tmp/silent.png")]
        save_state(state, run_dir / "state.json")

        path = build_report(run_dir)
        text = path.read_text(encoding="utf-8")

        assert "expected for this rigorous run" in text
        assert "Enable the rigorous protocol" not in text

    def test_in_progress_rigorous_run_reports_holdout_as_pending(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "rigorous-pending-holdout"
        run_dir.mkdir()
        state = make_loop_state(iteration=1)
        state.converged = False
        state.protocol = "rigorous"
        state.silent_refs = [Path("/tmp/silent.png")]
        save_state(state, run_dir / "state.json")

        text = build_report(run_dir).read_text(encoding="utf-8")

        assert "will be written when the run finishes" in text
        assert "expected for this rigorous run" not in text

    def test_report_shows_requested_actual_and_feedback_silent_counts(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "counts"
        run_dir.mkdir()
        state = make_loop_state(iteration=1)
        state.fixed_references = [Path("/tmp/ref0.png"), Path("/tmp/ref1.png"), Path("/tmp/ref2.png")]
        state.feedback_refs = state.fixed_references[:2]
        state.silent_refs = state.fixed_references[2:]
        save_state(state, run_dir / "state.json")

        manifest = RunManifest(
            protocol_version="rigorous_v1",
            seed=42,
            cli_args={},
            model_names={},
            reasoning_provider="openai",
            git_sha=None,
            python_version="3.11.8",
            platform="test",
            timestamp_utc="2026-04-12T00:00:00+00:00",
            reference_image_hashes={},
            num_fixed_refs=20,
            discovered_reference_count=5,
            uv_lock_hash=None,
        )
        save_manifest(manifest, run_dir / "run_manifest.json")

        text = build_report(run_dir).read_text(encoding="utf-8")
        assert "Requested refs" in text
        assert "Actual refs" in text
        assert "Feedback / silent" in text

    def test_promotion_table_uses_zero_based_iteration_numbers_consistently(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "promotion-iterations"
        run_dir.mkdir()
        save_state(make_loop_state(iteration=1), run_dir / "state.json")
        append_promotion_log(
            PromotionDecision(
                iteration=1,
                candidate_score=0.6,
                baseline_score=0.5,
                epsilon=0.01,
                delta=0.1,
                decision="promoted",
                reason="test",
                candidate_branch_id=0,
                candidate_hypothesis="hyp",
            ),
            run_dir / "promotion_log.jsonl",
        )

        text = build_report(run_dir).read_text(encoding="utf-8")
        assert '<tr class="promo-yes"><td>1</td><td>0</td>' in text
