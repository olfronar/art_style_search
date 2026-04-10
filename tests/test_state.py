"""Unit tests for art_style_search.state — serialization, persistence, round-trips."""

from __future__ import annotations

import json
from pathlib import Path

from art_style_search.state import (
    _aggregated_metrics_from_dict,
    _caption_from_dict,
    _iteration_result_from_dict,
    _knowledge_base_from_dict,
    _loop_state_from_dict,
    _metric_scores_from_dict,
    _prompt_section_from_dict,
    append_promotion_log,
    load_manifest,
    load_promotion_log,
    load_state,
    prompt_template_from_dict,
    save_iteration_log,
    save_manifest,
    save_state,
    to_dict,
)
from art_style_search.types import (
    AggregatedMetrics,
    Caption,
    ConvergenceReason,
    IterationResult,
    KnowledgeBase,
    LoopState,
    MetricScores,
    OpenProblem,
    PromotionDecision,
    PromptSection,
    PromptTemplate,
    RunManifest,
    StyleProfile,
)

# ---------------------------------------------------------------------------
# Factory helpers — build realistic test fixtures
# ---------------------------------------------------------------------------


def make_caption(*, index: int = 0) -> Caption:
    return Caption(
        image_path=Path(f"/data/reference_images/painting_{index:03d}.png"),
        text=f"A moody watercolor landscape with muted earth tones, soft edges, and a low horizon line (image {index}).",
    )


def make_metric_scores(*, seed: float = 0.0) -> MetricScores:
    return MetricScores(
        dreamsim_similarity=0.72 + seed * 0.01,
        hps_score=0.26 + seed * 0.002,
        aesthetics_score=6.1 + seed * 0.1,
    )


def make_aggregated_metrics(*, seed: float = 0.0) -> AggregatedMetrics:
    return AggregatedMetrics(
        dreamsim_similarity_mean=0.71 + seed * 0.01,
        dreamsim_similarity_std=0.03,
        hps_score_mean=0.25 + seed * 0.002,
        hps_score_std=0.01,
        aesthetics_score_mean=5.9 + seed * 0.1,
        aesthetics_score_std=0.4,
    )


def make_prompt_section(*, index: int = 0) -> PromptSection:
    sections = [
        ("medium", "Overall artistic medium", "Watercolor painting on rough cold-pressed paper"),
        ("palette", "Color palette", "Muted earth tones: ochre, burnt sienna, Payne's grey, sap green"),
        ("composition", "Layout and framing", "Low horizon line, asymmetric balance, negative space in upper third"),
    ]
    name, desc, val = sections[index % len(sections)]
    return PromptSection(name=name, description=desc, value=val)


def make_prompt_template(*, n_sections: int = 3) -> PromptTemplate:
    return PromptTemplate(
        sections=[make_prompt_section(index=i) for i in range(n_sections)],
        negative_prompt="photorealistic, 3D render, digital art, sharp edges",
    )


def make_style_profile() -> StyleProfile:
    return StyleProfile(
        color_palette="Muted earth tones — ochre, burnt sienna, slate blue, sap green.",
        composition="Low horizon, asymmetric balance, generous negative space.",
        technique="Wet-on-wet watercolor with dry-brush texture accents.",
        mood_atmosphere="Contemplative, quiet, slightly melancholic.",
        subject_matter="Rural landscapes, fields, isolated structures.",
        influences="Andrew Wyeth, J.M.W. Turner, Japanese ink wash.",
        gemini_raw_analysis="Gemini vision analysis text for reference images...",
        claude_raw_analysis="Claude structured analysis text for reference images...",
    )


def make_iteration_result(*, branch_id: int = 0, iteration: int = 1) -> IterationResult:
    return IterationResult(
        branch_id=branch_id,
        iteration=iteration,
        template=make_prompt_template(),
        rendered_prompt=make_prompt_template().render(),
        image_paths=[Path(f"/data/outputs/iter_{iteration:03d}/img_{i}.png") for i in range(4)],
        per_image_scores=[make_metric_scores(seed=float(i)) for i in range(4)],
        aggregated=make_aggregated_metrics(seed=float(iteration)),
        claude_analysis="Good progress on tonal range; edges still too crisp compared to references.",
        template_changes="Increased wet-on-wet emphasis, added dry-brush texture note.",
        kept=True,
    )


def make_loop_state(
    *,
    iteration: int = 5,
    converged: bool = False,
    convergence_reason: ConvergenceReason | None = None,
    global_best_metrics: AggregatedMetrics | None = None,
) -> LoopState:
    return LoopState(
        iteration=iteration,
        current_template=make_prompt_template(),
        best_template=make_prompt_template(n_sections=2),
        best_metrics=make_aggregated_metrics(seed=1.0),
        knowledge_base=KnowledgeBase(),
        captions=[make_caption(index=i) for i in range(3)],
        style_profile=make_style_profile(),
        experiment_history=[make_iteration_result(branch_id=i, iteration=i + 1) for i in range(2)],
        global_best_prompt="Watercolor painting on rough cold-pressed paper, muted earth tones...",
        global_best_metrics=global_best_metrics,
        converged=converged,
        convergence_reason=convergence_reason,
    )


# ---------------------------------------------------------------------------
# Tests for _to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    """Tests for the _to_dict serialization helper."""

    def test_caption_path_becomes_string(self) -> None:
        caption = make_caption()
        d = to_dict(caption)
        assert isinstance(d["image_path"], str)
        assert d["image_path"] == "/data/reference_images/painting_000.png"
        assert d["text"] == caption.text

    def test_metric_scores_all_floats(self) -> None:
        ms = make_metric_scores(seed=2.0)
        d = to_dict(ms)
        assert set(d.keys()) == {
            "dreamsim_similarity",
            "hps_score",
            "aesthetics_score",
            "color_histogram",
            "ssim",
            "vision_style",
            "vision_subject",
            "vision_composition",
        }
        for v in d.values():
            assert isinstance(v, float)

    def test_prompt_template_sections_list(self) -> None:
        pt = make_prompt_template(n_sections=3)
        d = to_dict(pt)
        assert isinstance(d["sections"], list)
        assert len(d["sections"]) == 3
        for sec in d["sections"]:
            assert set(sec.keys()) == {"name", "description", "value"}
            assert isinstance(sec["name"], str)

    def test_prompt_template_negative_prompt_none(self) -> None:
        pt = PromptTemplate(sections=[make_prompt_section()], negative_prompt=None)
        d = to_dict(pt)
        assert d["negative_prompt"] is None

    def test_convergence_reason_enum_becomes_value(self) -> None:
        for reason in ConvergenceReason:
            assert to_dict(reason) == reason.value

    def test_full_loop_state_is_json_serializable(self) -> None:
        state = make_loop_state(global_best_metrics=make_aggregated_metrics())
        d = to_dict(state)
        # Must not raise
        text = json.dumps(d)
        assert isinstance(text, str)

    def test_nested_paths_in_iteration_result(self) -> None:
        ir = make_iteration_result()
        d = to_dict(ir)
        for p in d["image_paths"]:
            assert isinstance(p, str)
            assert p.startswith("/data/outputs/")

    def test_plain_values_pass_through(self) -> None:
        assert to_dict(42) == 42
        assert to_dict("hello") == "hello"
        assert to_dict(None) is None
        assert to_dict(3.14) == 3.14

    def test_dict_keys_are_recursed(self) -> None:
        d = {"a": make_caption(), "b": [make_metric_scores()]}
        result = to_dict(d)
        assert isinstance(result["a"]["image_path"], str)
        assert isinstance(result["b"][0]["dreamsim_similarity"], float)


# ---------------------------------------------------------------------------
# Tests for round-trip: _to_dict -> JSON -> _*_from_dict
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Test serialization -> JSON string -> deserialization yields equivalent objects."""

    def test_caption_round_trip(self) -> None:
        original = make_caption(index=7)
        d = json.loads(json.dumps(to_dict(original)))
        restored = _caption_from_dict(d)
        assert restored == original

    def test_metric_scores_round_trip(self) -> None:
        original = make_metric_scores(seed=3.0)
        d = json.loads(json.dumps(to_dict(original)))
        restored = _metric_scores_from_dict(d)
        assert restored == original

    def test_aggregated_metrics_round_trip(self) -> None:
        original = make_aggregated_metrics(seed=2.0)
        d = json.loads(json.dumps(to_dict(original)))
        restored = _aggregated_metrics_from_dict(d)
        assert restored == original

    def test_prompt_section_round_trip(self) -> None:
        original = make_prompt_section(index=1)
        d = json.loads(json.dumps(to_dict(original)))
        restored = _prompt_section_from_dict(d)
        assert restored == original

    def test_prompt_template_round_trip(self) -> None:
        original = make_prompt_template(n_sections=3)
        d = json.loads(json.dumps(to_dict(original)))
        restored = prompt_template_from_dict(d)
        assert restored.sections == original.sections
        assert restored.negative_prompt == original.negative_prompt

    def test_prompt_template_no_negative_prompt_round_trip(self) -> None:
        original = PromptTemplate(sections=[make_prompt_section()], negative_prompt=None)
        d = json.loads(json.dumps(to_dict(original)))
        restored = prompt_template_from_dict(d)
        assert restored.negative_prompt is None
        assert len(restored.sections) == 1

    def test_iteration_result_round_trip(self) -> None:
        original = make_iteration_result(branch_id=2, iteration=4)
        d = json.loads(json.dumps(to_dict(original)))
        restored = _iteration_result_from_dict(d)
        assert restored.branch_id == original.branch_id
        assert restored.iteration == original.iteration
        assert restored.rendered_prompt == original.rendered_prompt
        assert restored.image_paths == original.image_paths
        assert restored.per_image_scores == original.per_image_scores
        assert restored.aggregated == original.aggregated
        assert restored.kept == original.kept

    def test_convergence_reason_round_trip(self) -> None:
        for reason in ConvergenceReason:
            state = make_loop_state(converged=True, convergence_reason=reason)
            d = json.loads(json.dumps(to_dict(state)))
            restored = _loop_state_from_dict(d)
            assert restored.convergence_reason == reason
            assert restored.converged is True

    def test_loop_state_round_trip(self) -> None:
        original = make_loop_state(
            iteration=10,
            converged=True,
            convergence_reason=ConvergenceReason.PLATEAU,
            global_best_metrics=make_aggregated_metrics(seed=5.0),
        )
        d = json.loads(json.dumps(to_dict(original)))
        restored = _loop_state_from_dict(d)
        assert restored.iteration == original.iteration
        assert restored.converged is True
        assert restored.convergence_reason == ConvergenceReason.PLATEAU
        assert restored.global_best_metrics == original.global_best_metrics
        assert restored.global_best_prompt == original.global_best_prompt
        assert len(restored.experiment_history) == 2
        assert len(restored.captions) == 3


# ---------------------------------------------------------------------------
# Tests for save_state / load_state (file I/O)
# ---------------------------------------------------------------------------


class TestSaveLoadState:
    """Test the public save_state() / load_state() round-trip using real files."""

    def test_basic_round_trip(self, tmp_path: Path) -> None:
        state = make_loop_state(global_best_metrics=make_aggregated_metrics())
        state_file = tmp_path / "state.json"

        save_state(state, state_file)
        assert state_file.exists()

        loaded = load_state(state_file)
        assert loaded is not None
        assert loaded.iteration == state.iteration
        assert loaded.global_best_prompt == state.global_best_prompt
        assert len(loaded.experiment_history) == len(state.experiment_history)
        assert len(loaded.captions) == len(state.captions)

    def test_full_fidelity_round_trip(self, tmp_path: Path) -> None:
        """Every field survives save -> load, including nested dataclasses."""
        state = make_loop_state(
            iteration=12,
            converged=True,
            convergence_reason=ConvergenceReason.MAX_ITERATIONS,
            global_best_metrics=make_aggregated_metrics(seed=4.0),
        )
        state_file = tmp_path / "subdir" / "state.json"

        save_state(state, state_file)
        loaded = load_state(state_file)

        assert loaded is not None
        assert loaded.iteration == 12
        assert loaded.converged is True
        assert loaded.convergence_reason == ConvergenceReason.MAX_ITERATIONS
        assert loaded.global_best_metrics == state.global_best_metrics
        assert loaded.style_profile == state.style_profile
        assert loaded.best_metrics == state.best_metrics
        assert loaded.plateau_counter == state.plateau_counter

        # Verify experiment history fidelity
        assert len(loaded.experiment_history) == len(state.experiment_history)
        for orig_h, load_h in zip(state.experiment_history, loaded.experiment_history, strict=True):
            assert load_h.per_image_scores == orig_h.per_image_scores
            assert load_h.image_paths == orig_h.image_paths

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        state = make_loop_state(global_best_metrics=make_aggregated_metrics())
        deep_path = tmp_path / "a" / "b" / "c" / "state.json"

        save_state(state, deep_path)
        assert deep_path.exists()

    def test_output_is_valid_json(self, tmp_path: Path) -> None:
        state = make_loop_state(global_best_metrics=make_aggregated_metrics())
        state_file = tmp_path / "state.json"
        save_state(state, state_file)

        raw = json.loads(state_file.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        assert "iteration" in raw
        assert "current_template" in raw

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        assert load_state(tmp_path / "does_not_exist.json") is None

    def test_none_best_metrics_survives(self, tmp_path: Path) -> None:
        state = make_loop_state(global_best_metrics=None)
        state.best_metrics = None
        state_file = tmp_path / "state.json"

        save_state(state, state_file)
        loaded = load_state(state_file)

        assert loaded is not None
        assert loaded.global_best_metrics is None
        assert loaded.best_metrics is None

    def test_convergence_reason_survives(self, tmp_path: Path) -> None:
        state = make_loop_state(convergence_reason=ConvergenceReason.REASONING_STOP)
        state_file = tmp_path / "state.json"

        save_state(state, state_file)
        loaded = load_state(state_file)

        assert loaded is not None
        assert loaded.convergence_reason == ConvergenceReason.REASONING_STOP


# ---------------------------------------------------------------------------
# Tests for save_iteration_log
# ---------------------------------------------------------------------------


class TestSaveIterationLog:
    """Test that save_iteration_log writes correct filenames with valid JSON."""

    def test_writes_correct_filename(self, tmp_path: Path) -> None:
        result = make_iteration_result(branch_id=2, iteration=7)
        save_iteration_log(result, tmp_path)

        expected = tmp_path / "iter_007_branch_2.json"
        assert expected.exists()

    def test_output_is_valid_json(self, tmp_path: Path) -> None:
        result = make_iteration_result(branch_id=0, iteration=1)
        save_iteration_log(result, tmp_path)

        log_file = tmp_path / "iter_001_branch_0.json"
        data = json.loads(log_file.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert data["branch_id"] == 0
        assert data["iteration"] == 1
        assert data["kept"] is True
        assert isinstance(data["per_image_scores"], list)

    def test_creates_log_dir_if_absent(self, tmp_path: Path) -> None:
        result = make_iteration_result()
        log_dir = tmp_path / "nested" / "logs"
        save_iteration_log(result, log_dir)

        assert (log_dir / "iter_001_branch_0.json").exists()

    def test_filename_zero_padded(self, tmp_path: Path) -> None:
        for it in (1, 10, 100):
            result = make_iteration_result(branch_id=0, iteration=it)
            save_iteration_log(result, tmp_path)

        assert (tmp_path / "iter_001_branch_0.json").exists()
        assert (tmp_path / "iter_010_branch_0.json").exists()
        assert (tmp_path / "iter_100_branch_0.json").exists()

    def test_content_matches_iteration_result(self, tmp_path: Path) -> None:
        result = make_iteration_result(branch_id=1, iteration=3)
        save_iteration_log(result, tmp_path)

        data = json.loads((tmp_path / "iter_003_branch_1.json").read_text(encoding="utf-8"))
        assert data["rendered_prompt"] == result.rendered_prompt
        assert data["claude_analysis"] == result.claude_analysis
        assert data["template_changes"] == result.template_changes
        assert len(data["image_paths"]) == len(result.image_paths)
        assert len(data["per_image_scores"]) == len(result.per_image_scores)


# ---------------------------------------------------------------------------
# Tests for KnowledgeBase serialization
# ---------------------------------------------------------------------------


def _make_knowledge_base() -> KnowledgeBase:
    """Build a KnowledgeBase with hypotheses, categories, and open problems."""
    kb = KnowledgeBase()
    kb.add_hypothesis(
        iteration=1,
        parent_id=None,
        statement="Color accuracy gap",
        experiment="Add hex codes",
        category="color_palette",
        kept=True,
        metric_delta={"dreamsim": 0.03},
        lesson="Hex codes improve color matching",
        confirmed="Hex codes improve color matching",
        rejected="",
    )
    kb.add_hypothesis(
        iteration=2,
        parent_id="H1",
        statement="Color temperature helps further",
        experiment="Add warm/cool descriptors",
        category="color_palette",
        kept=False,
        metric_delta={"dreamsim": -0.005},
        lesson="Temperature terms not followed",
        confirmed="",
        rejected="Temperature terms ignored by generator",
    )
    kb.open_problems = [
        OpenProblem(
            text="Texture detail in backgrounds",
            category="texture",
            priority="HIGH",
            metric_gap=0.12,
            since_iteration=1,
        ),
    ]
    return kb


class TestKnowledgeBaseSerialization:
    """Test round-trip serialization for KnowledgeBase and its parts."""

    def test_knowledge_base_round_trip(self) -> None:
        kb = _make_knowledge_base()
        d = json.loads(json.dumps(to_dict(kb)))
        restored = _knowledge_base_from_dict(d)

        assert len(restored.hypotheses) == 2
        assert restored.hypotheses[0].id == "H1"
        assert restored.hypotheses[0].parent_id is None
        assert restored.hypotheses[1].id == "H2"
        assert restored.hypotheses[1].parent_id == "H1"
        assert restored.next_id == 3

        assert "color_palette" in restored.categories
        cat = restored.categories["color_palette"]
        assert "Hex codes improve color matching" in cat.confirmed_insights
        assert len(cat.hypothesis_ids) == 2

        assert len(restored.open_problems) == 1
        assert restored.open_problems[0].priority == "HIGH"
        assert restored.open_problems[0].metric_gap == 0.12

    def test_loop_state_with_kb_round_trip(self, tmp_path: Path) -> None:
        state = make_loop_state(global_best_metrics=make_aggregated_metrics())
        state.knowledge_base = _make_knowledge_base()

        state_file = tmp_path / "state.json"
        save_state(state, state_file)
        loaded = load_state(state_file)

        assert loaded is not None
        kb = loaded.knowledge_base
        assert len(kb.hypotheses) == 2
        assert kb.hypotheses[0].statement == "Color accuracy gap"
        assert len(kb.open_problems) == 1

    def test_backward_compat_no_kb_field(self) -> None:
        """An old state.json without a knowledge_base key should load with an empty KB."""
        loop_state_dict = {
            "iteration": 0,
            "current_template": {"sections": [], "negative_prompt": None},
            "best_template": {"sections": [], "negative_prompt": None},
            "best_metrics": None,
            "captions": [],
            "style_profile": {
                "color_palette": "",
                "composition": "",
                "technique": "",
                "mood_atmosphere": "",
                "subject_matter": "",
                "influences": "",
                "gemini_raw_analysis": "",
                "claude_raw_analysis": "",
            },
            # No "knowledge_base" key!
        }
        state = _loop_state_from_dict(loop_state_dict)
        assert state.knowledge_base is not None
        assert len(state.knowledge_base.hypotheses) == 0
        assert state.knowledge_base.next_id == 1

    def test_empty_kb_round_trip(self) -> None:
        kb = KnowledgeBase()
        d = json.loads(json.dumps(to_dict(kb)))
        restored = _knowledge_base_from_dict(d)
        assert restored.hypotheses == []
        assert restored.categories == {}
        assert restored.open_problems == []
        assert restored.next_id == 1


# ---------------------------------------------------------------------------
# Tests for RunManifest save/load
# ---------------------------------------------------------------------------


class TestManifestRoundTrip:
    """Test save_manifest / load_manifest round-trip."""

    def test_save_and_load_manifest(self, tmp_path: Path) -> None:
        manifest = RunManifest(
            protocol_version="rigorous_v1",
            seed=42,
            cli_args={"num_fixed_refs": 20, "max_iterations": 50},
            model_names={
                "caption_model": "gemini-pro",
                "generator_model": "gemini-flash",
                "reasoning_model": "claude-opus",
            },
            reasoning_provider="anthropic",
            git_sha="abc1234",
            python_version="3.12.0",
            platform="darwin",
            timestamp_utc="2026-04-10T12:00:00Z",
            reference_image_hashes={"img_001.png": "deadbeef", "img_002.png": "cafebabe"},
            num_fixed_refs=20,
            uv_lock_hash="lockhash123",
        )
        manifest_path = tmp_path / "manifest.json"

        save_manifest(manifest, manifest_path)
        assert manifest_path.exists()

        loaded = load_manifest(manifest_path)
        assert loaded is not None
        assert loaded.protocol_version == manifest.protocol_version
        assert loaded.seed == manifest.seed
        assert loaded.cli_args == manifest.cli_args
        assert loaded.model_names == manifest.model_names
        assert loaded.reasoning_provider == manifest.reasoning_provider
        assert loaded.git_sha == manifest.git_sha
        assert loaded.python_version == manifest.python_version
        assert loaded.platform == manifest.platform
        assert loaded.timestamp_utc == manifest.timestamp_utc
        assert loaded.reference_image_hashes == manifest.reference_image_hashes
        assert loaded.num_fixed_refs == manifest.num_fixed_refs
        assert loaded.uv_lock_hash == manifest.uv_lock_hash


# ---------------------------------------------------------------------------
# Tests for PromotionDecision append/load log
# ---------------------------------------------------------------------------


class TestPromotionLog:
    """Test append_promotion_log / load_promotion_log round-trip."""

    def test_append_and_load(self, tmp_path: Path) -> None:
        log_path = tmp_path / "promotions.jsonl"

        d1 = PromotionDecision(
            iteration=1,
            candidate_score=0.65,
            baseline_score=0.60,
            epsilon=0.01,
            delta=0.05,
            decision="promoted",
            reason="Candidate exceeded baseline + epsilon",
            candidate_branch_id=0,
            candidate_hypothesis="Add hex color codes",
        )
        d2 = PromotionDecision(
            iteration=2,
            candidate_score=0.62,
            baseline_score=0.65,
            epsilon=0.01,
            delta=-0.03,
            decision="rejected",
            reason="Candidate below baseline",
            candidate_branch_id=1,
            candidate_hypothesis="Increase brush detail",
            replicate_scores=[0.61, 0.63, 0.62],
            p_value=0.35,
            test_statistic=1.2,
        )

        append_promotion_log(d1, log_path)
        append_promotion_log(d2, log_path)

        loaded = load_promotion_log(log_path)
        assert len(loaded) == 2

        assert loaded[0].iteration == 1
        assert loaded[0].decision == "promoted"
        assert loaded[0].candidate_score == 0.65
        assert loaded[0].candidate_hypothesis == "Add hex color codes"
        assert loaded[0].replicate_scores is None
        assert loaded[0].p_value is None

        assert loaded[1].iteration == 2
        assert loaded[1].decision == "rejected"
        assert loaded[1].candidate_score == 0.62
        assert loaded[1].candidate_hypothesis == "Increase brush detail"
        assert loaded[1].replicate_scores == [0.61, 0.63, 0.62]
        assert loaded[1].p_value == 0.35
        assert loaded[1].test_statistic == 1.2


# ---------------------------------------------------------------------------
# Tests for LoopState new scientific rigor fields
# ---------------------------------------------------------------------------


class TestLoopStateNewFields:
    """Test that seed, protocol, feedback_refs, silent_refs survive save/load."""

    def test_round_trip_with_new_fields(self, tmp_path: Path) -> None:
        state = make_loop_state(global_best_metrics=make_aggregated_metrics())
        state.seed = 12345
        state.protocol = "rigorous"
        state.feedback_refs = [Path("/data/refs/img_001.png"), Path("/data/refs/img_002.png")]
        state.silent_refs = [Path("/data/refs/img_003.png")]

        state_file = tmp_path / "state.json"
        save_state(state, state_file)
        loaded = load_state(state_file)

        assert loaded is not None
        assert loaded.seed == 12345
        assert loaded.protocol == "rigorous"
        assert loaded.feedback_refs == [Path("/data/refs/img_001.png"), Path("/data/refs/img_002.png")]
        assert loaded.silent_refs == [Path("/data/refs/img_003.png")]
