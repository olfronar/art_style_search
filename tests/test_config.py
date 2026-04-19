"""Unit tests for art_style_search.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from art_style_search.config import Config, parse_args


def _base_args(tmp_path: Path, ref_dir: Path | None = None, **overrides: str) -> list[str]:
    """Build a minimal valid arg list pointing at tmp_path for isolation."""
    if ref_dir is None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir(exist_ok=True)
    args = [
        "--reference-dir",
        str(ref_dir),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--anthropic-api-key",
        overrides.pop("anthropic_api_key", "sk-ant-fake"),
        "--google-api-key",
        overrides.pop("google_api_key", "gk-fake"),
    ]
    for k, v in overrides.items():
        args.extend([f"--{k.replace('_', '-')}", v])
    return args


class TestParseArgsWithAllRequired:
    """parse_args with valid required arguments."""

    def test_returns_config(self, tmp_path: Path) -> None:
        cfg = parse_args(_base_args(tmp_path))
        assert isinstance(cfg, Config)

    def test_api_keys_from_args(self, tmp_path: Path) -> None:
        cfg = parse_args(_base_args(tmp_path, anthropic_api_key="sk-ant-test-123", google_api_key="gk-test-456"))
        assert cfg.anthropic_api_key == "sk-ant-test-123"
        assert cfg.google_api_key == "gk-test-456"

    def test_paths_are_path_objects(self, tmp_path: Path) -> None:
        cfg = parse_args(_base_args(tmp_path))
        assert isinstance(cfg.reference_dir, Path)
        assert isinstance(cfg.output_dir, Path)
        assert isinstance(cfg.log_dir, Path)
        assert isinstance(cfg.state_file, Path)
        assert isinstance(cfg.run_dir, Path)

    def test_run_dirs_created(self, tmp_path: Path) -> None:
        cfg = parse_args(_base_args(tmp_path))
        assert cfg.output_dir.is_dir()
        assert cfg.log_dir.is_dir()
        assert cfg.run_dir.is_dir()

    def test_paths_inside_run_dir(self, tmp_path: Path) -> None:
        cfg = parse_args(_base_args(tmp_path))
        assert cfg.output_dir == cfg.run_dir / "outputs"
        assert cfg.log_dir == cfg.run_dir / "logs"
        assert cfg.state_file == cfg.run_dir / "state.json"

    def test_custom_loop_params(self, tmp_path: Path) -> None:
        # Classic protocol respects --max-iterations; short clamps to 3 (tested separately).
        cfg = parse_args(
            [
                *_base_args(
                    tmp_path,
                    max_iterations="50",
                    plateau_window="10",
                    num_branches="5",
                    aspect_ratio="16:9",
                ),
                "--protocol",
                "classic",
            ]
        )
        assert cfg.max_iterations == 50
        assert cfg.plateau_window == 10
        assert cfg.num_branches == 5
        assert cfg.aspect_ratio == "16:9"


class TestRunNaming:
    """Run name resolution in parse_args."""

    def test_auto_names_first_run(self, tmp_path: Path) -> None:
        cfg = parse_args(_base_args(tmp_path))
        assert cfg.run_dir.name == "run_001"

    def test_auto_increments(self, tmp_path: Path) -> None:
        (tmp_path / "runs" / "run_001").mkdir(parents=True)
        cfg = parse_args(_base_args(tmp_path))
        assert cfg.run_dir.name == "run_002"

    def test_explicit_name(self, tmp_path: Path) -> None:
        cfg = parse_args([*_base_args(tmp_path), "--run", "my-experiment"])
        assert cfg.run_dir.name == "my-experiment"
        assert cfg.run_dir == tmp_path / "runs" / "my-experiment"

    def test_new_flag_creates(self, tmp_path: Path) -> None:
        cfg = parse_args([*_base_args(tmp_path), "--run", "fresh", "--new"])
        assert cfg.run_dir.name == "fresh"

    def test_new_flag_errors_if_exists(self, tmp_path: Path) -> None:
        (tmp_path / "runs" / "taken").mkdir(parents=True)
        with pytest.raises(SystemExit):
            parse_args([*_base_args(tmp_path), "--run", "taken", "--new"])


class TestDefaults:
    """Verify default values when only required args are supplied."""

    @pytest.fixture()
    def cfg(self, tmp_path: Path) -> Config:
        return parse_args(_base_args(tmp_path))

    def test_max_iterations(self, cfg: Config) -> None:
        # Default protocol is `short` → hard-clamped to 3. Classic default is 5 (tested separately).
        assert cfg.max_iterations == 3

    def test_plateau_window(self, cfg: Config) -> None:
        # Tighter plateau window (was 5) — replicate-gated iterations past 5 offer little
        # marginal probability of a novel winner.
        assert cfg.plateau_window == 3

    def test_num_branches(self, cfg: Config) -> None:
        assert cfg.num_branches == 9

    def test_raw_proposals(self, cfg: Config) -> None:
        assert cfg.raw_proposals == 9

    def test_aspect_ratio(self, cfg: Config) -> None:
        assert cfg.aspect_ratio == "1:1"

    def test_num_fixed_refs(self, cfg: Config) -> None:
        assert cfg.num_fixed_refs == 20

    def test_caption_model(self, cfg: Config) -> None:
        assert cfg.caption_model == "gemini-3.1-pro-preview"

    def test_generator_model(self, cfg: Config) -> None:
        assert cfg.generator_model == "gemini-3.1-flash-image-preview"

    def test_reasoning_model(self, cfg: Config) -> None:
        assert cfg.reasoning_model == "claude-opus-4-7"
        assert cfg.reasoning_provider == "anthropic"

    def test_comparison_model(self, cfg: Config) -> None:
        assert cfg.comparison_provider == "gemini"
        assert cfg.comparison_model == cfg.caption_model

    def test_gemini_concurrency(self, cfg: Config) -> None:
        assert cfg.gemini_concurrency == 50

    def test_eval_concurrency(self, cfg: Config) -> None:
        assert cfg.eval_concurrency == 4


class TestApiKeysFromEnv:
    """API keys can be supplied via environment variables."""

    def test_anthropic_key_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-anthropic")
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-env-google")
        cfg = parse_args(["--reference-dir", str(ref_dir), "--runs-dir", str(tmp_path / "runs")])
        assert cfg.anthropic_api_key == "sk-env-anthropic"
        assert cfg.google_api_key == "gk-env-google"

    def test_cli_arg_overrides_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-old")
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-env-old")
        cfg = parse_args(
            [
                "--reference-dir",
                str(ref_dir),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--anthropic-api-key",
                "sk-cli-new",
                "--google-api-key",
                "gk-cli-new",
            ]
        )
        assert cfg.anthropic_api_key == "sk-cli-new"
        assert cfg.google_api_key == "gk-cli-new"

    def test_xai_key_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-anthropic")
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-env-google")
        monkeypatch.setenv("XAI_API_KEY", "xai-env-key")
        cfg = parse_args(
            [
                "--reference-dir",
                str(ref_dir),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--comparison-provider",
                "xai",
            ]
        )
        assert cfg.xai_api_key == "xai-env-key"


class TestMissingApiKey:
    """Missing API keys must cause SystemExit."""

    def test_missing_anthropic_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--reference-dir",
                    str(ref_dir),
                    "--runs-dir",
                    str(tmp_path / "runs"),
                    "--google-api-key",
                    "gk-fake",
                ]
            )

    def test_missing_google_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--reference-dir",
                    str(ref_dir),
                    "--runs-dir",
                    str(tmp_path / "runs"),
                    "--anthropic-api-key",
                    "sk-ant-fake",
                ]
            )

    def test_missing_both_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            parse_args(["--reference-dir", str(ref_dir), "--runs-dir", str(tmp_path / "runs")])

    def test_missing_xai_key_for_reasoning_provider(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            parse_args(
                [
                    *_base_args(tmp_path, ref_dir=ref_dir),
                    "--reasoning-provider",
                    "xai",
                ]
            )

    def test_missing_xai_key_for_comparison_provider(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            parse_args(
                [
                    *_base_args(tmp_path, ref_dir=ref_dir),
                    "--comparison-provider",
                    "xai",
                ]
            )


class TestXAIProviders:
    def test_xai_reasoning_defaults_to_grok_4_20_reasoning_latest(self, tmp_path: Path) -> None:
        cfg = parse_args(
            _base_args(
                tmp_path,
                reasoning_provider="xai",
                xai_api_key="xai-test-key",
            )
        )
        assert cfg.reasoning_provider == "xai"
        assert cfg.reasoning_model == "grok-4.20-reasoning-latest"

    def test_xai_comparison_defaults_to_grok_4_20_reasoning_latest(self, tmp_path: Path) -> None:
        cfg = parse_args(
            _base_args(
                tmp_path,
                comparison_provider="xai",
                xai_api_key="xai-test-key",
            )
        )
        assert cfg.comparison_provider == "xai"
        assert cfg.comparison_model == "grok-4.20-reasoning-latest"

    def test_gemini_comparison_defaults_to_caption_model(self, tmp_path: Path) -> None:
        cfg = parse_args(_base_args(tmp_path))
        assert cfg.comparison_provider == "gemini"
        assert cfg.comparison_model == cfg.caption_model


class TestReasoningEffort:
    """--reasoning-effort flag plumbing."""

    def test_defaults_to_medium(self, tmp_path: Path) -> None:
        cfg = parse_args(_base_args(tmp_path))
        assert cfg.reasoning_effort == "medium"

    def test_override_low(self, tmp_path: Path) -> None:
        cfg = parse_args(_base_args(tmp_path, reasoning_effort="low"))
        assert cfg.reasoning_effort == "low"

    def test_override_high(self, tmp_path: Path) -> None:
        cfg = parse_args(_base_args(tmp_path, reasoning_effort="high"))
        assert cfg.reasoning_effort == "high"

    def test_rejects_invalid(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            parse_args(_base_args(tmp_path, reasoning_effort="extreme"))


class TestNonExistentReferenceDir:
    """Non-existent reference directory must cause SystemExit."""

    def test_raises_system_exit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        bogus_dir = tmp_path / "does_not_exist"
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--reference-dir",
                    str(bogus_dir),
                    "--runs-dir",
                    str(tmp_path / "runs"),
                    "--anthropic-api-key",
                    "sk-ant-fake",
                    "--google-api-key",
                    "gk-fake",
                ]
            )

    def test_file_not_dir_raises_system_exit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        not_a_dir = tmp_path / "a_file.txt"
        not_a_dir.write_text("I am a file")
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--reference-dir",
                    str(not_a_dir),
                    "--runs-dir",
                    str(tmp_path / "runs"),
                    "--anthropic-api-key",
                    "sk-ant-fake",
                    "--google-api-key",
                    "gk-fake",
                ]
            )
