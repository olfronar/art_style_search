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
        cfg = parse_args(
            _base_args(
                tmp_path,
                max_iterations="50",
                plateau_window="10",
                num_branches="5",
                num_images="8",
                aspect_ratio="16:9",
            )
        )
        assert cfg.max_iterations == 50
        assert cfg.plateau_window == 10
        assert cfg.num_branches == 5
        assert cfg.num_images == 8
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
        assert cfg.max_iterations == 20

    def test_plateau_window(self, cfg: Config) -> None:
        assert cfg.plateau_window == 5

    def test_num_branches(self, cfg: Config) -> None:
        assert cfg.num_branches == 5

    def test_num_images(self, cfg: Config) -> None:
        assert cfg.num_images == 4

    def test_aspect_ratio(self, cfg: Config) -> None:
        assert cfg.aspect_ratio == "1:1"

    def test_max_analysis_images(self, cfg: Config) -> None:
        assert cfg.max_analysis_images == 10

    def test_max_eval_images(self, cfg: Config) -> None:
        assert cfg.max_eval_images == 10

    def test_num_fixed_refs(self, cfg: Config) -> None:
        assert cfg.num_fixed_refs == 20

    def test_caption_model(self, cfg: Config) -> None:
        assert cfg.caption_model == "gemini-3.1-pro-preview"

    def test_generator_model(self, cfg: Config) -> None:
        assert cfg.generator_model == "gemini-3.1-flash-image-preview"

    def test_reasoning_model(self, cfg: Config) -> None:
        assert cfg.reasoning_model == "claude-opus-4-6"
        assert cfg.reasoning_provider == "anthropic"

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
