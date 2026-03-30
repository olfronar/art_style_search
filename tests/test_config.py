"""Unit tests for art_style_search.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from art_style_search.config import Config, parse_args


class TestParseArgsWithAllRequired:
    """parse_args with valid required arguments."""

    def test_returns_config(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        cfg = parse_args(
            [
                "--reference-dir",
                str(ref_dir),
                "--anthropic-api-key",
                "sk-ant-fake",
                "--google-api-key",
                "gk-fake",
                "--output-dir",
                str(tmp_path / "out"),
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        assert isinstance(cfg, Config)

    def test_api_keys_from_args(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        cfg = parse_args(
            [
                "--reference-dir",
                str(ref_dir),
                "--anthropic-api-key",
                "sk-ant-test-123",
                "--google-api-key",
                "gk-test-456",
                "--output-dir",
                str(tmp_path / "out"),
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        assert cfg.anthropic_api_key == "sk-ant-test-123"
        assert cfg.google_api_key == "gk-test-456"

    def test_paths_are_path_objects(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        cfg = parse_args(
            [
                "--reference-dir",
                str(ref_dir),
                "--anthropic-api-key",
                "sk-ant-fake",
                "--google-api-key",
                "gk-fake",
                "--output-dir",
                str(tmp_path / "out"),
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        assert isinstance(cfg.reference_dir, Path)
        assert isinstance(cfg.output_dir, Path)
        assert isinstance(cfg.log_dir, Path)
        assert isinstance(cfg.state_file, Path)

    def test_output_and_log_dirs_created(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        out_dir = tmp_path / "generated_out"
        log_dir = tmp_path / "generated_logs"
        parse_args(
            [
                "--reference-dir",
                str(ref_dir),
                "--anthropic-api-key",
                "sk-ant-fake",
                "--google-api-key",
                "gk-fake",
                "--output-dir",
                str(out_dir),
                "--log-dir",
                str(log_dir),
            ]
        )
        assert out_dir.is_dir()
        assert log_dir.is_dir()

    def test_custom_loop_params(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        cfg = parse_args(
            [
                "--reference-dir",
                str(ref_dir),
                "--anthropic-api-key",
                "sk-ant-fake",
                "--google-api-key",
                "gk-fake",
                "--output-dir",
                str(tmp_path / "out"),
                "--log-dir",
                str(tmp_path / "logs"),
                "--max-iterations",
                "50",
                "--plateau-window",
                "10",
                "--num-branches",
                "5",
                "--num-images",
                "8",
                "--aspect-ratio",
                "16:9",
            ]
        )
        assert cfg.max_iterations == 50
        assert cfg.plateau_window == 10
        assert cfg.num_branches == 5
        assert cfg.num_images == 8
        assert cfg.aspect_ratio == "16:9"


class TestDefaults:
    """Verify default values when only required args are supplied."""

    @pytest.fixture()
    def cfg(self, tmp_path: Path) -> Config:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        return parse_args(
            [
                "--reference-dir",
                str(ref_dir),
                "--anthropic-api-key",
                "sk-ant-fake",
                "--google-api-key",
                "gk-fake",
                "--output-dir",
                str(tmp_path / "out"),
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )

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

    def test_state_file(self, cfg: Config) -> None:
        assert cfg.state_file == Path("state.json")


class TestApiKeysFromEnv:
    """API keys can be supplied via environment variables."""

    def test_anthropic_key_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-anthropic")
        monkeypatch.setenv("GOOGLE_API_KEY", "gk-env-google")
        cfg = parse_args(
            [
                "--reference-dir",
                str(ref_dir),
                "--output-dir",
                str(tmp_path / "out"),
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
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
                "--anthropic-api-key",
                "sk-cli-new",
                "--google-api-key",
                "gk-cli-new",
                "--output-dir",
                str(tmp_path / "out"),
                "--log-dir",
                str(tmp_path / "logs"),
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
                    "--google-api-key",
                    "gk-fake",
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--log-dir",
                    str(tmp_path / "logs"),
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
                    "--anthropic-api-key",
                    "sk-ant-fake",
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--log-dir",
                    str(tmp_path / "logs"),
                ]
            )

    def test_missing_both_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--reference-dir",
                    str(ref_dir),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--log-dir",
                    str(tmp_path / "logs"),
                ]
            )


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
                    "--anthropic-api-key",
                    "sk-ant-fake",
                    "--google-api-key",
                    "gk-fake",
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--log-dir",
                    str(tmp_path / "logs"),
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
                    "--anthropic-api-key",
                    "sk-ant-fake",
                    "--google-api-key",
                    "gk-fake",
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--log-dir",
                    str(tmp_path / "logs"),
                ]
            )
