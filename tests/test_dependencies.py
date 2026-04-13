from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def test_pyproject_uses_direct_hps_runtime_dependencies() -> None:
    pyproject = _load_toml(REPO_ROOT / "pyproject.toml")
    dependencies = pyproject["project"]["dependencies"]

    assert any(dependency.startswith("open-clip-torch") for dependency in dependencies)
    assert any(dependency.startswith("huggingface-hub") for dependency in dependencies)
    assert not any(dependency.startswith("hpsv2") for dependency in dependencies)
    assert not any(dependency.startswith("protobuf") for dependency in dependencies)


def test_uv_lock_has_no_hpsv2_or_protobuf() -> None:
    lock = _load_toml(REPO_ROOT / "uv.lock")
    package_names = {package["name"] for package in lock["package"]}

    assert "hpsv2" not in package_names
    assert "protobuf" not in package_names
