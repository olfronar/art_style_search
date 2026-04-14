from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from art_style_search import hps


class _FakeModel:
    def __init__(self) -> None:
        self.loaded_state: dict | None = None
        self.device: str | None = None
        self.eval_called = False

    def load_state_dict(self, state: dict) -> None:
        self.loaded_state = state

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def __call__(self, image: torch.Tensor, text: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "image_features": torch.tensor([[0.21, 0.0]], dtype=torch.float32),
            "text_features": torch.tensor([[2.0, 0.0]], dtype=torch.float32),
        }


@pytest.fixture(autouse=True)
def _reset_hps_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    hps._load_artifacts.cache_clear()
    monkeypatch.setattr(hps.torch.backends.mps, "is_available", lambda: False)


def test_score_uses_cached_local_hps_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _FakeModel()
    download_calls: list[tuple[str, str]] = []

    def fake_create_model_and_transforms(*args, **kwargs):
        return fake_model, None, lambda image: torch.tensor([1.0], dtype=torch.float32)

    def fake_get_tokenizer(model_name: str):
        assert model_name == "ViT-H-14"
        return lambda prompts: torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        download_calls.append((repo_id, filename))
        return "/tmp/hps-checkpoint.pt"

    monkeypatch.setattr("art_style_search.hps.open_clip.create_model_and_transforms", fake_create_model_and_transforms)
    monkeypatch.setattr("art_style_search.hps.open_clip.get_tokenizer", fake_get_tokenizer)
    monkeypatch.setattr("art_style_search.hps.huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        "art_style_search.hps.torch.load",
        lambda path, map_location, weights_only: {"state_dict": {"weights": 1}},
    )
    monkeypatch.setattr("art_style_search.hps.torch.cuda.is_available", lambda: False)

    image = Image.new("RGB", (4, 4), color=(12, 34, 56))
    first = hps.score(image, "prompt text")
    second = hps.score(image, "prompt text")

    assert first == [pytest.approx(0.42)]
    assert second == [pytest.approx(0.42)]
    assert fake_model.loaded_state == {"weights": 1}
    assert fake_model.device == "cpu"
    assert fake_model.eval_called is True
    assert download_calls == [("xswu/HPSv2", "HPS_v2_compressed.pt")]


def test_score_opens_paths_and_supports_image_lists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_model = _FakeModel()

    monkeypatch.setattr(
        "art_style_search.hps.open_clip.create_model_and_transforms",
        lambda *args, **kwargs: (fake_model, None, lambda image: torch.tensor([1.0], dtype=torch.float32)),
    )
    monkeypatch.setattr(
        "art_style_search.hps.open_clip.get_tokenizer",
        lambda model_name: lambda prompts: torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        "art_style_search.hps.huggingface_hub.hf_hub_download",
        lambda repo_id, filename: "/tmp/hps-checkpoint.pt",
    )
    monkeypatch.setattr(
        "art_style_search.hps.torch.load",
        lambda path, map_location, weights_only: {"state_dict": {}},
    )
    monkeypatch.setattr("art_style_search.hps.torch.cuda.is_available", lambda: False)

    image_path = tmp_path / "test.png"
    Image.new("RGB", (4, 4), color=(99, 88, 77)).save(image_path)

    scores = hps.score([image_path, Image.open(image_path)], "prompt text")

    assert scores == [pytest.approx(0.42), pytest.approx(0.42)]


def test_score_loads_checkpoint_with_weights_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _FakeModel()
    load_calls: list[tuple[str, str | None, bool | None]] = []

    monkeypatch.setattr(
        "art_style_search.hps.open_clip.create_model_and_transforms",
        lambda *args, **kwargs: (fake_model, None, lambda image: torch.tensor([1.0], dtype=torch.float32)),
    )
    monkeypatch.setattr(
        "art_style_search.hps.open_clip.get_tokenizer",
        lambda model_name: lambda prompts: torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        "art_style_search.hps.huggingface_hub.hf_hub_download",
        lambda repo_id, filename: "/tmp/hps-checkpoint.pt",
    )

    def fake_torch_load(path: str, *, map_location: str | None = None, weights_only: bool | None = None) -> dict:
        load_calls.append((path, map_location, weights_only))
        return {"state_dict": {}}

    monkeypatch.setattr("art_style_search.hps.torch.load", fake_torch_load)
    monkeypatch.setattr("art_style_search.hps.torch.cuda.is_available", lambda: False)

    hps.score(Image.new("RGB", (4, 4), color=(1, 2, 3)), "prompt text")

    assert load_calls == [("/tmp/hps-checkpoint.pt", "cpu", True)]


def test_score_rejects_unknown_hps_version() -> None:
    image = Image.new("RGB", (4, 4), color=(0, 0, 0))

    with pytest.raises(ValueError, match="Unsupported HPS version"):
        hps.score(image, "prompt text", hps_version="v9.9")
