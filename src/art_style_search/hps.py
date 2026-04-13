"""Local HPS v2 scorer built on OpenCLIP and the published HPS checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import huggingface_hub
import open_clip
import torch
from PIL import Image

_HPS_REPO_ID = "xswu/HPSv2"
_HPS_CHECKPOINTS = {
    "v2.0": "HPS_v2_compressed.pt",
    "v2.1": "HPS_v2.1_compressed.pt",
}


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=len(_HPS_CHECKPOINTS) * 3)
def _load_artifacts(device: str, hps_version: str):
    try:
        checkpoint_name = _HPS_CHECKPOINTS[hps_version]
    except KeyError as exc:
        msg = f"Unsupported HPS version: {hps_version}"
        raise ValueError(msg) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained=None,
        precision="amp" if device.startswith("cuda") else "fp32",
        device=device,
        output_dict=True,
    )
    checkpoint_path = huggingface_hub.hf_hub_download(_HPS_REPO_ID, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    return model, preprocess, tokenizer


def _load_image(image_input: Image.Image | str | Path) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    with Image.open(image_input) as image:
        return image.convert("RGB")


def score(
    image_input: Image.Image | str | Path | list[Image.Image | str | Path],
    prompt: str,
    *,
    hps_version: str = "v2.0",
    device: str | None = None,
) -> list[float]:
    resolved_device = _resolve_device(device)
    model, preprocess, tokenizer = _load_artifacts(resolved_device, hps_version)
    autocast = (lambda: torch.amp.autocast("cuda")) if resolved_device.startswith("cuda") else nullcontext

    inputs = image_input if isinstance(image_input, list) else [image_input]
    text = tokenizer([prompt]).to(device=resolved_device, non_blocking=True)
    results: list[float] = []

    for item in inputs:
        image = preprocess(_load_image(item)).unsqueeze(0).to(device=resolved_device, non_blocking=True)
        with torch.no_grad(), autocast():
            outputs = model(image, text)
            logits_per_image = outputs["image_features"] @ outputs["text_features"].T
            results.extend(float(score) for score in torch.diagonal(logits_per_image).detach().cpu().tolist())

    return results
