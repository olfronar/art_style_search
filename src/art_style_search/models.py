"""Lazily-loaded model registry for evaluation metrics.

Provides thread-safe access to DINOv2, LPIPS, HPS v2, and LAION Aesthetics
models.  Each model has its own ``threading.Lock`` so that different models can
run concurrently while preventing concurrent forward passes on the same model.

All inference methods are synchronous — the async loop is expected to call them
via ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


def _auto_device() -> torch.device:
    """Pick the best available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ModelRegistry:
    """Lazy-loading registry for the four evaluation models.

    Use the ``load_all`` class method to construct an instance.  Individual
    models are loaded on first access so that startup is fast when only a
    subset of metrics is needed.
    """

    device: torch.device = field(default_factory=_auto_device)

    # Private lazy-init state — populated on first use via properties.
    _dino_model: torch.nn.Module | None = field(default=None, init=False, repr=False)
    _dino_processor: object | None = field(default=None, init=False, repr=False)
    _dino_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    _lpips_model: torch.nn.Module | None = field(default=None, init=False, repr=False)
    _lpips_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    _hps_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    _aesthetics_model: torch.nn.Module | None = field(default=None, init=False, repr=False)
    _aesthetics_processor: object | None = field(default=None, init=False, repr=False)
    _aesthetics_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load_all(cls, device: str = "auto") -> ModelRegistry:
        """Create a registry, optionally pinning to a specific device.

        Models themselves are loaded lazily on first use; this method only
        resolves the device.
        """
        resolved = _auto_device() if device == "auto" else torch.device(device)
        log.info("ModelRegistry using device: %s", resolved)
        return cls(device=resolved)

    # ------------------------------------------------------------------
    # Lazy loaders (called under the relevant lock)
    # ------------------------------------------------------------------

    def _ensure_dino(self) -> None:
        if self._dino_model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        model_name = "facebook/dinov2-base"
        log.info("Loading DINOv2 (%s) ...", model_name)
        self._dino_processor = AutoImageProcessor.from_pretrained(model_name)
        self._dino_model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def _ensure_lpips(self) -> None:
        if self._lpips_model is not None:
            return
        import lpips

        log.info("Loading LPIPS (alex) ...")
        self._lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()

    def _ensure_aesthetics(self) -> None:
        if self._aesthetics_model is not None:
            return
        from aesthetics_predictor import AestheticsPredictorV2Linear
        from transformers import CLIPProcessor

        model_name = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
        clip_name = "openai/clip-vit-large-patch14"
        log.info("Loading LAION Aesthetics (%s) ...", model_name)
        self._aesthetics_processor = CLIPProcessor.from_pretrained(clip_name)
        self._aesthetics_model = AestheticsPredictorV2Linear.from_pretrained(model_name).to(self.device).eval()

    # ------------------------------------------------------------------
    # Public compute methods
    # ------------------------------------------------------------------

    def compute_dino(self, generated: Image.Image, references: list[Image.Image]) -> float:
        """Cosine similarity of DINOv2 CLS embeddings (generated vs mean reference).

        Returns a float in [-1, 1]; higher is better.
        """
        with self._dino_lock:
            self._ensure_dino()
            assert self._dino_model is not None
            assert self._dino_processor is not None

            with torch.no_grad():
                # Embed the generated image.
                gen_inputs = self._dino_processor(images=generated, return_tensors="pt").to(self.device)
                gen_cls = self._dino_model(**gen_inputs).last_hidden_state[:, 0]  # (1, D)

                # Embed each reference and compute mean embedding.
                ref_embeddings: list[torch.Tensor] = []
                for ref in references:
                    ref_inputs = self._dino_processor(images=ref, return_tensors="pt").to(self.device)
                    ref_cls = self._dino_model(**ref_inputs).last_hidden_state[:, 0]  # (1, D)
                    ref_embeddings.append(ref_cls)

                mean_ref = torch.cat(ref_embeddings, dim=0).mean(dim=0, keepdim=True)  # (1, D)

                similarity = torch.nn.functional.cosine_similarity(gen_cls, mean_ref, dim=-1)  # (1,)
                return similarity.item()

    def compute_lpips(self, generated: Image.Image, references: list[Image.Image]) -> float:
        """Mean LPIPS perceptual distance (generated vs each reference).

        Returns a float >= 0; lower is better.
        """
        with self._lpips_lock:
            self._ensure_lpips()
            assert self._lpips_model is not None

            with torch.no_grad():
                gen_tensor = self._pil_to_lpips_tensor(generated)

                distances: list[float] = []
                for ref in references:
                    ref_tensor = self._pil_to_lpips_tensor(ref)
                    dist = self._lpips_model(gen_tensor, ref_tensor)
                    distances.append(dist.item())

                return float(np.mean(distances))

    def compute_hps(self, generated: Image.Image, prompt: str) -> float:
        """HPS v2 score for the generated image against the prompt.

        Returns a float; higher is better.
        """
        with self._hps_lock:
            import hpsv2

            with torch.no_grad():
                scores = hpsv2.score(generated, prompt)
                return float(scores[0])

    def compute_aesthetics(self, generated: Image.Image) -> float:
        """LAION Aesthetics v2 score (1-10 scale).

        Returns a float; higher is better.
        """
        with self._aesthetics_lock:
            self._ensure_aesthetics()
            assert self._aesthetics_model is not None
            assert self._aesthetics_processor is not None

            with torch.no_grad():
                inputs = self._aesthetics_processor(images=generated, return_tensors="pt").to(self.device)
                # AestheticsPredictorV2Linear outputs logits directly.
                output = self._aesthetics_model(**inputs)
                score = output.logits.squeeze().item()
                return float(score)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pil_to_lpips_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a [-1, 1] tensor suitable for LPIPS.

        LPIPS expects (N, 3, H, W) tensors normalised to [-1, 1].
        We resize to 256x256 to keep memory usage constant.
        """
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),  # -> [0, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
            ]
        )
        img_rgb = img.convert("RGB")
        tensor: torch.Tensor = transform(img_rgb).unsqueeze(0).to(self.device)  # (1, 3, 256, 256)
        return tensor
