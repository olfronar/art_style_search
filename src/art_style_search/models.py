"""Lazily-loaded model registry for evaluation metrics.

Provides thread-safe access to DreamSim, HPS v2, and LAION Aesthetics
models.  Each model has its own ``threading.Lock`` so that different models can
run concurrently while preventing concurrent forward passes on the same model.

All inference methods are synchronous — the async loop is expected to call them
via ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from .hps import score as hps_score

logger = logging.getLogger(__name__)


def _auto_device() -> torch.device:
    """Pick the best available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ModelRegistry:
    """Lazy-loading registry for evaluation models.

    Use the ``load_all`` class method to construct an instance.  Individual
    models are loaded on first access so that startup is fast when only a
    subset of metrics is needed.
    """

    device: torch.device = field(default_factory=_auto_device)

    # Private lazy-init state — populated on first use via properties.
    _dreamsim_model: Any = field(default=None, init=False, repr=False)
    _dreamsim_preprocess: Any = field(default=None, init=False, repr=False)
    _dreamsim_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    _hps_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    _aesthetics_model: torch.nn.Module | None = field(default=None, init=False, repr=False)
    _aesthetics_processor: Any = field(default=None, init=False, repr=False)
    _aesthetics_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    _megastyle_encoder: torch.nn.Module | None = field(default=None, init=False, repr=False)
    _megastyle_processor: Any = field(default=None, init=False, repr=False)
    _megastyle_ref_cache: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _megastyle_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

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
        logger.info("ModelRegistry using device: %s", resolved)
        return cls(device=resolved)

    # ------------------------------------------------------------------
    # Lazy loaders (called under the relevant lock)
    # ------------------------------------------------------------------

    def _ensure_dreamsim(self) -> None:
        if self._dreamsim_model is not None:
            return
        from dreamsim import dreamsim

        logger.info("Loading DreamSim (dino_vitb16) ...")
        model, preprocess = dreamsim(pretrained=True, dreamsim_type="dino_vitb16", device=str(self.device))
        model.eval()
        self._dreamsim_model = model
        self._dreamsim_preprocess = preprocess

    def _ensure_megastyle(self) -> None:
        if self._megastyle_encoder is not None:
            return
        from huggingface_hub import hf_hub_download
        from transformers import AutoImageProcessor, SiglipVisionModel

        logger.info("Loading MegaStyle-Encoder (SigLIP SoViT-400M + Gaojunyao/MegaStyle fine-tune) ...")
        self._megastyle_processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        encoder = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        ckpt_path = hf_hub_download(repo_id="Gaojunyao/MegaStyle", filename="megastyle_encoder.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        encoder.load_state_dict(state, strict=False)
        self._megastyle_encoder = encoder.eval().to(self.device)  # type: ignore[arg-type]

    def _embed_megastyle(self, image: Image.Image) -> torch.Tensor:
        if self._megastyle_encoder is None or self._megastyle_processor is None:
            raise RuntimeError("MegaStyle-Encoder failed to load")
        inputs = self._megastyle_processor(images=[image.convert("RGB")], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        # SigLIP's default fp16 path breaks the pooler matmul on MPS; force fp32 there.
        if self.device.type == "mps":
            pixel_values = pixel_values.to(torch.float32)
        with torch.no_grad():
            out = self._megastyle_encoder(pixel_values=pixel_values)
        emb = out.pooler_output
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.squeeze(0).float().cpu()

    def _ensure_aesthetics(self) -> None:
        if self._aesthetics_model is not None:
            return
        from aesthetics_predictor import AestheticsPredictorV2Linear
        from transformers import CLIPProcessor

        model_name = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
        clip_name = "openai/clip-vit-large-patch14"
        logger.info("Loading LAION Aesthetics (%s) ...", model_name)
        self._aesthetics_processor = CLIPProcessor.from_pretrained(clip_name)
        self._aesthetics_model = (
            AestheticsPredictorV2Linear.from_pretrained(model_name).to(self.device).eval()  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Public compute methods
    # ------------------------------------------------------------------

    def compute_dreamsim(self, generated: Image.Image, reference: Image.Image) -> float:
        """DreamSim perceptual similarity (generated vs single paired reference).

        Returns a float in [0, 1]; higher is better.  DreamSim returns a
        distance (lower = more similar), so we convert: 1 - clamp(dist, 0, 1).
        """
        with self._dreamsim_lock:
            self._ensure_dreamsim()
            if self._dreamsim_model is None or self._dreamsim_preprocess is None:
                raise RuntimeError("DreamSim model failed to load")

            with torch.no_grad():
                gen_tensor = self._dreamsim_preprocess(generated.convert("RGB")).to(self.device)
                ref_tensor = self._dreamsim_preprocess(reference.convert("RGB")).to(self.device)
                distance = self._dreamsim_model(gen_tensor, ref_tensor)
                dist_val = float(distance.item()) if hasattr(distance, "item") else float(distance)
                return max(0.0, min(1.0, 1.0 - dist_val))

    def compute_hps(self, generated: Image.Image, prompt: str) -> float:
        """HPS v2 score for the generated image against the prompt.

        Returns a float; higher is better.
        """
        with self._hps_lock, torch.no_grad():
            scores = hps_score(generated, prompt, device=str(self.device))
            return float(scores[0])

    def compute_aesthetics(self, generated: Image.Image) -> float:
        """LAION Aesthetics v2 score (1-10 scale).

        Returns a float; higher is better.
        """
        with self._aesthetics_lock:
            self._ensure_aesthetics()
            if self._aesthetics_model is None or self._aesthetics_processor is None:
                raise RuntimeError("Aesthetics model failed to load")

            with torch.no_grad():
                inputs = self._aesthetics_processor(images=generated, return_tensors="pt").to(self.device)
                # AestheticsPredictorV2Linear outputs logits directly.
                output = self._aesthetics_model(**inputs)
                score = output.logits.squeeze().item()
                return float(score)

    def compute_color_histogram(self, generated: Image.Image, reference: Image.Image) -> float:
        """Color histogram similarity in HSV space via histogram intersection.

        Returns a float in [0, 1]; higher is better.  Both images are resized
        to 256x256, converted to HSV, and compared channel-by-channel.
        """
        gen_hsv = np.array(generated.convert("HSV").resize((256, 256)))
        ref_hsv = np.array(reference.convert("HSV").resize((256, 256)))

        # Histogram intersection per channel, averaged
        similarities: list[float] = []
        for ch in range(3):
            bins = 64 if ch == 0 else 32  # more bins for hue
            gen_hist, _ = np.histogram(gen_hsv[:, :, ch].ravel(), bins=bins, range=(0, 256))
            ref_hist, _ = np.histogram(ref_hsv[:, :, ch].ravel(), bins=bins, range=(0, 256))
            # Normalize
            gen_hist = gen_hist.astype(np.float64) / (gen_hist.sum() + 1e-10)
            ref_hist = ref_hist.astype(np.float64) / (ref_hist.sum() + 1e-10)
            # Intersection
            similarities.append(float(np.minimum(gen_hist, ref_hist).sum()))

        return float(np.mean(similarities))

    def compute_megastyle(
        self,
        generated: Image.Image,
        reference: Image.Image,
        *,
        reference_key: str | None = None,
    ) -> float:
        """MegaStyle-Encoder cosine similarity between generated and reference.

        Returns a float in [0, 1]; higher is better. Reference embeddings are
        memoized by ``reference_key`` (typically ``str(ref_path)``) so that
        multiple generated images paired to the same reference only pay for one
        reference forward pass per run.

        The cosine is clamped to [0, 1] — values below 0 aren't observed on
        natural image pairs but the clamp is defensive against edge cases.
        """
        with self._megastyle_lock:
            self._ensure_megastyle()
            ref_emb = self._megastyle_ref_cache.get(reference_key) if reference_key else None
            if ref_emb is None:
                ref_emb = self._embed_megastyle(reference)
                if reference_key is not None:
                    self._megastyle_ref_cache[reference_key] = ref_emb
            gen_emb = self._embed_megastyle(generated)
            cos = float((gen_emb * ref_emb).sum().item())
            return max(0.0, min(1.0, cos))

    def compute_ssim(self, generated: Image.Image, reference: Image.Image) -> float:
        """Structural Similarity Index (SSIM) for pixel-level comparison.

        Returns a float in [0, 1]; higher is better.  Both images are resized
        to 256x256 and converted to grayscale for comparison.
        """
        from skimage.metrics import structural_similarity

        gen_gray = np.array(generated.convert("L").resize((256, 256)), dtype=np.float64)
        ref_gray = np.array(reference.convert("L").resize((256, 256)), dtype=np.float64)
        result = structural_similarity(gen_gray, ref_gray, data_range=255.0)
        return float(result) if not isinstance(result, tuple) else float(result[0])
