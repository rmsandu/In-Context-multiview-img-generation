"""Identity embeddings: DINOv2 (fine-grained instance features) and DreamSim.

DINOv2 is the usual identity backbone in subject-driven generation work because
CLIP's image embedding is weak on fine-grained instance detail. DreamSim
ensembles CLIP, OpenCLIP and DINO features and was tuned on human similarity
judgements of diffusion-generated images, so it is the closer match to "would a
person call these the same thing". Both are reported; neither is trusted until
it passes the control-grid checks in evaluation/controls.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DINOV2_MODEL_ID = "facebook/dinov2-base"
DREAMSIM_CACHE = Path.home() / ".cache" / "dreamsim"


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class DinoV2Embedder:
    def __init__(self, device: str = "cuda", model_id: str = DINOV2_MODEL_ID) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.model_id = model_id
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        # Square inputs: resize the short side to 224 and the centre crop keeps everything.
        inputs = self.processor(
            images=[im.convert("RGB") for im in images],
            size={"shortest_edge": 224},
            crop_size={"height": 224, "width": 224},
            return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            pooled = self.model(**inputs).pooler_output  # layer-normed CLS token
        return _normalise(pooled.float().cpu().numpy())


class DreamSimEmbedder:
    def __init__(self, device: str = "cuda") -> None:
        import torch
        from dreamsim import dreamsim

        self._torch = torch
        self.device = device
        DREAMSIM_CACHE.mkdir(parents=True, exist_ok=True)
        self.model, self.preprocess = dreamsim(
            pretrained=True, device=device, cache_dir=str(DREAMSIM_CACHE)
        )
        self.model.eval()

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        batch = self._torch.cat([self.preprocess(im.convert("RGB")) for im in images])
        with self._torch.no_grad():
            emb = self.model.embed(batch.to(self.device))
        return _normalise(emb.float().cpu().numpy())


def pairwise_cosine(embeddings: np.ndarray, pairs) -> list[float]:
    return [float(embeddings[i] @ embeddings[j]) for i, j in pairs]
