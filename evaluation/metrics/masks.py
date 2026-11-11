"""Foreground masks so identity is measured on the object, not the backdrop.

Generated four-view grids usually repeat one background across all tiles. A
whole-tile embedding then rewards a matching backdrop as much as a matching
object, which inflates "identity" for any model that copies its background.
Every identity and pose score is therefore computed on a masked object crop; the
unmasked DINOv2 score is kept only as a sensitivity check.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_MASK_MODEL = "isnet-general-use"
MIN_FOREGROUND_FRACTION = 0.01


class ForegroundMasker:
    def __init__(self, model_name: str = DEFAULT_MASK_MODEL) -> None:
        import onnxruntime as ort
        import rembg

        # onnxruntime-gpu finds cuBLAS/cuDNN through the wheels torch already installed.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        wanted = ("CUDAExecutionProvider", "CPUExecutionProvider")
        providers = [p for p in wanted if p in ort.get_available_providers()]
        self.model_name = model_name
        self._rembg = rembg
        self._session = rembg.new_session(model_name, providers=providers)
        # Recorded in run_info.json: CPU and GPU masks are not bit-identical.
        self.providers = self._session.inner_session.get_providers()

    def __call__(self, tile: Image.Image) -> np.ndarray:
        mask = self._rembg.remove(tile.convert("RGB"), session=self._session, only_mask=True)
        return np.asarray(mask, dtype=np.uint8) >= 128


def foreground_fraction(mask: np.ndarray) -> float:
    return float(mask.mean())


def masked_rgba(tile: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = tile.convert("RGBA")
    alpha = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    rgba.putalpha(alpha)
    return rgba


def object_crop(
    tile: Image.Image,
    mask: np.ndarray,
    *,
    background: tuple[int, int, int] = (127, 127, 127),
    margin: float = 0.08,
) -> Image.Image:
    """Square crop around the mask with the background replaced by flat grey.

    Falls back to the whole tile when the mask is (nearly) empty, so a failed
    segmentation degrades to the unmasked score instead of crashing the run.
    """
    if mask.mean() < MIN_FOREGROUND_FRACTION:
        return tile.convert("RGB")
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    canvas = Image.new("RGB", tile.size, background)
    canvas.paste(tile.convert("RGB"), (0, 0), Image.fromarray((mask * 255).astype(np.uint8)))
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)
    square = Image.new("RGB", (side, side), background)
    square.paste(canvas.crop(box), (0, 0))
    return square
