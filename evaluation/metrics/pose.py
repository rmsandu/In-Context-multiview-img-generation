"""Viewpoint estimates from Orient Anything V2 (Wang et al., NeurIPS 2025).

Why this model: it predicts an object's orientation from a single image and,
given a (reference, target) pair, the *relative* rotation between them. It also
predicts a rotational-symmetry class (0 = no meaningful orientation, 1 = unique
front, 2/4 = symmetric), which matters for the bottles and jugs in the holdout set.
It is independent of the Gemini labeller that wrote the training captions, so it
does not grade Gemini's work with Gemini.

For each grid all 12 ordered tile pairs go through the model in one batch. Every
tile is therefore the reference three times, which gives three absolute-pose
readings per tile, and every unordered pair is measured in both directions. The
disagreement between i->j and j->i is stored as a free reliability signal.

Fetch the code and checkpoint with ``scripts/fetch_orient_anything.sh``.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.metrics.masks import MIN_FOREGROUND_FRACTION
from evaluation.metrics.views import TILE_PAIRS, angular_distance, wrap_degrees

DEFAULT_REPO = Path("third_party/Orient-Anything-V2")
CHECKPOINT_REPO = "Viglong/OriAnyV2_ckpt"
CHECKPOINT_FILE = "demo_ckpts/rotmod_realrotaug_best.pt"
CODE_COMMIT = "73b11c9dc83e84daeb563d0c766831f2c66b0a18"  # see scripts/fetch_orient_anything.sh
ORDERED_PAIRS = tuple(itertools.permutations(range(4), 2))


@dataclass(frozen=True)
class GridPose:
    abs_azimuth: list[float]  # per tile, circular mean of 3 reference readings
    abs_elevation: list[float]
    abs_azimuth_spread: list[float]  # max deviation among the 3 readings
    symmetry_alpha: list[int]  # majority vote of the 3 readings
    rel_azimuth: dict[tuple[int, int], float]  # i<j, camera change i->j, wrapped
    rel_elevation: dict[tuple[int, int], float]
    rel_inconsistency: dict[tuple[int, int], float]  # |rel(i->j) + rel(j->i)|


@contextlib.contextmanager
def _isolated_imports(repo_dir: Path, top_level: tuple[str, ...]):
    """Import the authors' flat-layout modules without clobbering same-named ones.

    The repo exposes top-level ``utils`` and ``vision_tower`` modules. DreamSim loads
    the DINO repo through torch.hub, which also registers a top-level ``utils``.
    Whatever was registered before is put back afterwards; the Orient Anything
    functions keep working because they hold references to their own modules.
    """

    def owned(name: str) -> bool:
        return any(name == top or name.startswith(top + ".") for top in top_level)

    saved = {name: mod for name, mod in sys.modules.items() if owned(name)}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(repo_dir))
    try:
        yield
    finally:
        sys.path.remove(str(repo_dir))
        for name in [n for n in sys.modules if owned(n)]:
            del sys.modules[name]
        sys.modules.update(saved)


def circular_mean(angles_deg: list[float]) -> float:
    rad = np.deg2rad(angles_deg)
    return float(np.rad2deg(math.atan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


class OrientAnythingV2:
    def __init__(self, device: str = "cuda", repo_dir: Path = DEFAULT_REPO) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        repo_dir = Path(repo_dir).resolve()
        if not (repo_dir / "vision_tower.py").is_file():
            raise FileNotFoundError(
                f"Orient Anything V2 code not found at {repo_dir}; "
                "run scripts/fetch_orient_anything.sh first"
            )
        with _isolated_imports(repo_dir, ("utils", "vision_tower", "vggt")):
            from utils.app_utils import preprocess_images, resize_foreground, val_fit_alpha
            from vision_tower import VGGT_OriAny_Ref

        self._torch = torch
        self._preprocess = preprocess_images
        self._resize_foreground = resize_foreground
        self._fit_alpha = val_fit_alpha
        self.device = device
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        ckpt = hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE)
        model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.model = model.eval().to(device)

    def _prepare(self, tile_rgba: Image.Image) -> Image.Image:
        # Same preparation as the authors' demo: object fills 85% of a padded square.
        # Below the same 1% floor that marks a tile invalid there is no object to crop
        # (and a few stray pixels give the authors' helper a zero-width image).
        alpha = np.asarray(tile_rgba.getchannel("A")) > 0
        if alpha.mean() < MIN_FOREGROUND_FRACTION:
            return tile_rgba.convert("RGB")
        return self._resize_foreground(tile_rgba, 0.85)

    def __call__(self, tiles_rgba: list[Image.Image]) -> GridPose:
        torch = self._torch
        prepared = [self._prepare(tile) for tile in tiles_rgba]
        tensors = self._preprocess(prepared, mode="pad")  # (4, 3, 518, 518)
        batch = torch.stack([torch.stack([tensors[i], tensors[j]]) for i, j in ORDERED_PAIRS])
        with torch.no_grad():
            enc = self.model(batch.to(self.device)).float()  # (12, 2, 900)
        az = torch.argmax(enc[..., 0:360], dim=-1).cpu().numpy().astype(float)
        el = (torch.argmax(enc[..., 360:540], dim=-1) - 90).cpu().numpy().astype(float)
        ref_dist = torch.sigmoid(enc[:, 0, 0:360]).cpu().numpy()
        with contextlib.redirect_stdout(io.StringIO()):  # the fitter prints every fit
            alphas = self._fit_alpha(distribute=ref_dist.copy()).numpy().astype(int)

        ref_az: dict[int, list[float]] = {k: [] for k in range(4)}
        ref_el: dict[int, list[float]] = {k: [] for k in range(4)}
        ref_alpha: dict[int, list[int]] = {k: [] for k in range(4)}
        rel: dict[tuple[int, int], float] = {}
        rel_el: dict[tuple[int, int], float] = {}
        for row, (i, j) in enumerate(ORDERED_PAIRS):
            ref_az[i].append(az[row, 0])
            ref_el[i].append(el[row, 0])
            ref_alpha[i].append(int(alphas[row]))
            rel[(i, j)] = wrap_degrees(az[row, 1])
            rel_el[(i, j)] = float(el[row, 1])

        abs_az = [circular_mean(ref_az[k]) for k in range(4)]
        spread = [max(angular_distance(a, abs_az[k]) for a in ref_az[k]) for k in range(4)]
        sym = [max(set(ref_alpha[k]), key=ref_alpha[k].count) for k in range(4)]
        rel_pair, rel_el_pair, inconsistency = {}, {}, {}
        for i, j in TILE_PAIRS:
            forward, backward = rel[(i, j)], rel[(j, i)]
            # Average i->j with the reversed j->i reading on the circle.
            rel_pair[(i, j)] = wrap_degrees(circular_mean([forward, -backward]))
            rel_el_pair[(i, j)] = (rel_el[(i, j)] - rel_el[(j, i)]) / 2.0
            inconsistency[(i, j)] = angular_distance(forward, -backward)
        return GridPose(
            abs_azimuth=abs_az,
            abs_elevation=[float(np.mean(ref_el[k])) for k in range(4)],
            abs_azimuth_spread=spread,
            symmetry_alpha=sym,
            rel_azimuth=rel_pair,
            rel_elevation=rel_el_pair,
            rel_inconsistency=inconsistency,
        )
