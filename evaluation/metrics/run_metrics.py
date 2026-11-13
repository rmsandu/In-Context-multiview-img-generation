"""Score four-view grids: identity, view diversity, pose agreement, duplicates.

Two layers, deliberately separated:

1. ``measure_grid`` runs the models on one image and returns raw numbers
   (embeddings similarities, pHash distances, Orient Anything angles). This is
   the expensive part. It depends only on the image bytes, so results are cached
   by SHA-256 and reused across reruns and threshold changes.
2. ``derive_grid_metrics`` turns raw numbers plus the grid's caption into the
   reported metrics. It is pure, CPU-only and unit-tested; every threshold it
   uses is written to ``run_info.json``.

Input is either a generation manifest from ``evaluation.generate_pairs`` or a
grid list (JSONL with ``grid_id``, ``image_path``, ``prompt_id``, ``seed``,
``condition``, ``view_caption``), e.g. the controls from ``evaluation.controls``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from evaluation.metrics.views import (
    TILE_PAIRS,
    angular_distance,
    count_distinct_views,
    expected_relative_azimuth,
    parse_tile_views,
)

METRIC_VERSION = "1"


@dataclass(frozen=True)
class Thresholds:
    # Two tiles count as the same viewpoint when the camera moved less than this.
    same_view_deg: float = 30.0
    # A pose pair is correct when the measured camera change is within this of the
    # change implied by the caption (half the 90-degree spacing of F/R/B/L).
    pose_tolerance_deg: float = 45.0
    # Sign that maps Orient Anything's relative azimuth onto the caption convention
    # (camera moving from front towards the object's right side is +90).
    # Calibrated once on the real holdout composites (-1: 0.42 vs +1: 0.29 pair accuracy);
    # see docs/study1_protocol.md.
    azimuth_sign: int = -1
    # A tile with less foreground than this is treated as "no object".
    min_foreground: float = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- inputs


def load_grid_list(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    base = path.parent
    out = []
    for row in rows:
        if "output_path" in row:  # generation_manifest.jsonl from generate_pairs
            is_base = row["condition"] == "base"
            if is_base:
                label = "base"
            elif row.get("lora_label"):
                label = row["lora_label"]
            else:  # classic two-condition run: name the arm by its scale
                label = f"lora@{row.get('lora_scale', 1.0):g}"
            out.append(
                {
                    "grid_id": f"{row['prompt_id']}/{row['seed']}/{label}",
                    "image_path": str(base / row["output_path"]),
                    "prompt_id": row["prompt_id"],
                    "seed": row["seed"],
                    "condition": row["condition"],
                    "arm": label,
                    "lora_scale": 0.0 if is_base else row.get("lora_scale"),
                    "lora_sha256": None if is_base else row.get("lora_sha256"),
                    "training_run_name": None if is_base else row.get("training_run_name"),
                    "quantization": row.get("quantization", "none"),
                    "view_caption": row["prompt"],
                    "expected_sha256": row.get("output_sha256"),
                }
            )
        else:
            row = dict(row)
            image = Path(row["image_path"])
            row["image_path"] = str(image if image.is_absolute() else base / image)
            row.setdefault("arm", row.get("condition"))
            out.append(row)
    return out


# ------------------------------------------------------------------------ measuring


class Scorers:
    """Loads every model once. Heavy imports happen here, not at module import."""

    def __init__(self, device: str = "cuda") -> None:
        from evaluation.metrics.embeddings import DinoV2Embedder, DreamSimEmbedder
        from evaluation.metrics.masks import ForegroundMasker
        from evaluation.metrics.pose import OrientAnythingV2

        self.masker = ForegroundMasker()
        self.dino = DinoV2Embedder(device)
        self.dreamsim = DreamSimEmbedder(device)
        self.pose = OrientAnythingV2(device)

    def describe(self) -> dict[str, Any]:
        return {**describe_models(), "mask_providers": self.masker.providers}


def describe_models() -> dict[str, Any]:
    """What produced the cached measurements. Static, so cache-only runs keep it."""
    from evaluation.metrics.embeddings import DINOV2_MODEL_ID
    from evaluation.metrics.masks import DEFAULT_MASK_MODEL
    from evaluation.metrics.pose import CHECKPOINT_FILE, CHECKPOINT_REPO, CODE_COMMIT

    return {
        "mask_model": f"rembg/{DEFAULT_MASK_MODEL}",
        "dinov2": DINOV2_MODEL_ID,
        "dreamsim": "dreamsim ensemble (pretrained=True)",
        "orient_anything": f"{CHECKPOINT_REPO}/{CHECKPOINT_FILE}",
        "orient_anything_code": f"SpatialVision/Orient-Anything-V2@{CODE_COMMIT}",
    }


def measure_grid(image_path: Path, scorers: Scorers) -> dict[str, Any]:
    import imagehash
    from PIL import Image

    from evaluation.metrics.embeddings import pairwise_cosine
    from evaluation.metrics.masks import foreground_fraction, masked_rgba, object_crop
    from evaluation.metrics.views import split_grid

    tiles = split_grid(Image.open(image_path).convert("RGB"))
    masks = [scorers.masker(tile) for tile in tiles]
    crops = [object_crop(tile, mask) for tile, mask in zip(tiles, masks, strict=True)]
    rgba = [masked_rgba(tile, mask) for tile, mask in zip(tiles, masks, strict=True)]

    dino_masked = pairwise_cosine(scorers.dino(crops), TILE_PAIRS)
    dino_unmasked = pairwise_cosine(scorers.dino(tiles), TILE_PAIRS)
    dreamsim_sim = pairwise_cosine(scorers.dreamsim(crops), TILE_PAIRS)
    hashes = [imagehash.phash(tile) for tile in tiles]
    pose = scorers.pose(rgba)

    return {
        "tiles": [
            {
                "foreground": foreground_fraction(masks[k]),
                "abs_azimuth": pose.abs_azimuth[k],
                "abs_elevation": pose.abs_elevation[k],
                "abs_azimuth_spread": pose.abs_azimuth_spread[k],
                "symmetry_alpha": pose.symmetry_alpha[k],
            }
            for k in range(4)
        ],
        "pairs": [
            {
                "i": i,
                "j": j,
                "dino_masked": dino_masked[n],
                "dino_unmasked": dino_unmasked[n],
                "dreamsim_distance": 1.0 - dreamsim_sim[n],
                "phash_distance": int(hashes[i] - hashes[j]),
                "rel_azimuth": pose.rel_azimuth[(i, j)],
                "rel_elevation": pose.rel_elevation[(i, j)],
                "rel_inconsistency": pose.rel_inconsistency[(i, j)],
            }
            for n, (i, j) in enumerate(TILE_PAIRS)
        ],
    }


# ------------------------------------------------------------------------- deriving


def derive_grid_metrics(
    raw: dict[str, Any], caption: str, thresholds: Thresholds = Thresholds()
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Grid-level metrics plus annotated pair rows. Pure function."""
    views = parse_tile_views(caption)
    pairs = []
    same_view: dict[tuple[int, int], bool] = {}
    pose_hits: list[bool] = []
    for pair in raw["pairs"]:
        i, j = pair["i"], pair["j"]
        same = angular_distance(pair["rel_azimuth"], 0.0) < thresholds.same_view_deg
        same_view[(i, j)] = same
        expected = expected_relative_azimuth(views, i, j)
        correct = None
        if expected is not None:
            measured = thresholds.azimuth_sign * pair["rel_azimuth"]
            correct = angular_distance(measured, expected) <= thresholds.pose_tolerance_deg
            pose_hits.append(correct)
        pairs.append({**pair, "same_view": same, "expected_rel": expected, "pose_ok": correct})

    def mean(key: str) -> float:
        return sum(p[key] for p in raw["pairs"]) / len(raw["pairs"])

    tiles = raw["tiles"]
    grid = {
        "identity_dino": mean("dino_masked"),
        "identity_dino_unmasked": mean("dino_unmasked"),
        "identity_dreamsim": 1.0 - mean("dreamsim_distance"),
        "min_pair_dreamsim_distance": min(p["dreamsim_distance"] for p in raw["pairs"]),
        "min_pair_phash_distance": min(p["phash_distance"] for p in raw["pairs"]),
        "distinct_views": count_distinct_views(same_view),
        "any_same_view": any(same_view.values()),
        "pose_accuracy": (sum(pose_hits) / len(pose_hits)) if pose_hits else None,
        "pose_pairs": len(pose_hits),
        "valid_grid": all(t["foreground"] >= thresholds.min_foreground for t in tiles),
        "orientable_tiles": sum(t["symmetry_alpha"] == 1 for t in tiles),
        "mean_rel_inconsistency": mean("rel_inconsistency"),
    }
    return grid, pairs


# ----------------------------------------------------------------------------- CLI


def _versions() -> dict[str, str]:
    out = {}
    for name in ("torch", "transformers", "dreamsim", "rembg", "imagehash", "onnxruntime-gpu"):
        try:
            out[name] = package_version(name)
        except PackageNotFoundError:
            pass
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(
    grid_list: Path,
    output_dir: Path,
    cache_dir: Path,
    thresholds: Thresholds,
    device: str,
    limit: int | None,
    arms: list[str] | None = None,
) -> None:
    import pandas as pd
    from PIL import Image

    from evaluation.metrics.views import letterbox_fraction, split_grid

    grids = load_grid_list(grid_list)
    if arms:
        grids = [grid for grid in grids if grid["arm"] in arms]
    grids = grids[:limit]
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorers: Scorers | None = None
    grid_rows, pair_rows, tile_rows = [], [], []
    started = time.time()
    for n, grid in enumerate(grids, 1):
        path = Path(grid["image_path"])
        digest = sha256_file(path)
        if grid.get("expected_sha256") and grid["expected_sha256"] != digest:
            raise ValueError(f"{path} does not match its manifest hash")
        cached = cache_dir / f"{digest}.v{METRIC_VERSION}.json"
        if cached.is_file():
            raw = json.loads(cached.read_text())
        else:
            scorers = scorers or Scorers(device)
            raw = measure_grid(path, scorers)
            cached.write_text(json.dumps(raw))
        metrics, pairs = derive_grid_metrics(raw, grid["view_caption"], thresholds)
        # Cheap pixel statistic, recomputed every run rather than cached.
        metrics["max_letterbox_fraction"] = max(
            letterbox_fraction(tile) for tile in split_grid(Image.open(path).convert("RGB"))
        )
        meta = {k: v for k, v in grid.items() if k not in ("view_caption", "expected_sha256")}
        meta["image_sha256"] = digest
        grid_rows.append({**meta, **metrics})
        pair_rows += [{"grid_id": grid["grid_id"], **p} for p in pairs]
        tile_rows += [
            {"grid_id": grid["grid_id"], "tile": k, **t} for k, t in enumerate(raw["tiles"])
        ]
        if n % 25 == 0 or n == len(grids):
            rate = (time.time() - started) / n
            print(f"[{n}/{len(grids)}] {rate:.2f}s/grid", flush=True)

    pd.DataFrame(grid_rows).to_csv(output_dir / "grids.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "pairs.csv", index=False)
    pd.DataFrame(tile_rows).to_csv(output_dir / "tiles.csv", index=False)
    info = {
        "grid_list": str(grid_list),
        "grid_list_sha256": sha256_file(grid_list),
        "n_grids": len(grid_rows),
        "arms_filter": arms,
        "metric_version": METRIC_VERSION,
        "thresholds": asdict(thresholds),
        "models": scorers.describe() if scorers else describe_models(),
        "measured_this_run": scorers is not None,
        "software_versions": _versions(),
        "git_commit": _git_commit(),
    }
    (output_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--grids", type=Path, required=True, help="generation manifest or grid list"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".metric_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--arm",
        action="append",
        dest="arms",
        help="only score these arms (repeatable), e.g. --arm base",
    )
    defaults = Thresholds()
    parser.add_argument("--same-view-deg", type=float, default=defaults.same_view_deg)
    parser.add_argument("--pose-tolerance-deg", type=float, default=defaults.pose_tolerance_deg)
    parser.add_argument("--azimuth-sign", type=int, choices=(1, -1), default=defaults.azimuth_sign)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    thresholds = Thresholds(
        same_view_deg=args.same_view_deg,
        pose_tolerance_deg=args.pose_tolerance_deg,
        azimuth_sign=args.azimuth_sign,
    )
    run(
        args.grids,
        args.output_dir,
        args.cache_dir,
        thresholds,
        args.device,
        args.limit,
        args.arms,
    )


if __name__ == "__main__":
    main()
