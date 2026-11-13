"""Rebuild an existing composite split with object-centred square crops instead of bars.

Study 1 letterboxed every 9:16 frame into a square tile, so 44% of each training
tile was black. This keeps everything else about a split fixed: the same objects,
the same four frames per object (checked by SHA-256), the same split and the same
caption files (copied byte for byte). Only the pixels of each tile change.

Each frame is cropped to the largest square that fits (its full short side), slid
along the long side so the square is centred on the object. The object is the
largest connected foreground region found by rembg (``isnet-general-use``, the
same masker the metrics use). Without a usable mask the square is centred on the
frame. Every crop decision is written to ``crop_manifest.jsonl``.

Needs ``.venv-eval`` (rembg, onnxruntime, scipy):

    .venv-eval/bin/python -m src.crop_composites \\
        --source-split training/study1_pilot \\
        --source-manifest training/composites_4view_grid_all/manifest.jsonl \\
        --objects-dir data --output-dir training/study1b_crop
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.composite_img import Box, make_composite_grid, object_square_box

GRID_POSITIONS = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT")
MASK_LONG_SIDE = 1024  # frames are masked at this size; boxes are scaled back up
MIN_OBJECT_FRACTION = 0.01  # same floor as evaluation.metrics.masks
TILE_SIZE = 512
SPLIT_KEYS = (
    "caption",
    "holdout_fraction",
    "image",
    "image_sha256",
    "seed",
    "source_image_sha256",
    "source_instance",
    "split",
    "stem",
)

Masker = Callable[[Image.Image], np.ndarray]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def largest_component_box(mask: np.ndarray) -> tuple[Box | None, float]:
    """Bounding box of the largest 4-connected foreground region, and its area share.

    Returns ``(None, share)`` when the region covers less than 1% of the frame,
    which is treated as "no object found".
    """
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("Mask must be a non-empty 2D array")
    labels, count = ndimage.label(mask)
    if count == 0:
        return None, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    largest = 1 + int(np.argmax(sizes))
    share = float(sizes[largest - 1]) / mask.size
    if share < MIN_OBJECT_FRACTION:
        return None, share
    ys, xs = np.nonzero(labels == largest)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), share


def scale_box(box: Box, scale_x: float, scale_y: float, size: tuple[int, int]) -> Box:
    """Map a box from mask pixels to frame pixels, rounding outwards."""
    left, top, right, bottom = box
    width, height = size
    return (
        max(0, int(np.floor(left * scale_x))),
        max(0, int(np.floor(top * scale_y))),
        min(width, int(np.ceil(right * scale_x))),
        min(height, int(np.ceil(bottom * scale_y))),
    )


def cut_fraction(object_box: Box | None, crop_box: Box) -> float:
    """Share of the object box's extent that falls outside the crop (0 = fully inside)."""
    if object_box is None:
        return 0.0
    ol, ot, orr, ob = object_box
    cl, ct, cr, cb = crop_box
    outside_x = max(0, cl - ol) + max(0, orr - cr)
    outside_y = max(0, ct - ot) + max(0, ob - cb)
    return max(outside_x / (orr - ol), outside_y / (ob - ot))


def locate_object(frame: Image.Image, masker: Masker) -> dict[str, Any]:
    """Find the object in one frame and choose its square crop."""
    small = frame.copy()
    small.thumbnail((MASK_LONG_SIDE, MASK_LONG_SIDE), Image.Resampling.LANCZOS)
    mask = masker(small)
    if mask.shape != (small.height, small.width):
        raise ValueError(f"Masker returned shape {mask.shape} for a {small.size} image")
    small_box, share = largest_component_box(mask)
    object_box = (
        None
        if small_box is None
        else scale_box(
            small_box, frame.width / small.width, frame.height / small.height, frame.size
        )
    )
    crop_box = object_square_box(frame.size, object_box)
    cut = cut_fraction(object_box, crop_box)
    return {
        "frame_size": list(frame.size),
        "object_box": None if object_box is None else list(object_box),
        "object_fraction": round(share, 6),
        "mask_fallback": object_box is None,
        "crop_box": list(crop_box),
        "object_inside": cut == 0.0,
        "cut_fraction": round(cut, 6),
    }


def code_version() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD") or None,
        "python_sources_dirty": bool(git("status", "--porcelain", "--", "src", "*.py")),
    }


def _installed(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def build(
    *,
    source_split: Path,
    source_manifest: Path,
    objects_dir: Path,
    output_dir: Path,
    masker: Masker,
    masker_info: dict[str, Any],
    tile_size: int = TILE_SIZE,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; refusing to overwrite a built split")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = read_jsonl(source_split / "split_manifest.jsonl")
    views_by_stem = {
        Path(row["output_image"]).stem: row["views"] for row in read_jsonl(source_manifest)
    }

    split_out: list[dict[str, Any]] = []
    crop_out: list[dict[str, Any]] = []
    for row in sorted(split_rows, key=lambda r: r["stem"]):
        stem = row["stem"]
        views = views_by_stem.get(stem)
        if views is None:
            raise ValueError(f"{stem} is in the split but not in {source_manifest}")
        if [v["position"] for v in views] != list(GRID_POSITIONS):
            raise ValueError(f"{stem}: views are not in grid order")
        frame_paths = [objects_dir / v["path"] for v in views]
        hashes = [sha256_file(path) for path in frame_paths]
        if hashes != [v["sha256"] for v in views] or hashes != list(row["source_image_sha256"]):
            raise ValueError(f"{stem}: source frames do not match their recorded hashes")

        # The letterboxed rebuild must equal the composite that was trained on, pixel
        # for pixel. That proves these are the same frames, in the same order.
        padded = make_composite_grid(frame_paths, target_h=tile_size, target_w=tile_size)
        with Image.open(source_split / row["image"]) as original:
            if not np.array_equal(np.asarray(original.convert("RGB")), np.asarray(padded)):
                raise ValueError(f"{stem}: letterboxed rebuild differs from {row['image']}")

        crops = []
        for view, path in zip(views, frame_paths, strict=True):
            with Image.open(path) as frame:
                crops.append(
                    {"position": view["position"], "path": view["path"], "sha256": view["sha256"]}
                    | locate_object(frame.convert("RGB"), masker)
                )
        grid = make_composite_grid(
            frame_paths,
            target_h=tile_size,
            target_w=tile_size,
            crop_boxes=[tuple(c["crop_box"]) for c in crops],
        )

        image_rel = Path(row["image"])
        caption_rel = Path(row["caption"])
        (output_dir / image_rel.parent).mkdir(parents=True, exist_ok=True)
        grid.save(output_dir / image_rel)
        shutil.copyfile(source_split / caption_rel, output_dir / caption_rel)
        if sha256_file(output_dir / caption_rel) != sha256_file(source_split / caption_rel):
            raise ValueError(f"{stem}: caption copy differs from the source")

        image_hash = sha256_file(output_dir / image_rel)
        split_out.append({key: row[key] for key in SPLIT_KEYS} | {"image_sha256": image_hash})
        crop_out.append(
            {
                "stem": stem,
                "split": row["split"],
                "image": image_rel.as_posix(),
                "image_sha256": image_hash,
                "padded_image_sha256": row["image_sha256"],
                "caption_sha256": sha256_file(output_dir / caption_rel),
                "views": crops,
            }
        )

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    write_jsonl(output_dir / "split_manifest.jsonl", split_out)
    write_jsonl(output_dir / "crop_manifest.jsonl", crop_out)
    summary = summarize(crop_out)
    info = {
        "source_split": source_split.as_posix(),
        "source_split_manifest_sha256": sha256_file(source_split / "split_manifest.jsonl"),
        "source_manifest": source_manifest.as_posix(),
        "objects_dir": objects_dir.as_posix(),
        "tile_size": tile_size,
        "mask_long_side": MASK_LONG_SIDE,
        "min_object_fraction": MIN_OBJECT_FRACTION,
        "masker": masker_info,
        "software_versions": {
            name: _installed(name)
            for name in ("pillow", "numpy", "scipy", "rembg", "onnxruntime-gpu")
        },
        **code_version(),
        "summary": summary,
    }
    (output_dir / "build_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return info


def summarize(crop_rows: list[dict[str, Any]]) -> dict[str, Any]:
    views = [v for row in crop_rows for v in row["views"]]
    cuts = np.array([v["cut_fraction"] for v in views])
    by_split: dict[str, int] = {}
    for row in crop_rows:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
    return {
        "composites": len(crop_rows),
        "by_split": by_split,
        "views": len(views),
        "portrait_views": sum(v["frame_size"][1] > v["frame_size"][0] for v in views),
        "landscape_views": sum(v["frame_size"][0] > v["frame_size"][1] for v in views),
        "mask_fallback_views": sum(v["mask_fallback"] for v in views),
        "object_inside_share": round(float(np.mean([v["object_inside"] for v in views])), 4),
        "cut_fraction_p90": round(float(np.percentile(cuts, 90)), 4),
        "cut_fraction_max": round(float(cuts.max()), 4),
        "composites_all_views_inside": sum(
            all(v["object_inside"] for v in row["views"]) for row in crop_rows
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source-split", type=Path, default=Path("training/study1_pilot"))
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("training/composites_4view_grid_all/manifest.jsonl"),
    )
    parser.add_argument("--objects-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/study1b_crop"))
    args = parser.parse_args(argv)

    from evaluation.metrics.masks import ForegroundMasker

    masker = ForegroundMasker()
    info = build(
        source_split=args.source_split,
        source_manifest=args.source_manifest,
        objects_dir=args.objects_dir,
        output_dir=args.output_dir,
        masker=masker,
        masker_info={"model": masker.model_name, "providers": list(masker.providers)},
    )
    print(json.dumps(info["summary"], indent=2))


if __name__ == "__main__":
    main()
