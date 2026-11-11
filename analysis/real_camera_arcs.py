"""Measure the real camera arcs of the composites from MVImgNet's COLMAP poses.

The four photos in a composite are frames of one hand-held video. For 831 of the
1,896 downloaded instances MVImgNet ships a COLMAP reconstruction
(``sparse/0/images.bin``), which gives every frame's camera pose. That is a
reference for "how far apart are the four views" that depends on neither Gemini
nor Orient Anything, so it can grade both:

* Gemini's captions name a view per tile; label differences imply relative angles.
* Orient Anything's measured relative azimuths on the real holdout tiles
  (``results/metrics/controls/pairs.csv``) can be checked directly.

Geometry: camera centre C = -R^T t; the object sits where the optical axes meet
(least-squares point); "up" is the normal of the plane through the camera centres
(cameras circle at roughly constant height), oriented to agree with the cameras'
own up vectors. Azimuth is measured counter-clockwise seen from above.

    python -m analysis.real_camera_arcs
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from evaluation.metrics.views import (
    HORIZONTAL_AZIMUTH,
    TILE_PAIRS,
    count_distinct_views,
    parse_tile_views,
    wrap_degrees,
)

SAME_VIEW_DEG = 30.0  # same rule as evaluation.metrics.run_metrics.Thresholds


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_images_bin(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """COLMAP ``images.bin`` -> {image name: (R, t)} with x_cam = R x_world + t."""
    poses = {}
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            handle.read(4)  # image_id
            q = np.array(struct.unpack("<4d", handle.read(32)))
            t = np.array(struct.unpack("<3d", handle.read(24)))
            handle.read(4)  # camera_id
            name = bytearray()
            while (char := handle.read(1)) != b"\0":
                name += char
            (n_points,) = struct.unpack("<Q", handle.read(8))
            handle.seek(24 * n_points, 1)
            poses[name.decode()] = (qvec_to_rotmat(q), t)
    return poses


def orbit_geometry(poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict]:
    """Per-frame azimuth/elevation (degrees) around the point all cameras look at."""
    names = sorted(poses)
    rotations = [poses[n][0] for n in names]
    centres = np.array([-r.T @ poses[n][1] for r, n in zip(rotations, names, strict=True)])
    axes = np.array([r.T @ np.array([0.0, 0.0, 1.0]) for r in rotations])
    ups = np.array([r.T @ np.array([0.0, -1.0, 0.0]) for r in rotations])

    projector = [np.eye(3) - np.outer(d, d) for d in axes]
    target = np.linalg.solve(sum(projector), sum(p @ c for p, c in zip(projector, centres)))

    _, singular, vt = np.linalg.svd(centres - centres.mean(axis=0))
    up = vt[2] * np.sign(vt[2] @ ups.mean(axis=0))
    e1 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    rays = centres - target
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    azimuth = np.degrees(np.unwrap(np.arctan2(rays @ e2, rays @ e1)))
    elevation = np.degrees(np.arcsin(np.clip(rays @ up, -1, 1)))
    return {
        "frames": {
            name: {"azimuth": float(az - azimuth[0]), "elevation": float(el), "ray": ray}
            for name, az, el, ray in zip(names, azimuth, elevation, rays, strict=True)
        },
        "video_arc": float(azimuth.max() - azimuth.min()),
        # Close to 0 when the camera centres really lie in a plane.
        "planarity": float(singular[2] / singular[1]),
        "n_frames": len(names),
    }


def annotation_azimuths(annotation: dict) -> list[float | None]:
    """Azimuth of each tile from the structured Gemini annotation (None if unusable)."""
    out = []
    for view in annotation["views"]:
        horizontal, side = view["horizontal_view"], view["side"]
        label = {
            "side": f"{side} side",
            "front_three_quarter": f"front-{side} three-quarter",
            "rear_three_quarter": f"rear-{side} three-quarter",
        }.get(horizontal, horizontal)
        out.append(HORIZONTAL_AZIMUTH.get(label))
    return out


def composite_rows(record: dict, geometry: dict, source: str) -> tuple[dict, list[dict]]:
    frames = [geometry["frames"][view["filename"]] for view in record["views"]]
    stem = Path(record["output_image"]).stem
    if source == "accepted":
        caption = (Path(record["_dir"]) / record["output_caption"]).read_text()
        views = parse_tile_views(caption)
        gemini = [views[p].azimuth for p in views]
    else:
        gemini = annotation_azimuths(record["annotation"])
    pairs, same = [], {}
    for i, j in TILE_PAIRS:
        rel = frames[j]["azimuth"] - frames[i]["azimuth"]
        sep = float(np.degrees(np.arccos(np.clip(frames[i]["ray"] @ frames[j]["ray"], -1, 1))))
        same[(i, j)] = abs(rel) < SAME_VIEW_DEG
        g = gemini[i] is not None and gemini[j] is not None
        pairs.append(
            {
                "stem": stem,
                "source": source,
                "i": i,
                "j": j,
                "gt_rel_azimuth": round(rel, 2),
                "gt_separation": round(sep, 2),
                "gt_rel_elevation": round(frames[j]["elevation"] - frames[i]["elevation"], 2),
                "gemini_rel_azimuth": wrap_degrees(gemini[j] - gemini[i]) if g else None,
            }
        )
    grid = {
        "stem": stem,
        "source": source,
        "instance": record["instance"],
        "gt_span": round(frames[3]["azimuth"] - frames[0]["azimuth"], 2),
        "gt_mean_elevation": round(float(np.mean([f["elevation"] for f in frames])), 2),
        "gt_distinct_views": count_distinct_views(same),
        "video_arc": round(geometry["video_arc"], 2),
        "planarity": round(geometry["planarity"], 4),
        "n_frames": geometry["n_frames"],
        "gemini_labelled_tiles": sum(a is not None for a in gemini),
    }
    return grid, pairs


def circular_error(measured: np.ndarray, reference: np.ndarray, sign: int) -> np.ndarray:
    return np.abs((sign * measured - reference + 180.0) % 360.0 - 180.0)


def compare(pairs, measured: str, reference: str = "gt_rel_azimuth") -> dict:
    """Agreement of relative azimuths, with the unknown sign convention calibrated."""
    frame = pairs.dropna(subset=[measured])
    m, r = frame[measured].to_numpy(float), frame[reference].to_numpy(float)
    errors = {s: circular_error(m, r, s) for s in (1, -1)}
    sign = min(errors, key=lambda s: errors[s].mean())
    err = errors[sign]
    return {
        "n_pairs": int(len(frame)),
        "n_composites": int(frame.stem.nunique()),
        "sign": sign,
        "median_abs_error_deg": round(float(np.median(err)), 1),
        "within_45deg": round(float(np.mean(err <= 45)), 3),
        "pearson_abs": round(float(np.corrcoef(np.abs(m), np.abs(r))[0, 1]), 3),
        "median_abs_measured": round(float(np.median(np.abs(m))), 1),
        "median_abs_reference": round(float(np.median(np.abs(r))), 1),
    }


def main(argv: list[str] | None = None) -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--accepted", type=Path, default=Path("training/composites_4view_grid_all"))
    parser.add_argument(
        "--abstention", type=Path, default=Path("training/composites_4view_grid_all_abstention")
    )
    parser.add_argument(
        "--split", type=Path, default=Path("training/study1_pilot/split_manifest.jsonl")
    )
    parser.add_argument("--controls", type=Path, default=Path("results/metrics/controls/pairs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/real_arcs"))
    args = parser.parse_args(argv)

    sources = [
        ("accepted", args.accepted / "manifest.jsonl"),
        ("abstention", args.abstention / "abstention_manifest.jsonl"),
    ]
    grids, pairs, missing = [], [], {"accepted": 0, "abstention": 0}
    for source, manifest in sources:
        for line in manifest.read_text().splitlines():
            record = json.loads(line) | {"_dir": str(manifest.parent)}
            images_bin = args.data_root / record["instance"] / "sparse" / "0" / "images.bin"
            if not images_bin.is_file():
                missing[source] += 1
                continue
            geometry = orbit_geometry(read_images_bin(images_bin))
            if not all(v["filename"] in geometry["frames"] for v in record["views"]):
                missing[source] += 1  # a selected frame was not registered by COLMAP
                continue
            grid, rows = composite_rows(record, geometry, source)
            grids.append(grid)
            pairs.extend(rows)
    grids, pairs = pd.DataFrame(grids), pd.DataFrame(pairs)

    holdout = {
        json.loads(line)["stem"]
        for line in args.split.read_text().splitlines()
        if json.loads(line)["split"] == "holdout"
    }
    grids["holdout"] = grids.stem.isin(holdout)
    pairs["holdout"] = pairs.stem.isin(holdout)

    summary = {"composites_with_poses": grids.source.value_counts().to_dict(), "without": missing}
    for source, frame in grids.groupby("source"):
        consecutive = pairs[(pairs.source == source) & (pairs.j == pairs.i + 1)]
        summary[source] = {
            "median_span_deg": round(float(frame.gt_span.abs().median()), 1),
            "iqr_span_deg": [
                round(float(q), 1) for q in frame.gt_span.abs().quantile([0.25, 0.75])
            ],
            "median_consecutive_deg": round(float(consecutive.gt_rel_azimuth.abs().median()), 1),
            "median_video_arc_deg": round(float(frame.video_arc.median()), 1),
            "median_elevation_deg": round(float(frame.gt_mean_elevation.median()), 1),
            "mean_distinct_views_30deg": round(float(frame.gt_distinct_views.mean()), 3),
            "median_planarity": round(float(frame.planarity.median()), 4),
        }
    gemini = pairs[pairs.source == "accepted"]
    summary["gemini_vs_colmap"] = compare(gemini, "gemini_rel_azimuth")
    span = gemini[(gemini.i == 0) & (gemini.j == 3)]
    summary["gemini_vs_colmap"]["median_abs_span_gemini"] = round(
        float(span.gemini_rel_azimuth.abs().median()), 1
    )

    if args.controls.is_file():
        oa = pd.read_csv(args.controls)
        oa = oa[oa.grid_id.str.endswith("/real")].assign(
            stem=lambda f: f.grid_id.str.split("/").str[0]
        )
        merged = pairs.merge(oa[["stem", "i", "j", "rel_azimuth"]], on=["stem", "i", "j"])
        summary["orient_anything_vs_colmap"] = compare(merged, "rel_azimuth")
        pairs = pairs.merge(
            oa[["stem", "i", "j", "rel_azimuth"]].rename(columns={"rel_azimuth": "oa_rel_azimuth"}),
            on=["stem", "i", "j"],
            how="left",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grids.to_csv(args.output_dir / "composites.csv", index=False)
    pairs.to_csv(args.output_dir / "pairs.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
