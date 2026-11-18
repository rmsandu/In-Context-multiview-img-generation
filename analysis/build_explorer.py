"""Build the static results explorer's data: docs/explorer/data.json + thumbnails.

Only generated grids are exported. Real MVImgNet composites are covered by the
dataset's Terms of Use, so the explorer shows their *scores* but not the photos.

    python -m analysis.build_explorer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from analysis.study1_report import load_grids

THUMB_PX = 512
QUALITY = 74


def thumbnail(source: Path, target: Path) -> None:
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGB")
    image.thumbnail((THUMB_PX, THUMB_PX), Image.Resampling.LANCZOS)
    image.save(target, "WEBP", quality=QUALITY, method=6)


def grid_records(metrics_dir: Path) -> dict[str, dict]:
    grids = load_grids(metrics_dir)
    pairs = pd.read_csv(metrics_dir / "pairs.csv")
    out = {}
    for row in grids.itertuples(index=False):
        record = {
            "prompt": row.prompt_id,
            "seed": int(row.seed),
            "arm": row.arm,
            "identity": round(float(row.identity_dino), 4),
            "dreamsim": round(float(row.identity_dreamsim), 4),
            "views": int(row.distinct_views),
            "views_cal": int(row.distinct_views_cal),
            "letterbox": round(float(row.max_letterbox_fraction), 3),
            "valid": bool(row.valid_grid),
            "image": row.image_path,
        }
        # Camera azimuth of tiles 2-4 relative to tile 1, as Orient Anything reads it.
        rel = pairs[(pairs.grid_id == row.grid_id) & (pairs.i == 0)].sort_values("j")
        record["rel_az"] = [0.0] + [round(float(v), 1) for v in rel.rel_azimuth]
        out[row.grid_id] = record
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics"))
    parser.add_argument("--summary", type=Path, default=Path("results/study1/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/explorer"))
    args = parser.parse_args(argv)

    prompts = {}
    for line in Path("evaluation/prompts_holdout.jsonl").read_text().splitlines():
        row = json.loads(line)
        prompts[row["id"]] = {"group": row["group"].removeprefix("holdout_"), "text": row["prompt"]}

    data = {"prompts": prompts, "main": {}, "sweep": {}}
    for split in ("main", "sweep"):
        if not (args.metrics / split / "grids.csv").is_file():
            continue
        for grid_id, record in grid_records(args.metrics / split).items():
            rel = Path("thumbs") / split / f"{grid_id.replace('/', '_')}.webp"
            thumbnail(Path(record.pop("image")), args.output / rel)
            record["thumb"] = rel.as_posix()
            data[split][grid_id] = record

    controls = load_grids(args.metrics / "controls")
    data["reference"] = {
        arm: {
            "identity": round(float(frame.identity_dino.mean()), 4),
            "views": round(float(frame.distinct_views.mean()), 3),
            "views_cal": round(float(frame.distinct_views_cal.mean()), 3),
            "letterbox": round(float(frame.max_letterbox_fraction.mean()), 3),
            "n": int(len(frame)),
        }
        for arm, frame in controls.groupby("arm")
    }
    arcs = Path("results/real_arcs/summary.json")
    if arcs.is_file():
        data["camera_truth"] = json.loads(arcs.read_text())
    if args.summary.is_file():
        summary = json.loads(args.summary.read_text())
        data["summary"] = {
            "verdict": summary["decision"]["verdict"],
            "gates": summary["decision"]["gates"],
            "primary": {
                metric: summary["primary"]["metrics"][metric]["pooled"]
                for metric in (
                    "identity_dino",
                    "distinct_views",
                    "distinct_views_cal",
                    "identity_dreamsim",
                    "max_letterbox_fraction",
                )
            },
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
    print(f"{len(data['main'])} main + {len(data['sweep'])} sweep grids, {size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
