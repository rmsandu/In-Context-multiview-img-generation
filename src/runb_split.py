"""Assemble run B's training set: run A's 381 crops plus the newly captioned objects.

Two steps, around the unchanged crop builder (``src.crop_composites``):

1. ``new``: put the accepted new letterboxed composites in a train-only split folder
   that the crop builder can read. Every new composite goes to train (Study 1b
   protocol, run B). Before anything is copied, the new composites are checked
   against the Study 1 split: no shared source instance, composite hash or source
   frame hash with the holdout, and no shared instance with run A's training set.
2. ``combine``: after cropping, copy run A's cropped training composites and the
   cropped new ones into one folder, checking every file against its split
   manifest. Run A's part is also checked against the committed run A build record,
   so run B's training set contains run A's unchanged.

    .venv-eval/bin/python -m src.runb_split new
    .venv-eval/bin/python -m src.crop_composites --source-split training/runb_new \\
        --source-manifest training/composites_4view_grid_runb/manifest.jsonl \\
        --objects-dir data_runb --output-dir training/runb_new_crop
    .venv-eval/bin/python -m src.runb_split combine
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from src.study1_split import Pair, load_pairs, materialize_split, sha256_file, validate_partition

STUDY1_SPLIT = Path("training/study1_pilot/split_manifest.jsonl")
RUNA_RECORD = Path("results/study1b/crop/split_manifest.jsonl")
NEW_SOURCE = Path("training/composites_4view_grid_runb")
NEW_SPLIT = Path("training/runb_new")
NEW_CROP = Path("training/runb_new_crop")
RUNA_CROP = Path("training/study1b_crop")
COMBINED = Path("training/runb_crop")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _pairs_from_split(rows: list[dict[str, Any]], root: Path) -> list[Pair]:
    return [
        Pair(
            stem=r["stem"],
            instance=r["source_instance"],
            image=root / r["image"],
            caption=root / r["caption"],
            image_sha256=r["image_sha256"],
            source_image_sha256=tuple(r["source_image_sha256"]),
        )
        for r in rows
    ]


def check_new_against_study1(new: list[Pair], study1_rows: list[dict[str, Any]]) -> None:
    """Raise if a new composite shares anything with the holdout, or an object with train."""
    holdout = _pairs_from_split([r for r in study1_rows if r["split"] == "holdout"], Path())
    if not holdout:
        raise ValueError("The Study 1 split manifest has no holdout rows")
    validate_partition(new, holdout)  # instances, composite hashes, frame hashes
    study1_instances = {r["source_instance"] for r in study1_rows}
    repeated = sorted({p.instance for p in new} & study1_instances)
    if repeated:
        raise ValueError(f"New composites repeat Study 1 objects: {repeated[:5]}")
    study1_stems = {r["stem"] for r in study1_rows}
    clash = sorted({p.stem for p in new} & study1_stems)
    if clash:
        raise ValueError(f"New composite names clash with Study 1: {clash[:5]}")


def build_new_split(
    source_dir: Path, study1_split: Path, output_dir: Path, *, expected_count: int | None
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; refusing to overwrite a built split")
    pairs = load_pairs(source_dir, expected_count=expected_count)
    for pair in pairs:
        if len(pair.source_image_sha256) != 4:
            raise ValueError(f"{pair.stem}: expected 4 source frame hashes")
    check_new_against_study1(pairs, read_jsonl(study1_split))
    # seed and holdout_fraction are None: no random split, everything is train.
    return materialize_split(pairs, [], output_dir, seed=None, holdout_fraction=None)


def _train_rows(split_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(split_dir / "split_manifest.jsonl")
    return [r for r in rows if r["split"] == "train"]


def combine(
    parts: list[Path], output_dir: Path, *, runa_record: Path, study1_split: Path
) -> dict[str, Any]:
    """Copy the train rows of each cropped split into ``output_dir/train``."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; refusing to overwrite a built split")
    record = read_jsonl(runa_record)
    runa = {r["stem"]: r["image_sha256"] for r in record if r["split"] == "train"}
    if {r["stem"]: r["image_sha256"] for r in _train_rows(parts[0])} != runa:
        raise ValueError(f"{parts[0]} does not match the run A build record {runa_record}")

    # Letterboxed and cropped holdout composites, so both kinds of image hash are checked.
    holdout = [r for r in read_jsonl(study1_split) + record if r["split"] == "holdout"]
    rows_out: list[dict[str, Any]] = []
    seen_stems: set[str] = set()
    seen_instances: set[str] = set()
    counts: dict[str, int] = {}
    for part in parts:
        rows = _train_rows(part)
        counts[part.as_posix()] = len(rows)
        validate_partition(_pairs_from_split(rows, part), _pairs_from_split(holdout, Path()))
        for r in rows:
            if r["stem"] in seen_stems or r["source_instance"] in seen_instances:
                raise ValueError(f"{r['stem']} appears in more than one part")
            seen_stems.add(r["stem"])
            seen_instances.add(r["source_instance"])
            for key in ("image", "caption"):
                source = part / r[key]
                target = output_dir / "train" / Path(r[key]).name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            if sha256_file(output_dir / "train" / Path(r["image"]).name) != r["image_sha256"]:
                raise ValueError(f"{r['stem']}: copied image does not match its split manifest")
            rows_out.append(
                r
                | {
                    "image": f"train/{Path(r['image']).name}",
                    "caption": f"train/{Path(r['caption']).name}",
                    "caption_sha256": sha256_file(output_dir / "train" / Path(r["caption"]).name),
                    "part": part.as_posix(),
                }
            )
    rows_out.sort(key=lambda r: r["stem"])
    (output_dir / "split_manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows_out), encoding="utf-8"
    )
    pngs = {p.stem for p in (output_dir / "train").glob("*.png")}
    txts = {p.stem for p in (output_dir / "train").glob("*.txt")}
    if pngs != txts or pngs != seen_stems:
        raise ValueError(f"{output_dir / 'train'} does not hold exactly the combined pairs")
    info = {
        "parts": counts,
        "train": len(rows_out),
        "runa_record": runa_record.as_posix(),
        "runa_record_sha256": sha256_file(runa_record),
        "split_manifest_sha256": sha256_file(output_dir / "split_manifest.jsonl"),
    }
    (output_dir / "combine_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    new = sub.add_parser("new", help="train-only split of the new letterboxed composites")
    new.add_argument("--source-dir", type=Path, default=NEW_SOURCE)
    new.add_argument("--output-dir", type=Path, default=NEW_SPLIT)
    new.add_argument("--expected-count", type=int, default=936)
    both = sub.add_parser("combine", help="join run A's crops and the new crops")
    both.add_argument("--parts", type=Path, nargs="+", default=[RUNA_CROP, NEW_CROP])
    both.add_argument("--output-dir", type=Path, default=COMBINED)
    both.add_argument("--runa-record", type=Path, default=RUNA_RECORD)
    for p in (new, both):
        p.add_argument("--study1-split", type=Path, default=STUDY1_SPLIT)
    args = parser.parse_args(argv)

    if args.step == "new":
        manifest = build_new_split(
            args.source_dir, args.study1_split, args.output_dir, expected_count=args.expected_count
        )
        print(f"Wrote {len(read_jsonl(manifest))} train pairs; no overlap with Study 1: {manifest}")
    else:
        info = combine(
            args.parts, args.output_dir, runa_record=args.runa_record, study1_split=args.study1_split
        )
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
