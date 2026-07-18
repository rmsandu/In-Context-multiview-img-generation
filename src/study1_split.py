"""Create and verify the deterministic instance-level Study 1 pilot split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOURCE = Path("training/composites_4view_grid_all")
DEFAULT_OUTPUT = Path("training/study1_pilot")


@dataclass(frozen=True)
class Pair:
    stem: str
    instance: str
    image: Path
    caption: Path
    image_sha256: str
    source_image_sha256: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object on {path}:{line_number}")
        records.append(record)
    return records


def load_pairs(source_dir: Path, *, expected_count: int | None = None) -> list[Pair]:
    """Load accepted pairs and their canonical source-instance IDs."""
    manifest_path = source_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest does not exist: {manifest_path}")

    records = _read_jsonl(manifest_path)
    png_stems = {path.stem for path in source_dir.glob("*.png")}
    txt_stems = {path.stem for path in source_dir.glob("*.txt")}
    if png_stems != txt_stems:
        only_png = sorted(png_stems - txt_stems)
        only_txt = sorted(txt_stems - png_stems)
        raise ValueError(
            "PNG/TXT stems do not match; "
            f"missing captions={only_png[:5]}, missing images={only_txt[:5]}"
        )

    pairs: list[Pair] = []
    manifest_stems: set[str] = set()
    for record in records:
        try:
            instance = str(record["instance"])
            image_name = str(record["output_image"])
            caption_name = str(record["output_caption"])
        except KeyError as exc:
            raise ValueError(f"Source manifest record is missing {exc.args[0]!r}") from exc
        image = source_dir / image_name
        caption = source_dir / caption_name
        if image.suffix.lower() != ".png" or caption.suffix.lower() != ".txt":
            raise ValueError(f"Expected a PNG/TXT pair, got {image_name!r} and {caption_name!r}")
        if image.stem != caption.stem:
            raise ValueError(f"Manifest pair stems do not match: {image_name!r}, {caption_name!r}")
        if image.stem in manifest_stems:
            raise ValueError(f"Duplicate manifest stem: {image.stem}")
        if not image.is_file() or not caption.is_file():
            raise FileNotFoundError(f"Manifest pair is missing: {image} or {caption}")

        views = record.get("views", [])
        if not isinstance(views, list):
            raise ValueError(f"Manifest views must be a list for {image.stem}")
        source_hashes = tuple(
            str(view["sha256"]) for view in views if isinstance(view, dict) and "sha256" in view
        )
        pairs.append(
            Pair(
                stem=image.stem,
                instance=instance,
                image=image,
                caption=caption,
                image_sha256=sha256_file(image),
                source_image_sha256=source_hashes,
            )
        )
        manifest_stems.add(image.stem)

    if manifest_stems != png_stems:
        missing = sorted(png_stems - manifest_stems)
        extra = sorted(manifest_stems - png_stems)
        raise ValueError(
            f"Manifest/file stems differ; missing from manifest={missing[:5]}, "
            f"missing from directory={extra[:5]}"
        )
    if expected_count is not None and len(pairs) != expected_count:
        raise ValueError(f"Expected {expected_count} accepted pairs, found {len(pairs)}")
    return sorted(pairs, key=lambda pair: pair.stem)


def split_pairs(
    pairs: list[Pair], *, seed: int = 17, holdout_fraction: float = 0.10
) -> tuple[list[Pair], list[Pair]]:
    """Split whole source-instance groups after a seeded shuffle."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    groups: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        groups[pair.instance].append(pair)
    if len(groups) < 2:
        raise ValueError("At least two source instances are required")

    instance_ids = sorted(groups)
    random.Random(seed).shuffle(instance_ids)
    holdout_group_count = round(len(instance_ids) * holdout_fraction)
    holdout_group_count = min(max(holdout_group_count, 1), len(instance_ids) - 1)
    holdout_instances = set(instance_ids[:holdout_group_count])
    train = sorted(
        (pair for pair in pairs if pair.instance not in holdout_instances),
        key=lambda pair: pair.stem,
    )
    holdout = sorted(
        (pair for pair in pairs if pair.instance in holdout_instances),
        key=lambda pair: pair.stem,
    )
    validate_partition(train, holdout, expected_total=len(pairs))
    return train, holdout


def validate_partition(
    train: list[Pair], holdout: list[Pair], *, expected_total: int | None = None
) -> None:
    if expected_total is not None and len(train) + len(holdout) != expected_total:
        raise ValueError(
            f"Split contains {len(train) + len(holdout)} pairs, expected {expected_total}"
        )
    train_instances = {pair.instance for pair in train}
    holdout_instances = {pair.instance for pair in holdout}
    instance_overlap = train_instances & holdout_instances
    if instance_overlap:
        raise ValueError(f"Source-instance overlap: {sorted(instance_overlap)[:5]}")

    train_hashes = {pair.image_sha256 for pair in train}
    holdout_hashes = {pair.image_sha256 for pair in holdout}
    image_overlap = train_hashes & holdout_hashes
    if image_overlap:
        raise ValueError(f"Composite image SHA-256 overlap: {sorted(image_overlap)[:5]}")

    train_source_hashes = {digest for pair in train for digest in pair.source_image_sha256}
    holdout_source_hashes = {digest for pair in holdout for digest in pair.source_image_sha256}
    source_overlap = train_source_hashes & holdout_source_hashes
    if source_overlap:
        raise ValueError(f"Source image SHA-256 overlap: {sorted(source_overlap)[:5]}")


def _copy_pair(pair: Pair, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in (pair.image, pair.caption):
        target = destination / source.name
        if target.exists():
            if sha256_file(target) != sha256_file(source):
                raise FileExistsError(f"Refusing to overwrite different file: {target}")
            continue
        shutil.copy2(source, target)


def _assert_exact_stems(directory: Path, expected: set[str]) -> None:
    png_stems = {path.stem for path in directory.glob("*.png")}
    txt_stems = {path.stem for path in directory.glob("*.txt")}
    if png_stems != txt_stems or png_stems != expected:
        raise ValueError(f"Materialized files do not exactly match the split in {directory}")


def materialize_split(
    train: list[Pair],
    holdout: list[Pair],
    output_dir: Path,
    *,
    seed: int = 17,
    holdout_fraction: float = 0.10,
) -> Path:
    """Copy pairs and write a provenance-rich split manifest."""
    validate_partition(train, holdout, expected_total=len(train) + len(holdout))
    train_dir = output_dir / "train"
    holdout_dir = output_dir / "holdout"
    for pair in train:
        _copy_pair(pair, train_dir)
    for pair in holdout:
        _copy_pair(pair, holdout_dir)
    _assert_exact_stems(train_dir, {pair.stem for pair in train})
    _assert_exact_stems(holdout_dir, {pair.stem for pair in holdout})

    manifest_path = output_dir / "split_manifest.jsonl"
    records = []
    for split_name, split_pairs_ in (("train", train), ("holdout", holdout)):
        for pair in split_pairs_:
            records.append(
                {
                    "caption": f"{split_name}/{pair.caption.name}",
                    "holdout_fraction": holdout_fraction,
                    "image": f"{split_name}/{pair.image.name}",
                    "image_sha256": pair.image_sha256,
                    "seed": seed,
                    "source_image_sha256": list(pair.source_image_sha256),
                    "source_instance": pair.instance,
                    "split": split_name,
                    "stem": pair.stem,
                }
            )
    records.sort(key=lambda record: (str(record["split"]), str(record["stem"])))
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--expected-count", type=int, default=423)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pairs = load_pairs(args.source_dir, expected_count=args.expected_count)
    train, holdout = split_pairs(pairs, seed=args.seed, holdout_fraction=args.holdout_fraction)
    manifest = materialize_split(
        train,
        holdout,
        args.output_dir,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
    )
    print(f"Wrote {len(train)} train and {len(holdout)} holdout pairs")
    print("Verified matching stems and zero instance/image-hash overlap")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
