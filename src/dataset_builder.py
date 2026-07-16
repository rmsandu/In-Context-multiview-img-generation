import argparse
import hashlib
import json
import traceback
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .captioner import generate_caption_composite_grid
from .composite_img import make_composite_grid

GRID_POSITIONS = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT")
CAPTION_SCHEMA_VERSION = "neutral-positions-v1"


class InvalidInstanceError(ValueError):
    """An image instance cannot produce a valid four-view datapoint."""


def find_image_dirs(root: Path) -> list[Path]:
    """Return sorted leaf directories named ``images``."""
    return sorted(path for path in root.rglob("images") if path.is_dir())


def load_categories(path: Path) -> dict[str, str]:
    """Load an MVImgNet category mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"Category file does not exist: {path}")
    return dict(
        line.strip().split(",", maxsplit=1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "," in line
    )


def choose_four_views(img_paths: Iterable[Path]) -> list[Path]:
    """Choose four deterministic temporal views without selecting the endpoint."""
    paths = sorted(Path(path) for path in img_paths)
    if len(paths) < 4:
        raise InvalidInstanceError(
            f"At least four source images are required, received {len(paths)}"
        )

    indices = np.linspace(0, len(paths), num=4, endpoint=False, dtype=int)
    selected = [paths[index] for index in indices]
    resolved_paths = [path.resolve() for path in selected]
    if len(set(resolved_paths)) != 4:
        raise InvalidInstanceError("Selected images must have four distinct resolved paths")
    return selected


def hash_selected_views(view_paths: Iterable[Path]) -> list[str]:
    """Hash four selected files and reject exact duplicate contents."""
    paths = list(view_paths)
    if len(paths) != 4:
        raise ValueError(f"Expected exactly four selected images, received {len(paths)}")

    hashes: list[str] = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes.append(digest.hexdigest())

    if len(set(hashes)) != 4:
        raise InvalidInstanceError("Selected images contain exact duplicate file contents")
    return hashes


def _caption_cache_path(
    cache_dir: Path, stem: str, views: list[Path], hashes: list[str]
) -> Path:
    cache_identity = "\n".join(
        [CAPTION_SCHEMA_VERSION, *(path.name for path in views), *hashes]
    )
    digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{stem}_{CAPTION_SCHEMA_VERSION}_{digest}.txt"


def process_one(
    img_dir: Path,
    id2cat: dict[str, str],
    output_dir: Path,
    cache_dir: Path,
    *,
    objects_dir: Path,
    tile_width: int,
    tile_height: int,
) -> dict[str, object]:
    obj_id = img_dir.parent.parent.name
    category = id2cat.get(obj_id, "object").strip().replace(" ", "-")
    views = choose_four_views(img_dir.glob("*.jpg"))
    hashes = hash_selected_views(views)
    composite = make_composite_grid(views, target_h=tile_height, target_w=tile_width)
    stem = f"{category}_{obj_id}_{img_dir.parent.name}"
    cached_caption = _caption_cache_path(cache_dir, stem, views, hashes)
    if cached_caption.exists():
        joint_caption = cached_caption.read_text(encoding="utf-8")
    else:
        joint_caption = generate_caption_composite_grid(composite, category)
        cached_caption.write_text(joint_caption, encoding="utf-8")
    composite.save(output_dir / f"{stem}.png")
    (output_dir / f"{stem}.txt").write_text(joint_caption, encoding="utf-8")
    return {
        "instance": img_dir.parent.relative_to(objects_dir).as_posix(),
        "output_image": f"{stem}.png",
        "output_caption": f"{stem}.txt",
        "views": [
            {
                "position": position,
                "filename": path.name,
                "path": path.relative_to(objects_dir).as_posix(),
                "sha256": digest,
            }
            for position, path, digest in zip(GRID_POSITIONS, views, hashes, strict=True)
        ],
    }


def write_manifest(output_dir: Path, records: list[dict[str, object]]) -> None:
    """Rewrite the successful-datapoint manifest in deterministic JSONL form."""
    contents = "".join(
        f"{json.dumps(record, sort_keys=True)}\n"
        for record in sorted(records, key=lambda record: str(record["instance"]))
    )
    (output_dir / "manifest.jsonl").write_text(contents, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build captioned 2x2 composites from an MVImgNet-style dataset."
    )
    parser.add_argument("--objects-dir", type=Path, required=True)
    parser.add_argument("--category-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".gemini_cache"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tile-width", type=int, default=512)
    parser.add_argument("--tile-height", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.objects_dir.is_dir():
        raise FileNotFoundError(f"Objects directory does not exist: {args.objects_dir}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.tile_width < 1 or args.tile_height < 1:
        raise ValueError("Tile dimensions must be positive")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    categories = load_categories(args.category_file)
    image_dirs = find_image_dirs(args.objects_dir)
    if args.limit is not None:
        image_dirs = image_dirs[: args.limit]
    if not image_dirs:
        raise ValueError(f"No image directories found below {args.objects_dir}")

    records: list[dict[str, object]] = []
    skipped = 0
    failures = 0
    for img_dir in tqdm(image_dirs, desc="Processing objects"):
        try:
            records.append(
                process_one(
                    img_dir,
                    categories,
                    args.output_dir,
                    args.cache_dir,
                    objects_dir=args.objects_dir,
                    tile_width=args.tile_width,
                    tile_height=args.tile_height,
                )
            )
        except InvalidInstanceError as error:
            skipped += 1
            print(f"Skipped {img_dir}: {error}")
        except Exception as error:
            failures += 1
            print(f"Failed on {img_dir}: {error}")
            traceback.print_exc()

    write_manifest(args.output_dir, records)
    print(f"Datapoints used: {len(records)}")
    print(f"Images used: {len(records) * 4}")
    print(f"Invalid instances skipped: {skipped}")
    print(f"Unexpected processing failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
