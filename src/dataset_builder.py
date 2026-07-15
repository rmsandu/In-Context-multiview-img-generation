import argparse
import traceback
from collections.abc import Iterable
from pathlib import Path

from tqdm import tqdm

from .captioner import generate_caption_composite_grid
from .composite_img import make_composite_grid


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
    """Choose four deterministic, spaced views, duplicating short sweeps."""
    paths = sorted(img_paths)
    if not paths:
        raise ValueError("Cannot choose views from an empty image set")
    count = len(paths)
    indices = [0, count // 4, count // 2, max(0, count - 2)]
    return [paths[min(index, count - 1)] for index in indices]


def process_one(
    img_dir: Path,
    id2cat: dict[str, str],
    output_dir: Path,
    cache_dir: Path,
    *,
    tile_width: int,
    tile_height: int,
) -> None:
    obj_id = img_dir.parent.parent.name
    category = id2cat.get(obj_id, "object").strip().replace(" ", "-")
    views = choose_four_views(img_dir.glob("*.jpg"))
    composite = make_composite_grid(
        views, target_h=tile_height, target_w=tile_width
    )
    stem = f"{category}_{obj_id}_{img_dir.parent.name}"
    cached_caption = cache_dir / f"{stem}.txt"
    if cached_caption.exists():
        joint_caption = cached_caption.read_text(encoding="utf-8")
    else:
        joint_caption = generate_caption_composite_grid(composite, category)
        cached_caption.write_text(joint_caption, encoding="utf-8")
    composite.save(output_dir / f"{stem}.png")
    (output_dir / f"{stem}.txt").write_text(joint_caption, encoding="utf-8")


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

    failures = 0
    for img_dir in tqdm(image_dirs, desc="Processing objects"):
        try:
            process_one(
                img_dir,
                categories,
                args.output_dir,
                args.cache_dir,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
            )
        except Exception as error:
            failures += 1
            print(f"Failed on {img_dir}: {error}")
            traceback.print_exc()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
