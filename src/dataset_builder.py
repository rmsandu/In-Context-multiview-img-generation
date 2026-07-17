import argparse
import hashlib
import json
import traceback
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .captioner import (
    MODEL_NAME,
    PROMPT_VERSION,
    AnnotationValidationError,
    CaptionResponse,
    eligibility_reasons,
    generate_structured_annotation,
    hash_composite_image,
    render_caption,
    validate_annotation,
)
from .composite_img import make_composite_grid

GRID_POSITIONS = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT")
CAPTION_SCHEMA_VERSION = "structured-viewpoints-v4"


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
    cache_dir: Path,
    stem: str,
    category: str,
    composite_hash: str,
    *,
    model_id: str = MODEL_NAME,
    prompt_version: str = PROMPT_VERSION,
) -> Path:
    cache_identity = "\n".join(
        [CAPTION_SCHEMA_VERSION, model_id, prompt_version, category, composite_hash]
    )
    digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{stem}_{CAPTION_SCHEMA_VERSION}_{digest}.json"


def _cache_envelope(
    response: CaptionResponse,
    *,
    category: str,
    source_hashes: list[str],
) -> dict[str, object]:
    return {
        "cache_schema_version": CAPTION_SCHEMA_VERSION,
        "model_id": response.model_id,
        "prompt_version": response.prompt_version,
        "latency_ms": response.latency_ms,
        "input_image_sha256": response.input_image_sha256,
        "source_image_sha256": source_hashes,
        "category": category,
        "raw_response_text": response.raw_response_text,
        "parsed_annotation": response.annotation,
        "validation_outcome": "valid" if not response.validation_errors else "invalid",
        "validation_errors": list(response.validation_errors),
    }


def _write_cache(path: Path, envelope: dict[str, object]) -> None:
    """Write a complete response envelope without risking a partial final file."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_cached_annotation(path: Path) -> dict[str, object]:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise AnnotationValidationError(f"Invalid cache entry {path}: {error}") from error
    if not isinstance(envelope, dict):
        raise AnnotationValidationError(f"Invalid cache entry {path}: expected an object")
    required = {
        "cache_schema_version",
        "model_id",
        "prompt_version",
        "latency_ms",
        "input_image_sha256",
        "source_image_sha256",
        "category",
        "raw_response_text",
        "parsed_annotation",
        "validation_outcome",
        "validation_errors",
    }
    missing = required - envelope.keys()
    if missing:
        raise AnnotationValidationError(
            f"Invalid cache entry {path}: missing {', '.join(sorted(missing))}"
        )
    return envelope


def _view_manifest(
    views: list[Path], hashes: list[str], *, objects_dir: Path
) -> list[dict[str, str]]:
    return [
        {
            "position": position,
            "filename": path.name,
            "path": path.relative_to(objects_dir).as_posix(),
            "sha256": digest,
        }
        for position, path, digest in zip(GRID_POSITIONS, views, hashes, strict=True)
    ]


def process_one(
    img_dir: Path,
    id2cat: dict[str, str],
    output_dir: Path,
    cache_dir: Path,
    abstention_dir: Path,
    *,
    objects_dir: Path,
    tile_width: int,
    tile_height: int,
    model_id: str = MODEL_NAME,
) -> tuple[str, dict[str, object]]:
    obj_id = img_dir.parent.parent.name
    category = id2cat.get(obj_id, "object").strip().replace(" ", "-")
    views = choose_four_views(img_dir.glob("*.jpg"))
    hashes = hash_selected_views(views)
    composite = make_composite_grid(views, target_h=tile_height, target_w=tile_width)
    composite_hash = hash_composite_image(composite)
    stem = f"{category}_{obj_id}_{img_dir.parent.name}"
    cached_response = _caption_cache_path(
        cache_dir, stem, category, composite_hash, model_id=model_id
    )
    if cached_response.exists():
        envelope = _load_cached_annotation(cached_response)
    else:
        response = generate_structured_annotation(composite, category, model_id=model_id)
        envelope = _cache_envelope(response, category=category, source_hashes=hashes)
        _write_cache(cached_response, envelope)

    expected_cache_values: dict[str, object] = {
        "cache_schema_version": CAPTION_SCHEMA_VERSION,
        "model_id": model_id,
        "prompt_version": PROMPT_VERSION,
        "input_image_sha256": composite_hash,
        "source_image_sha256": hashes,
        "category": category,
    }
    for field, expected in expected_cache_values.items():
        if envelope[field] != expected:
            raise AnnotationValidationError(
                f"Cache identity mismatch for {field}: expected {expected!r}"
            )

    cached_errors = envelope["validation_errors"]
    if not isinstance(cached_errors, list):
        raise AnnotationValidationError("Cached validation_errors must be an array")
    annotation = envelope["parsed_annotation"]
    current_errors = validate_annotation(annotation)
    all_errors = [str(error) for error in cached_errors] + current_errors
    if all_errors:
        raise AnnotationValidationError("; ".join(dict.fromkeys(all_errors)))
    assert isinstance(annotation, dict)

    rejection_reasons = eligibility_reasons(annotation)
    base_record: dict[str, object] = {
        "instance": img_dir.parent.relative_to(objects_dir).as_posix(),
        "output_image": f"{stem}.png",
        "cache_file": cached_response.as_posix(),
        "annotation": annotation,
        "views": _view_manifest(views, hashes, objects_dir=objects_dir),
    }
    if rejection_reasons:
        (output_dir / f"{stem}.png").unlink(missing_ok=True)
        (output_dir / f"{stem}.txt").unlink(missing_ok=True)
        composite.save(abstention_dir / f"{stem}.png")
        base_record["rejection_reasons"] = rejection_reasons
        return "abstention", base_record

    joint_caption = render_caption(annotation)
    (abstention_dir / f"{stem}.png").unlink(missing_ok=True)
    composite.save(output_dir / f"{stem}.png")
    (output_dir / f"{stem}.txt").write_text(joint_caption, encoding="utf-8")
    base_record["output_caption"] = f"{stem}.txt"
    return "accepted", base_record


def write_manifest(output_dir: Path, records: list[dict[str, object]]) -> None:
    """Rewrite the successful-datapoint manifest in deterministic JSONL form."""
    contents = "".join(
        f"{json.dumps(record, sort_keys=True)}\n"
        for record in sorted(records, key=lambda record: str(record["instance"]))
    )
    (output_dir / "manifest.jsonl").write_text(contents, encoding="utf-8")


def write_abstention_manifest(
    abstention_dir: Path, records: list[dict[str, object]]
) -> None:
    """Rewrite the ambiguous-example manifest in deterministic JSONL form."""
    contents = "".join(
        f"{json.dumps(record, sort_keys=True)}\n"
        for record in sorted(records, key=lambda record: str(record["instance"]))
    )
    (abstention_dir / "abstention_manifest.jsonl").write_text(contents, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build captioned 2x2 composites from an MVImgNet-style dataset."
    )
    parser.add_argument("--objects-dir", type=Path, required=True)
    parser.add_argument("--category-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--abstention-dir",
        type=Path,
        default=None,
        help="Directory for ambiguous composites (default: <output-dir>_abstention)",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".gemini_cache"))
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Gemini model ID (default: {MODEL_NAME})",
    )
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

    abstention_dir = args.abstention_dir or args.output_dir.with_name(
        f"{args.output_dir.name}_abstention"
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    abstention_dir.mkdir(parents=True, exist_ok=True)
    categories = load_categories(args.category_file)
    image_dirs = find_image_dirs(args.objects_dir)
    if args.limit is not None:
        image_dirs = image_dirs[: args.limit]
    if not image_dirs:
        raise ValueError(f"No image directories found below {args.objects_dir}")

    records: list[dict[str, object]] = []
    abstention_records: list[dict[str, object]] = []
    skipped = 0
    failures = 0
    for img_dir in tqdm(image_dirs, desc="Processing objects"):
        try:
            outcome, record = process_one(
                img_dir,
                categories,
                args.output_dir,
                args.cache_dir,
                abstention_dir,
                objects_dir=args.objects_dir,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                model_id=args.model,
            )
            if outcome == "accepted":
                records.append(record)
            else:
                abstention_records.append(record)
        except InvalidInstanceError as error:
            skipped += 1
            print(f"Skipped {img_dir}: {error}")
        except Exception as error:
            failures += 1
            print(f"Failed on {img_dir}: {error}")
            traceback.print_exc()

    write_manifest(args.output_dir, records)
    write_abstention_manifest(abstention_dir, abstention_records)
    print(f"Datapoints used: {len(records)}")
    print(f"Images used: {len(records) * 4}")
    print(f"Ambiguous instances retained: {len(abstention_records)}")
    print(f"Invalid instances skipped: {skipped}")
    print(f"Unexpected processing failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
