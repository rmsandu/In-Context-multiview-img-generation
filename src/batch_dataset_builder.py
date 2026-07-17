"""Prepare, submit, and collect resumable Gemini batch caption jobs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

from google.genai import types
from PIL import Image
from pydantic import ValidationError
from tqdm import tqdm

from .captioner import (
    GEMINI_RESPONSE_SCHEMA,
    MAX_OUTPUT_TOKENS,
    MODEL_NAME,
    PROMPT_TEMPLATE,
    PROMPT_VERSION,
    CaptionResponse,
    MultiviewAnnotation,
    create_client,
    hash_composite_image,
    normalize_lateral_abstentions,
    validate_annotation,
)
from .composite_img import make_composite_grid
from .dataset_builder import (
    CAPTION_SCHEMA_VERSION,
    InvalidInstanceError,
    _cache_envelope,
    _caption_cache_path,
    _load_cached_annotation,
    _write_cache,
    choose_four_views,
    find_image_dirs,
    hash_selected_views,
    load_categories,
)
from .dataset_builder import (
    main as build_dataset,
)

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _encode_jpeg(image: Image.Image, quality: int) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _batch_request(image: Image.Image, category: str, *, jpeg_quality: int) -> dict[str, Any]:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": _encode_jpeg(image, jpeg_quality),
                        }
                    },
                    {"text": PROMPT_TEMPLATE.format(category=category)},
                ],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_RESPONSE_SCHEMA,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }


def _request_key(model: str, category: str, composite_hash: str) -> str:
    identity = "\n".join([model, PROMPT_VERSION, category, composite_hash])
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def prepare(args: argparse.Namespace) -> int:
    if not args.objects_dir.is_dir():
        raise FileNotFoundError(f"Objects directory does not exist: {args.objects_dir}")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    categories = load_categories(args.category_file)
    image_dirs = find_image_dirs(args.objects_dir)
    records_file = args.work_dir / "requests.jsonl"
    chunks_dir = args.work_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    records: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    skipped = 0
    cached = 0
    chunk_paths: list[Path] = []

    def flush_chunk() -> None:
        if not requests:
            return
        path = chunks_dir / f"input-{len(chunk_paths):03d}.jsonl"
        path.write_text(
            "".join(json.dumps(request, separators=(",", ":")) + "\n" for request in requests),
            encoding="utf-8",
        )
        chunk_paths.append(path)
        requests.clear()

    for img_dir in tqdm(image_dirs, desc="Preparing batch requests"):
        try:
            views = choose_four_views(img_dir.glob("*.jpg"))
            hashes = hash_selected_views(views)
        except InvalidInstanceError:
            skipped += 1
            continue
        obj_id = img_dir.parent.parent.name
        category = categories.get(obj_id, "object").strip().replace(" ", "-")
        composite = make_composite_grid(views, target_h=args.tile_height, target_w=args.tile_width)
        composite_hash = hash_composite_image(composite)
        stem = f"{category}_{obj_id}_{img_dir.parent.name}"
        cache_path = _caption_cache_path(
            args.cache_dir, stem, category, composite_hash, model_id=args.model
        )
        key = _request_key(args.model, category, composite_hash)
        record = {
            "key": key,
            "instance": img_dir.parent.relative_to(args.objects_dir).as_posix(),
            "image_dir": img_dir.relative_to(args.objects_dir).as_posix(),
            "category": category,
            "stem": stem,
            "composite_sha256": composite_hash,
            "source_image_sha256": hashes,
            "cache_file": cache_path.as_posix(),
        }
        records.append(record)
        if cache_path.exists():
            envelope = _load_cached_annotation(cache_path)
            if not envelope["validation_errors"]:
                cached += 1
                continue
        requests.append(
            {
                "key": key,
                "request": _batch_request(composite, category, jpeg_quality=args.jpeg_quality),
            }
        )
        if len(requests) >= args.chunk_size:
            flush_chunk()
    flush_chunk()

    records_file.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    state = {
        "schema_version": 1,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "caption_schema_version": CAPTION_SCHEMA_VERSION,
        "objects_dir": args.objects_dir.resolve().as_posix(),
        "category_file": args.category_file.resolve().as_posix(),
        "output_dir": args.output_dir.resolve().as_posix(),
        "abstention_dir": args.abstention_dir.resolve().as_posix(),
        "cache_dir": args.cache_dir.resolve().as_posix(),
        "tile_width": args.tile_width,
        "tile_height": args.tile_height,
        "total_instances": len(records),
        "already_cached": cached,
        "skipped_instances": skipped,
        "chunks": [
            {"input_file": path.resolve().as_posix(), "job_name": None, "result_file": None}
            for path in chunk_paths
        ],
    }
    _atomic_json(args.work_dir / "state.json", state)
    print(f"Prepared {len(records) - cached} requests in {len(chunk_paths)} chunks")
    print(f"Already cached: {cached}; invalid/short instances skipped: {skipped}")
    return 0


def submit(args: argparse.Namespace) -> int:
    state_path = args.work_dir / "state.json"
    state = _load_json(state_path)
    client = create_client()
    for index, chunk in enumerate(state["chunks"]):
        if chunk.get("job_name"):
            print(f"Chunk {index:03d} already submitted as {chunk['job_name']}")
            continue
        input_path = Path(chunk["input_file"])
        uploaded = client.files.upload(
            file=input_path,
            config=types.UploadFileConfig(
                display_name=f"multiview-caption-input-{index:03d}", mime_type="jsonl"
            ),
        )
        job = client.batches.create(
            model=state["model"],
            src=uploaded.name,
            config={"display_name": f"multiview-captions-{index:03d}"},
        )
        chunk["uploaded_file"] = uploaded.name
        chunk["job_name"] = job.name
        _atomic_json(state_path, state)
        print(f"Submitted chunk {index:03d}: {job.name}")
    return 0


def _response_text(response: dict[str, Any]) -> str:
    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Batch response has no candidate content") from error
    return "".join(str(part.get("text", "")) for part in parts)


def _caption_response(raw_text: str, record: dict[str, Any], model: str) -> CaptionResponse:
    annotation: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        errors.append(f"response is not valid JSON: {error.msg}")
    else:
        try:
            parsed = normalize_lateral_abstentions(parsed)
            annotation = MultiviewAnnotation.model_validate(parsed).model_dump(mode="json")
        except ValidationError:
            errors.extend(validate_annotation(parsed))
    return CaptionResponse(
        annotation=annotation,
        raw_response_text=raw_text,
        model_id=model,
        prompt_version=PROMPT_VERSION,
        latency_ms=0.0,
        input_image_sha256=record["composite_sha256"],
        validation_errors=tuple(errors),
    )


def _download_completed_jobs(state: dict[str, Any], state_path: Path) -> tuple[int, int]:
    client = create_client()
    pending = 0
    failed = 0
    for index, chunk in enumerate(state["chunks"]):
        job = client.batches.get(name=chunk["job_name"])
        state_name = job.state.name
        chunk["state"] = state_name
        print(f"Chunk {index:03d}: {state_name}")
        if state_name not in TERMINAL_STATES:
            pending += 1
            continue
        if state_name != "JOB_STATE_SUCCEEDED":
            chunk["error"] = str(job.error)
            failed += 1
            continue
        if not chunk.get("result_file"):
            result_path = state_path.parent / "chunks" / f"result-{index:03d}.jsonl"
            result_path.write_bytes(client.files.download(file=job.dest.file_name))
            chunk["result_file"] = result_path.resolve().as_posix()
    _atomic_json(state_path, state)
    return pending, failed


def collect(args: argparse.Namespace) -> int:
    state_path = args.work_dir / "state.json"
    state = _load_json(state_path)
    while True:
        pending, failed_jobs = _download_completed_jobs(state, state_path)
        if failed_jobs:
            print(f"{failed_jobs} batch jobs failed")
            return 1
        if not pending or not args.wait:
            break
        print(f"{pending} jobs still running; polling again in {args.poll_seconds}s")
        time.sleep(args.poll_seconds)
        state = _load_json(state_path)
    if pending:
        return 2

    records = {
        record["key"]: record
        for line in (args.work_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line
        for record in [json.loads(line)]
    }
    result_errors = 0
    collected = 0
    for chunk in state["chunks"]:
        for line in Path(chunk["result_file"]).read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            result = json.loads(line)
            key = result.get("key")
            record = records.get(key)
            if record is None:
                print(f"Unknown or missing result key: {key!r}")
                result_errors += 1
                continue
            if result.get("error"):
                print(f"Request {key} failed: {result['error']}")
                result_errors += 1
                continue
            try:
                raw_text = _response_text(result["response"])
            except (KeyError, ValueError) as error:
                print(f"Request {key} has an invalid response: {error}")
                result_errors += 1
                continue
            response = _caption_response(raw_text, record, state["model"])
            envelope = _cache_envelope(
                response, category=record["category"], source_hashes=record["source_image_sha256"]
            )
            _write_cache(Path(record["cache_file"]), envelope)
            if response.validation_errors:
                result_errors += 1
            collected += 1
    print(f"Collected and cached {collected} responses; invalid/failed responses: {result_errors}")
    if result_errors:
        return 1

    build_args = [
        "--objects-dir",
        state["objects_dir"],
        "--category-file",
        state["category_file"],
        "--output-dir",
        state["output_dir"],
        "--abstention-dir",
        state["abstention_dir"],
        "--cache-dir",
        state["cache_dir"],
        "--model",
        state["model"],
        "--tile-width",
        str(state["tile_width"]),
        "--tile-height",
        str(state["tile_height"]),
    ]
    return build_dataset(build_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Build keyed batch JSONL files")
    prepare_parser.add_argument("--objects-dir", type=Path, required=True)
    prepare_parser.add_argument("--category-file", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--abstention-dir", type=Path, required=True)
    prepare_parser.add_argument("--cache-dir", type=Path, default=Path(".gemini_cache"))
    prepare_parser.add_argument("--work-dir", type=Path, required=True)
    prepare_parser.add_argument("--model", default=MODEL_NAME)
    prepare_parser.add_argument("--chunk-size", type=int, default=100)
    prepare_parser.add_argument("--jpeg-quality", type=int, default=92)
    prepare_parser.add_argument("--tile-width", type=int, default=512)
    prepare_parser.add_argument("--tile-height", type=int, default=512)
    prepare_parser.set_defaults(handler=prepare)

    for name, handler in (("submit", submit), ("collect", collect)):
        child = subparsers.add_parser(name)
        child.add_argument("--work-dir", type=Path, required=True)
        if name == "collect":
            child.add_argument("--wait", action="store_true")
            child.add_argument("--poll-seconds", type=int, default=30)
        child.set_defaults(handler=handler)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
