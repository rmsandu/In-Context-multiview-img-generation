"""Generate paired base-FLUX and Study 1 LoRA four-view grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
SEEDS = (1001, 1002)
GUIDANCE = 3.5
STEPS = 20
WIDTH = 1024
HEIGHT = 1024
CONDITIONS = ("base", "lora")
REQUIRED_TAGS = ("[FOUR-VIEWS]", "[TOP-LEFT]", "[TOP-RIGHT]", "[BOTTOM-LEFT]", "[BOTTOM-RIGHT]")


@dataclass(frozen=True)
class Generation:
    prompt_id: str
    prompt: str
    seed: int
    condition: str
    guidance_scale: float = GUIDANCE
    num_inference_steps: int = STEPS
    width: int = WIDTH
    height: int = HEIGHT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
            prompt_id = value["id"]
            prompt = value["prompt"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid prompt record on {path}:{line_number}") from exc
        if not isinstance(prompt_id, str) or not isinstance(prompt, str):
            raise ValueError(f"Prompt id and text must be strings on {path}:{line_number}")
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id: {prompt_id}")
        missing_tags = [tag for tag in REQUIRED_TAGS if tag not in prompt]
        if missing_tags:
            raise ValueError(f"Prompt {prompt_id!r} is missing tags: {missing_tags}")
        seen_ids.add(prompt_id)
        prompts.append({"id": prompt_id, "prompt": prompt})
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def build_generation_plan(prompts: list[dict[str, str]]) -> list[Generation]:
    return [
        Generation(prompt["id"], prompt["prompt"], seed, condition)
        for prompt in prompts
        for seed in SEEDS
        for condition in CONDITIONS
    ]


def validate_paired_records(records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["prompt_id"]), int(record["seed"]))
        grouped.setdefault(key, []).append(record)
    for key, pair in grouped.items():
        conditions = {str(record["condition"]) for record in pair}
        if len(pair) != 2 or conditions != set(CONDITIONS):
            raise ValueError(f"Expected one base and one LoRA output for {key}, got {conditions}")
        first, second = pair
        controlled_fields = (
            "prompt",
            "seed",
            "guidance_scale",
            "num_inference_steps",
            "width",
            "height",
        )
        if any(first[field] != second[field] for field in controlled_fields):
            raise ValueError(f"Paired generation settings differ for {key}")


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def generate(
    pipe: Any,
    plan: list[Generation],
    *,
    torch_module: Any,
    image_dir: Path,
    manifest_path: Path,
    run_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run a plan, recreating a CPU torch generator for every condition."""
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []
    for item in plan:
        if item.condition == "base":
            pipe.disable_lora()
        else:
            pipe.enable_lora()
        generator = torch_module.Generator(device="cpu").manual_seed(item.seed)
        result = pipe(
            item.prompt,
            generator=generator,
            guidance_scale=item.guidance_scale,
            num_inference_steps=item.num_inference_steps,
            width=item.width,
            height=item.height,
        )
        output_path = image_dir / f"{item.prompt_id}_seed{item.seed}_{item.condition}.png"
        result.images[0].save(output_path)
        record = {
            **asdict(item),
            **run_config,
            "generator_device": "cpu",
            "generator_reset_per_condition": True,
            "lora_fused": False,
            "output_path": _relative_or_absolute(output_path, manifest_path.parent),
            "output_sha256": sha256_file(output_path),
        }
        records.append(record)
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    validate_paired_records(records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", type=Path, required=True, help="LoRA file or directory")
    parser.add_argument("--prompts", type=Path, default=Path("evaluation/prompts.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/outputs/study1_pilot"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--cpu-offload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.lora.exists():
        raise FileNotFoundError(f"LoRA does not exist: {args.lora}")

    import torch
    from diffusers import FluxPipeline

    dtype = getattr(torch, args.dtype)
    pipe = FluxPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    pipe.load_lora_weights(str(args.lora), adapter_name="study1_pilot")
    pipe.set_adapters("study1_pilot", adapter_weights=[args.lora_scale])
    pipe.set_progress_bar_config(disable=False)

    manifest_path = args.output_dir / "generation_manifest.jsonl"
    lora_hash = sha256_file(args.lora) if args.lora.is_file() else None
    run_config = {
        "base_model": args.base_model,
        "device": args.device,
        "dtype": args.dtype,
        "lora_path": args.lora.resolve().as_posix(),
        "lora_scale": args.lora_scale,
        "lora_sha256": lora_hash,
        "model_cpu_offload": args.cpu_offload,
    }
    with torch.inference_mode():
        records = generate(
            pipe,
            build_generation_plan(load_prompts(args.prompts)),
            torch_module=torch,
            image_dir=args.output_dir / "images",
            manifest_path=manifest_path,
            run_config=run_config,
        )
    print(f"Generated {len(records)} outputs and wrote {manifest_path}")


if __name__ == "__main__":
    main()
