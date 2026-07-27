"""Generate paired base-FLUX and Study 1 LoRA four-view grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import yaml

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
    prompt_group: str
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


def validate_training_checkpoint(config_path: Path, lora_path: Path) -> str:
    """Return the AI Toolkit run name after checking it matches the checkpoint."""
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_name = document["config"]["name"]
    except (FileNotFoundError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read the training run name from {config_path}") from exc
    if not isinstance(run_name, str) or not run_name:
        raise ValueError(f"Invalid training run name in {config_path}")
    numbered_checkpoint = re.fullmatch(rf"{re.escape(run_name)}_\d+", lora_path.stem)
    if lora_path.stem != run_name and numbered_checkpoint is None:
        raise ValueError(
            f"Checkpoint {lora_path.name!r} does not match training run "
            f"{run_name!r} from {config_path}"
        )
    return run_name


def _installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]
    return str(value)


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
            prompt_id = value["id"]
            prompt_group = value["group"]
            prompt = value["prompt"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid prompt record on {path}:{line_number}") from exc
        if not all(
            isinstance(item, str) and item.strip()
            for item in (prompt_id, prompt_group, prompt)
        ):
            raise ValueError(
                f"Prompt id, group, and text must be non-empty strings on "
                f"{path}:{line_number}"
            )
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id: {prompt_id}")
        missing_tags = [tag for tag in REQUIRED_TAGS if tag not in prompt]
        if missing_tags:
            raise ValueError(f"Prompt {prompt_id!r} is missing tags: {missing_tags}")
        seen_ids.add(prompt_id)
        prompts.append({"id": prompt_id, "group": prompt_group, "prompt": prompt})
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def build_generation_plan(
    prompts: list[dict[str, str]],
    *,
    seeds: Sequence[int] = SEEDS,
    guidance_scale: float = GUIDANCE,
    num_inference_steps: int = STEPS,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> list[Generation]:
    if not seeds:
        raise ValueError("At least one generation seed is required")
    if any(isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("Generation seeds must be non-negative integers")
    return [
        Generation(
            prompt_id=prompt["id"],
            prompt_group=prompt["group"],
            prompt=prompt["prompt"],
            seed=seed,
            condition=condition,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
        )
        for condition in CONDITIONS
        for prompt in prompts
        for seed in seeds
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
            "prompt_group",
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


def _output_path(output_dir: Path, item: Generation) -> Path:
    return output_dir / "images" / item.prompt_id / str(item.seed) / f"{item.condition}.png"


def _metadata_path(output_dir: Path, item: Generation) -> Path:
    return output_dir / "images" / item.prompt_id / str(item.seed) / "metadata.json"


def _pair_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = records[0]
    parameter_exclusions = {
        "condition",
        "model",
        "adapter_active",
        "output_path",
        "output_sha256",
        "metadata_path",
        "prompt_id",
        "prompt_group",
        "prompt",
        "seed",
    }
    return {
        "schema_version": 1,
        "prompt_id": first["prompt_id"],
        "prompt_group": first["prompt_group"],
        "prompt": first["prompt"],
        "seed": first["seed"],
        "generation_parameters": {
            key: _json_safe(value)
            for key, value in first.items()
            if key not in parameter_exclusions
        },
        "outputs": {
            str(record["condition"]): {
                "adapter_active": record["adapter_active"],
                "path": record["output_path"],
                "sha256": record["output_sha256"],
            }
            for record in records
        },
    }


def _write_pair_metadata(
    *,
    records: list[dict[str, Any]],
    item: Generation,
    output_dir: Path,
) -> None:
    pair = [
        record
        for record in records
        if record["prompt_id"] == item.prompt_id and record["seed"] == item.seed
    ]
    metadata_path = _metadata_path(output_dir, item)
    metadata_path.write_text(
        json.dumps(_pair_metadata(pair), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_condition_controller(pipe: Any, condition: str) -> None:
    if condition == "base":
        pipe.disable_lora()
    else:
        pipe.enable_lora()


def build_condition_controller(
    pipe: Any,
    *,
    lora_path: Path,
    lora_scale: float,
    adapter_name: str = "study1_pilot",
) -> Callable[[str], None]:
    """Keep the base phase pristine, then load and activate one exact LoRA file."""
    lora_loaded = False

    def select_condition(condition: str) -> None:
        nonlocal lora_loaded
        if condition == "base":
            if lora_loaded:
                pipe.disable_lora()
            return
        if not lora_loaded:
            pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
            pipe.set_adapters(adapter_name, adapter_weights=[lora_scale])
            lora_loaded = True
        else:
            pipe.enable_lora()

    return select_condition


def generate(
    pipe: Any,
    plan: list[Generation],
    *,
    torch_module: Any,
    output_dir: Path,
    manifest_path: Path,
    run_config: dict[str, Any],
    condition_controller: Callable[[str], None] | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Run base then LoRA, recreating a CPU torch generator for every output."""
    existing = [path for item in plan if (path := _output_path(output_dir, item)).exists()]
    if manifest_path.exists():
        existing.append(manifest_path)
    if existing and not overwrite:
        preview = ", ".join(path.as_posix() for path in existing[:3])
        raise FileExistsError(
            f"Paired-generation outputs already exist ({preview}). "
            "Use a new --output-dir or pass --overwrite."
        )

    controller = condition_controller or (
        lambda condition: _default_condition_controller(pipe, condition)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []
    active_condition: str | None = None
    for item in plan:
        if item.condition != active_condition:
            controller(item.condition)
            active_condition = item.condition
        generator = torch_module.Generator(device="cpu").manual_seed(item.seed)
        result = pipe(
            item.prompt,
            generator=generator,
            guidance_scale=item.guidance_scale,
            num_inference_steps=item.num_inference_steps,
            width=item.width,
            height=item.height,
        )
        output_path = _output_path(output_dir, item)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output_path)
        metadata_path = _metadata_path(output_dir, item)
        record = {
            **asdict(item),
            **run_config,
            "generator_device": "cpu",
            "generator_reset_per_condition": True,
            "lora_fused": False,
            "adapter_active": item.condition == "lora",
            "model": item.condition,
            "output_path": _relative_or_absolute(output_path, manifest_path.parent),
            "output_sha256": sha256_file(output_path),
            "metadata_path": _relative_or_absolute(metadata_path, manifest_path.parent),
        }
        records.append(record)
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        _write_pair_metadata(records=records, item=item, output_dir=output_dir)
    validate_paired_records(records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lora",
        type=Path,
        required=True,
        help="Exact LoRA checkpoint file to evaluate",
    )
    parser.add_argument("--prompts", type=Path, default=Path("evaluation/prompts.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-model-revision")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/study1_pilot.yaml"),
        help="Config used to train the selected LoRA checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE)
    parser.add_argument("--num-inference-steps", type=int, default=STEPS)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.lora.is_file():
        raise FileNotFoundError(
            f"--lora must name the exact LoRA checkpoint file: {args.lora}"
        )
    if not args.training_config.is_file():
        raise FileNotFoundError(
            f"Training config does not exist: {args.training_config}"
        )
    training_run_name = validate_training_checkpoint(args.training_config, args.lora)

    import torch
    from diffusers import FluxPipeline

    dtype = getattr(torch, args.dtype)
    pipeline_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if args.base_model_revision:
        pipeline_kwargs["revision"] = args.base_model_revision
    pipe = FluxPipeline.from_pretrained(args.base_model, **pipeline_kwargs)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)

    manifest_path = args.output_dir / "generation_manifest.jsonl"
    lora_hash = sha256_file(args.lora) if args.lora.is_file() else None
    scheduler_config = _json_safe(getattr(pipe.scheduler, "config", {}))
    run_config = {
        "base_model": args.base_model,
        "base_model_revision": args.base_model_revision,
        "device": args.device,
        "dtype": args.dtype,
        "prompts_path": args.prompts.as_posix(),
        "prompts_resolved_path": args.prompts.resolve().as_posix(),
        "prompts_sha256": sha256_file(args.prompts),
        "training_run_name": training_run_name,
        "training_config_path": args.training_config.as_posix(),
        "training_config_resolved_path": args.training_config.resolve().as_posix(),
        "training_config_sha256": sha256_file(args.training_config),
        "lora_path": args.lora.as_posix(),
        "lora_resolved_path": args.lora.resolve().as_posix(),
        "lora_scale": args.lora_scale,
        "lora_sha256": lora_hash,
        "model_cpu_offload": args.cpu_offload,
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_config": scheduler_config,
        "condition_order": list(CONDITIONS),
        "base_generated_before_lora_load": True,
        "software_versions": {
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", None),
            "diffusers": _installed_version("diffusers"),
            "transformers": _installed_version("transformers"),
            "peft": _installed_version("peft"),
        },
    }

    plan = build_generation_plan(
        load_prompts(args.prompts),
        seeds=args.seeds,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        width=args.width,
        height=args.height,
    )
    with torch.inference_mode():
        records = generate(
            pipe,
            plan,
            torch_module=torch,
            output_dir=args.output_dir,
            manifest_path=manifest_path,
            run_config=run_config,
            condition_controller=build_condition_controller(
                pipe,
                lora_path=args.lora,
                lora_scale=args.lora_scale,
                adapter_name=training_run_name,
            ),
            overwrite=args.overwrite,
        )
    print(f"Generated {len(records)} outputs and wrote {manifest_path}")


if __name__ == "__main__":
    main()
