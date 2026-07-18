"""Compare base FLUX with selected Study 1 LoRA checkpoints.

This module keeps heavyweight torch and diffusers imports inside ``main`` so its
configuration, checkpoint-discovery, and manifest helpers remain CPU-only testable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

REQUIRED_STEPS = (100, 300, 500)
PROMPT_IDS = ("snack_container", "baby_sweatshirt")
CONDITION_LABELS = {
    "base": "BASE",
    "checkpoint_100": "STEP 100",
    "checkpoint_300": "STEP 300",
    "checkpoint_500": "STEP 500",
}
SCHEDULER_CLASSES = {"flowmatch": "FlowMatchEulerDiscreteScheduler"}
CHECKPOINT_STEP_RE = re.compile(r"(?:_|-)(\d+)$")


@dataclass(frozen=True)
class StudyConfig:
    """Sampling values read directly from an AI Toolkit Study 1 config."""

    name: str
    base_model_id: str
    prompts: tuple[str, str]
    seed: int
    inference_steps: int
    guidance_scale: float
    width: int
    height: int
    scheduler: str
    scheduler_class: str
    training_folder: Path
    training_steps: int

    def default_checkpoint_dir(self, config_path: Path) -> Path:
        training_folder = self.training_folder
        if not training_folder.is_absolute():
            # AI Toolkit is launched from this repository root, one level above configs/.
            training_folder = config_path.resolve().parent.parent / training_folder
        return training_folder / self.name


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping at {location}")
    return value


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Expected a positive integer at {location}, got {value!r}")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Expected a number at {location}, got {value!r}")
    return float(value)


def load_study_config(path: Path) -> StudyConfig:
    """Load the sampling controls and output location from a Study 1 YAML file."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Study config does not exist: {path}") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in Study config {path}: {exc}") from exc

    root = _mapping(document, "root")
    config = _mapping(root.get("config"), "config")
    processes = config.get("process")
    if not isinstance(processes, list) or len(processes) != 1:
        raise ValueError("Expected exactly one entry at config.process")
    process = _mapping(processes[0], "config.process[0]")
    model = _mapping(process.get("model"), "config.process[0].model")
    sample = _mapping(process.get("sample"), "config.process[0].sample")
    train = _mapping(process.get("train"), "config.process[0].train")

    prompts = sample.get("prompts")
    if (
        not isinstance(prompts, list)
        or len(prompts) != len(PROMPT_IDS)
        or not all(isinstance(prompt, str) and prompt for prompt in prompts)
    ):
        raise ValueError(
            "Study 1 sample.prompts must contain exactly two non-empty strings"
        )

    scheduler = sample.get("sampler")
    if not isinstance(scheduler, str):
        raise ValueError(
            "Expected a scheduler name at config.process[0].sample.sampler"
        )
    normalized_scheduler = scheduler.lower().replace("_", "").replace("-", "")
    try:
        scheduler_class = SCHEDULER_CLASSES[normalized_scheduler]
    except KeyError:
        supported = ", ".join(sorted(SCHEDULER_CLASSES))
        raise ValueError(
            f"Unsupported sample scheduler {scheduler!r}; supported values: {supported}"
        ) from None

    base_model_id = model.get("name_or_path")
    name = config.get("name")
    training_folder = process.get("training_folder")
    if not isinstance(base_model_id, str) or not base_model_id:
        raise ValueError(
            "Expected a base-model ID at config.process[0].model.name_or_path"
        )
    if not isinstance(name, str) or not name:
        raise ValueError("Expected a job name at config.name")
    if not isinstance(training_folder, str) or not training_folder:
        raise ValueError(
            "Expected config.process[0].training_folder to be a path string"
        )

    seed = sample.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"Expected an integer at config.process[0].sample.seed, got {seed!r}"
        )
    if sample.get("walk_seed") is not False:
        raise ValueError("Controlled comparison requires sample.walk_seed: false")

    return StudyConfig(
        name=name,
        base_model_id=base_model_id,
        prompts=(prompts[0], prompts[1]),
        seed=seed,
        inference_steps=_positive_int(
            sample.get("sample_steps"), "config.process[0].sample.sample_steps"
        ),
        guidance_scale=_number(
            sample.get("guidance_scale"), "config.process[0].sample.guidance_scale"
        ),
        width=_positive_int(sample.get("width"), "config.process[0].sample.width"),
        height=_positive_int(sample.get("height"), "config.process[0].sample.height"),
        scheduler=scheduler,
        scheduler_class=scheduler_class,
        training_folder=Path(training_folder),
        training_steps=_positive_int(
            train.get("steps"), "config.process[0].train.steps"
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_step(path: Path, config: StudyConfig) -> tuple[int | None, bool]:
    """Return (step, explicitly_numbered) for an AI Toolkit checkpoint."""
    match = CHECKPOINT_STEP_RE.search(path.stem)
    if match:
        return int(match.group(1)), True
    if path.stem == config.name:
        # AI Toolkit saves the configured final step without a numeric suffix.
        return config.training_steps, False
    return None, False


def _format_discovered(
    checkpoint_dir: Path,
    candidates: dict[int, list[tuple[Path, bool]]],
    unrecognized: list[Path],
) -> str:
    lines = [f"Discovered .safetensors checkpoints in {checkpoint_dir.resolve()}:"]
    if not candidates and not unrecognized:
        lines.append("  (none)")
    for step in sorted(candidates):
        for path, explicit in candidates[step]:
            suffix = "numbered" if explicit else "unnumbered final"
            lines.append(f"  step {step}: {path.resolve()} ({suffix})")
    for path in sorted(unrecognized):
        lines.append(f"  unrecognized: {path.resolve()}")
    return "\n".join(lines)


def discover_checkpoints(
    checkpoint_dir: Path,
    config: StudyConfig,
    required_steps: Sequence[int] = REQUIRED_STEPS,
) -> dict[int, Path]:
    """Resolve requested AI Toolkit checkpoints, including its unnumbered final save."""
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )

    candidates: dict[int, list[tuple[Path, bool]]] = {}
    unrecognized: list[Path] = []
    for path in sorted(checkpoint_dir.rglob("*.safetensors")):
        step, explicit = _checkpoint_step(path, config)
        if step is None:
            unrecognized.append(path)
        else:
            candidates.setdefault(step, []).append((path, explicit))

    resolved: dict[int, Path] = {}
    ambiguous: list[int] = []
    for step in required_steps:
        choices = candidates.get(step, [])
        explicit_choices = [path for path, explicit in choices if explicit]
        if len(explicit_choices) == 1:
            resolved[step] = explicit_choices[0]
        elif len(explicit_choices) > 1:
            ambiguous.append(step)
        elif len(choices) == 1:
            resolved[step] = choices[0][0]
        elif len(choices) > 1:
            ambiguous.append(step)

    missing = [
        step
        for step in required_steps
        if step not in resolved and step not in ambiguous
    ]
    if missing or ambiguous:
        problems = []
        if missing:
            problems.append(
                "Missing required checkpoint steps: " + ", ".join(map(str, missing))
            )
        if ambiguous:
            problems.append(
                "Ambiguous checkpoint steps: " + ", ".join(map(str, ambiguous))
            )
        report = _format_discovered(checkpoint_dir, candidates, unrecognized)
        raise FileNotFoundError("\n".join([*problems, report]))
    return resolved


def output_paths(output_dir: Path, prompt_id: str) -> dict[str, Path]:
    prompt_dir = output_dir / prompt_id
    return {
        "base": prompt_dir / "base.png",
        "checkpoint_100": prompt_dir / "checkpoint_100.png",
        "checkpoint_300": prompt_dir / "checkpoint_300.png",
        "checkpoint_500": prompt_dir / "checkpoint_500.png",
        "comparison": prompt_dir / "comparison.png",
        "manifest": prompt_dir / "manifest.json",
    }


def build_manifest(
    *,
    config: StudyConfig,
    config_path: Path,
    prompt_id: str,
    prompt: str,
    checkpoint_paths: dict[int, Path],
    paths: dict[str, Path],
    scheduler_class: str,
) -> dict[str, Any]:
    """Build a complete, JSON-serializable record for one comparison row."""
    checkpoints = {
        str(step): {
            "path": checkpoint_paths[step].resolve().as_posix(),
            "sha256": sha256_file(checkpoint_paths[step]),
        }
        for step in REQUIRED_STEPS
    }
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "seed": config.seed,
        "inference_steps": config.inference_steps,
        "guidance_scale": config.guidance_scale,
        "resolution": {"width": config.width, "height": config.height},
        "scheduler": config.scheduler,
        "scheduler_class": scheduler_class,
        "base_model_id": config.base_model_id,
        "config_path": config_path.resolve().as_posix(),
        "generator": {"device": "cpu", "reset_before_every_generation": True},
        "lora_fused": False,
        "checkpoints": checkpoints,
        "output_paths": {
            condition: path.resolve().as_posix() for condition, path in paths.items()
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _label_font(image_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(24, image_height // 28)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def create_comparison(
    condition_paths: Sequence[tuple[str, Path]],
    destination: Path,
    *,
    expected_size: tuple[int, int],
) -> None:
    images: list[Image.Image] = []
    for label, path in condition_paths:
        with Image.open(path) as image:
            if image.size != expected_size:
                raise ValueError(
                    f"{label} image has size {image.size}, expected configured size "
                    f"{expected_size}: "
                    f"{path}"
                )
            images.append(image.convert("RGB"))

    font = _label_font(expected_size[1])
    label_height = max(64, expected_size[1] // 14)
    canvas = Image.new(
        "RGB",
        (expected_size[0] * len(images), expected_size[1] + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, ((label, _), image) in enumerate(
        zip(condition_paths, images, strict=True)
    ):
        left = index * expected_size[0]
        canvas.paste(image, (left, label_height))
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        draw.text(
            (
                left + (expected_size[0] - text_width) / 2,
                (label_height - text_height) / 2,
            ),
            label,
            fill="black",
            font=font,
        )
        if index:
            draw.line((left, 0, left, canvas.height), fill="#888888", width=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _generate_one(
    pipe: Any, torch_module: Any, config: StudyConfig, prompt: str
) -> Image.Image:
    generator = torch_module.Generator(device="cpu").manual_seed(config.seed)
    result = pipe(
        prompt=prompt,
        generator=generator,
        num_inference_steps=config.inference_steps,
        guidance_scale=config.guidance_scale,
        width=config.width,
        height=config.height,
    )
    return result.images[0]


def run_comparison(
    *,
    pipe: Any,
    torch_module: Any,
    config: StudyConfig,
    config_path: Path,
    checkpoints: dict[int, Path],
    output_dir: Path,
) -> None:
    """Generate every base image first, then load one unfused LoRA at a time."""
    actual_scheduler_class = pipe.scheduler.__class__.__name__
    if actual_scheduler_class != config.scheduler_class:
        raise ValueError(
            f"Configured scheduler {config.scheduler!r} requires {config.scheduler_class}, "
            f"but the loaded base pipeline uses {actual_scheduler_class}"
        )

    prompt_paths = {
        prompt_id: output_paths(output_dir, prompt_id) for prompt_id in PROMPT_IDS
    }
    for paths in prompt_paths.values():
        paths["base"].parent.mkdir(parents=True, exist_ok=True)

    # The pipeline is pristine here: all base generations finish before any adapter is loaded.
    for prompt_id, prompt in zip(PROMPT_IDS, config.prompts, strict=True):
        image = _generate_one(pipe, torch_module, config, prompt)
        image.save(prompt_paths[prompt_id]["base"])

    for step in REQUIRED_STEPS:
        adapter_name = f"study1_step_{step}"
        adapter_loaded = False
        try:
            pipe.load_lora_weights(str(checkpoints[step]), adapter_name=adapter_name)
            adapter_loaded = True
            pipe.set_adapters(adapter_name, adapter_weights=1.0)
            for prompt_id, prompt in zip(PROMPT_IDS, config.prompts, strict=True):
                image = _generate_one(pipe, torch_module, config, prompt)
                image.save(prompt_paths[prompt_id][f"checkpoint_{step}"])
        finally:
            # No weights are fused; completely remove this adapter before the next checkpoint.
            if adapter_loaded:
                pipe.unload_lora_weights()

    for prompt_id, prompt in zip(PROMPT_IDS, config.prompts, strict=True):
        paths = prompt_paths[prompt_id]
        create_comparison(
            [
                (CONDITION_LABELS[condition], paths[condition])
                for condition in (
                    "base",
                    "checkpoint_100",
                    "checkpoint_300",
                    "checkpoint_500",
                )
            ],
            paths["comparison"],
            expected_size=(config.width, config.height),
        )
        manifest = build_manifest(
            config=config,
            config_path=config_path,
            prompt_id=prompt_id,
            prompt=prompt,
            checkpoint_paths=checkpoints,
            paths=paths,
            scheduler_class=actual_scheduler_class,
        )
        write_manifest(paths["manifest"], manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/study1_pilot.yaml")
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="AI Toolkit job output directory (default: derived from the YAML)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--cpu-offload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_study_config(args.config)
    checkpoint_dir = args.checkpoint_dir or config.default_checkpoint_dir(args.config)
    checkpoints = discover_checkpoints(checkpoint_dir, config)

    import torch
    from diffusers import FluxPipeline

    dtype = getattr(torch, args.dtype)
    pipe = FluxPipeline.from_pretrained(config.base_model_id, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)

    with torch.inference_mode():
        run_comparison(
            pipe=pipe,
            torch_module=torch,
            config=config,
            config_path=args.config,
            checkpoints=checkpoints,
            output_dir=args.output_dir,
        )
    print(f"Wrote controlled comparisons to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
