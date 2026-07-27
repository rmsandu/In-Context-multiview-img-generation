"""Compare base FLUX with selected Study 1 LoRA checkpoints.

This module keeps heavyweight torch and diffusers imports inside ``main`` so its
configuration, checkpoint-discovery, and manifest helpers remain CPU-only testable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import yaml
from packaging.version import Version
from PIL import Image, ImageDraw, ImageFont

REQUIRED_STEPS = (100, 300, 500)
PROMPT_ID = "snack_container"
SNACK_CONTAINER_PROMPT = (
    "[FOUR-VIEWS] Four views of A clear plastic container of Chinese snacks, specifically "
    "sesame candy, labeled with a green and white 'Weilong Moyu Shuang' brand tag.; "
    "[TOP-LEFT] Left side, high angle; clear plastic container lid and body, green and white "
    "brand tag with Chinese characters, barcode on the side of the tag, partially visible "
    "sesame candy inside; [TOP-RIGHT] Front, high angle; front of the green and white brand "
    "tag with 'Moyu Shuang' logo, clear plastic container lid, sesame candy visible inside "
    "the container; [BOTTOM-LEFT] Front-right three-quarter, high angle; front of the green "
    "and white brand tag, green label on the side of the container with 'Sesame Candy' in "
    "Chinese, stacked clear plastic containers, sesame candy visible inside; [BOTTOM-RIGHT] "
    "Rear-right three-quarter, high angle; back of the green label on the container, green "
    "and white brand tag, stacked clear plastic containers, background showing a store aisle "
    "with people."
)
CONDITION_LABELS = {
    "base": "BASE",
    "checkpoint_100": "STEP 100",
    "checkpoint_300": "STEP 300",
    "checkpoint_500": "STEP 500",
}
SCHEDULER_CLASSES = {"flowmatch": "FlowMatchEulerDiscreteScheduler"}
CHECKPOINT_STEP_RE = re.compile(r"(?:_|-)(\d+)$")
IMAGE_CONDITIONS = ("base", "checkpoint_100", "checkpoint_300", "checkpoint_500")
RESUME_FIELDS = (
    "prompt",
    "seed",
    "inference_steps",
    "guidance_scale",
    "resolution",
    "scheduler",
    "scheduler_class",
    "base_model_id",
    "quantization",
    "dtype",
)


@dataclass(frozen=True)
class StudyConfig:
    """Sampling values read directly from an AI Toolkit Study 1 config."""

    name: str
    base_model_id: str
    prompt: str
    seed: int
    inference_steps: int
    guidance_scale: float
    width: int
    height: int
    scheduler: str
    scheduler_class: str
    training_folder: Path
    training_steps: int
    train_text_encoder: bool

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
        or not prompts
        or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
    ):
        raise ValueError(
            "Study 1 sample.prompts must contain at least one non-empty string"
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
    train_text_encoder = train.get("train_text_encoder")
    if not isinstance(train_text_encoder, bool):
        raise ValueError(
            "Expected a boolean at config.process[0].train.train_text_encoder"
        )

    return StudyConfig(
        name=name,
        base_model_id=base_model_id,
        # Checkpoint comparison owns this fixed evaluation prompt. Training-monitor
        # prompts may change between runs without changing the controlled comparison.
        prompt=SNACK_CONTAINER_PROMPT,
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
        train_text_encoder=train_text_encoder,
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


def output_paths(output_dir: Path) -> dict[str, Path]:
    prompt_dir = output_dir / PROMPT_ID
    return {
        "base": prompt_dir / "base.png",
        "checkpoint_100": prompt_dir / "checkpoint_100.png",
        "checkpoint_300": prompt_dir / "checkpoint_300.png",
        "checkpoint_500": prompt_dir / "checkpoint_500.png",
        "comparison": prompt_dir / "comparison.png",
        "manifest": prompt_dir / "manifest.json",
        "log": prompt_dir / "generation.log",
    }


def build_manifest(
    *,
    config: StudyConfig,
    config_path: Path,
    checkpoint_paths: dict[int, Path],
    paths: dict[str, Path],
    scheduler_class: str,
    quantization: str,
    dtype: str,
    gpu: str,
    images: dict[str, dict[str, Any]] | None = None,
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
        "prompt_id": PROMPT_ID,
        "prompt": config.prompt,
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
        "quantization": quantization,
        "dtype": dtype,
        "gpu": gpu,
        "train_text_encoder": config.train_text_encoder,
        "expected_lora_components": ["transformer"],
        "checkpoints": checkpoints,
        "output_paths": {
            condition: path.resolve().as_posix() for condition, path in paths.items()
        },
        "images": images or {},
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def append_log(path: Path, record: dict[str, Any], *, truncate: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if truncate else "a"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"logged_at_unix": round(time.time(), 3), **record}, sort_keys=True
            )
            + "\n"
        )


def detect_gpu(torch_module: Any) -> str:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return "unavailable"
    return str(cuda.get_device_name(cuda.current_device()))


def _condition_step(condition: str) -> int | None:
    if condition == "base":
        return None
    return int(condition.removeprefix("checkpoint_"))


def _valid_output_image(path: Path, expected_size: tuple[int, int]) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == expected_size
    except (OSError, SyntaxError):
        return False


def reusable_image_record(
    *,
    existing_manifest: dict[str, Any] | None,
    expected_manifest: dict[str, Any],
    condition: str,
    output_path: Path,
    expected_size: tuple[int, int],
) -> dict[str, Any] | None:
    """Return prior telemetry only when every controlled resume field still matches."""
    if existing_manifest is None:
        return None
    if any(
        existing_manifest.get(field) != expected_manifest.get(field)
        for field in RESUME_FIELDS
    ):
        return None

    images = existing_manifest.get("images")
    if not isinstance(images, dict):
        return None
    record = images.get(condition)
    if not isinstance(record, dict):
        return None

    step = _condition_step(condition)
    expected_checkpoint = (
        None if step is None else expected_manifest["checkpoints"][str(step)]
    )
    expected_hash = (
        None if expected_checkpoint is None else expected_checkpoint["sha256"]
    )
    expected_path = None if expected_checkpoint is None else expected_checkpoint["path"]
    if record.get("checkpoint_sha256") != expected_hash:
        return None
    if record.get("checkpoint_path") != expected_path:
        return None
    if record.get("output_path") != output_path.resolve().as_posix():
        return None
    if record.get("quantization") != expected_manifest["quantization"]:
        return None
    if record.get("dtype") != expected_manifest["dtype"]:
        return None
    if not isinstance(record.get("elapsed_seconds"), int | float):
        return None
    if not isinstance(record.get("peak_vram_bytes"), int):
        return None
    if not isinstance(record.get("gpu"), str):
        return None
    if not _valid_output_image(output_path, expected_size):
        return None
    return {**record, "reused": True}


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
) -> tuple[Image.Image, float, int]:
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = cuda is not None and cuda.is_available()
    if cuda_available:
        cuda.synchronize()
        cuda.reset_peak_memory_stats()
    generator = torch_module.Generator(device="cpu").manual_seed(config.seed)
    started = time.perf_counter()
    result = pipe(
        prompt=prompt,
        generator=generator,
        num_inference_steps=config.inference_steps,
        guidance_scale=config.guidance_scale,
        width=config.width,
        height=config.height,
    )
    if cuda_available:
        cuda.synchronize()
        peak_vram_bytes = int(cuda.max_memory_allocated())
    else:
        peak_vram_bytes = 0
    elapsed_seconds = time.perf_counter() - started
    return result.images[0], elapsed_seconds, peak_vram_bytes


def _image_record(
    *,
    condition: str,
    output_path: Path,
    expected_manifest: dict[str, Any],
    elapsed_seconds: float,
    peak_vram_bytes: int,
    gpu: str,
) -> dict[str, Any]:
    step = _condition_step(condition)
    checkpoint = None if step is None else expected_manifest["checkpoints"][str(step)]
    return {
        "output_path": output_path.resolve().as_posix(),
        "checkpoint_path": None if checkpoint is None else checkpoint["path"],
        "checkpoint_sha256": None if checkpoint is None else checkpoint["sha256"],
        "quantization": expected_manifest["quantization"],
        "dtype": expected_manifest["dtype"],
        "gpu": gpu,
        "peak_vram_bytes": peak_vram_bytes,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "reused": False,
    }


def load_and_verify_lora(
    pipe: Any,
    checkpoint: Path,
    *,
    adapter_name: str,
    quantization: str,
) -> None:
    """Load an unfused adapter and prove it is attached to the transformer."""
    if quantization == "8bit" and not getattr(
        pipe.transformer, "is_loaded_in_8bit", False
    ):
        raise RuntimeError("8-bit LoRA comparison requires an 8-bit transformer")
    if quantization == "8bit":
        prepare_quantized_transformer_for_peft(pipe.transformer)
    loaded = False
    try:
        pipe.load_lora_weights(str(checkpoint), adapter_name=adapter_name)
        loaded = True
        pipe.set_adapters(adapter_name, adapter_weights=1.0)
        transformer_adapters = pipe.get_list_adapters().get("transformer", [])
        if adapter_name not in transformer_adapters:
            raise RuntimeError(
                f"LoRA {adapter_name!r} did not attach to the transformer: {checkpoint}"
            )
    except Exception:
        if loaded:
            pipe.unload_lora_weights()
        raise


def prepare_quantized_transformer_for_peft(transformer: Any) -> int:
    """Restore a deprecated BnB state field still read by PEFT 0.10."""
    modules = getattr(transformer, "modules", None)
    if not callable(modules):
        return 0
    patched = 0
    for module in modules():
        if module.__class__.__name__ != "Linear8bitLt":
            continue
        state = getattr(module, "state", None)
        if state is not None and not hasattr(state, "memory_efficient_backward"):
            state.memory_efficient_backward = False
            patched += 1
    return patched


def run_comparison(
    *,
    pipe: Any,
    torch_module: Any,
    config: StudyConfig,
    config_path: Path,
    checkpoints: dict[int, Path],
    output_dir: Path,
    quantization: str,
    dtype: str,
    gpu: str,
    resume: bool,
) -> None:
    """Generate the base image first, then load one unfused LoRA at a time."""
    actual_scheduler_class = pipe.scheduler.__class__.__name__
    if actual_scheduler_class != config.scheduler_class:
        raise ValueError(
            f"Configured scheduler {config.scheduler!r} requires {config.scheduler_class}, "
            f"but the loaded base pipeline uses {actual_scheduler_class}"
        )

    paths = output_paths(output_dir)
    paths["base"].parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        checkpoint_paths=checkpoints,
        paths=paths,
        scheduler_class=actual_scheduler_class,
        quantization=quantization,
        dtype=dtype,
        gpu=gpu,
    )
    existing_manifest = read_manifest(paths["manifest"]) if resume else None
    reusable_records = {
        condition: record
        for condition in IMAGE_CONDITIONS
        if (
            record := reusable_image_record(
                existing_manifest=existing_manifest,
                expected_manifest=manifest,
                condition=condition,
                output_path=paths[condition],
                expected_size=(config.width, config.height),
            )
        )
        is not None
    }
    manifest["images"] = dict(reusable_records)
    append_log(
        paths["log"],
        {
            "event": "run_started",
            "resume": resume,
            "quantization": quantization,
            "dtype": dtype,
            "gpu": gpu,
        },
        truncate=not resume,
    )
    write_manifest(paths["manifest"], manifest)

    def generate_and_record(condition: str) -> None:
        image, elapsed_seconds, peak_vram_bytes = _generate_one(
            pipe, torch_module, config, config.prompt
        )
        image.save(paths[condition])
        record = _image_record(
            condition=condition,
            output_path=paths[condition],
            expected_manifest=manifest,
            elapsed_seconds=elapsed_seconds,
            peak_vram_bytes=peak_vram_bytes,
            gpu=gpu,
        )
        manifest["images"][condition] = record
        write_manifest(paths["manifest"], manifest)
        append_log(
            paths["log"], {"event": "image_generated", "condition": condition, **record}
        )

    def log_reuse(condition: str) -> None:
        append_log(
            paths["log"],
            {
                "event": "image_reused",
                "condition": condition,
                **manifest["images"][condition],
            },
        )

    # The pipeline is pristine here: base generation finishes before any adapter is loaded.
    if "base" in reusable_records:
        log_reuse("base")
    else:
        generate_and_record("base")

    for step in REQUIRED_STEPS:
        adapter_name = f"study1_step_{step}"
        adapter_loaded = False
        try:
            load_and_verify_lora(
                pipe,
                checkpoints[step],
                adapter_name=adapter_name,
                quantization=quantization,
            )
            adapter_loaded = True
            condition = f"checkpoint_{step}"
            append_log(
                paths["log"],
                {
                    "event": "lora_verified",
                    "condition": condition,
                    "adapter_name": adapter_name,
                    "checkpoint_path": checkpoints[step].resolve().as_posix(),
                    "text_encoder_lora_expected": config.train_text_encoder,
                },
            )
            if condition in reusable_records:
                log_reuse(condition)
            else:
                generate_and_record(condition)
        finally:
            # No weights are fused; completely remove this adapter before the next checkpoint.
            if adapter_loaded:
                pipe.unload_lora_weights()

    create_comparison(
        [
            (CONDITION_LABELS[condition], paths[condition])
            for condition in IMAGE_CONDITIONS
        ],
        paths["comparison"],
        expected_size=(config.width, config.height),
    )
    append_log(
        paths["log"],
        {
            "event": "run_completed",
            "comparison_path": paths["comparison"].resolve().as_posix(),
        },
    )
    write_manifest(paths["manifest"], manifest)


def validate_runtime_options(*, quantization: str, cpu_offload: bool) -> None:
    if quantization == "8bit" and cpu_offload:
        raise ValueError(
            "--quantization 8bit cannot be combined with --cpu-offload; "
            "8-bit loading uses device_map='auto'"
        )


def validate_8bit_dependency_versions() -> None:
    """Reject known Diffusers/Accelerate and PEFT/BitsAndBytes conflicts."""
    bitsandbytes_version = Version(package_version("bitsandbytes"))
    accelerate_version = Version(package_version("accelerate"))
    if bitsandbytes_version >= Version("0.48") and accelerate_version >= Version("1"):
        raise RuntimeError(
            "Incompatible 8-bit runtime: Diffusers 0.33 cannot use BitsAndBytes "
            f"{bitsandbytes_version} with Accelerate {accelerate_version} because Accelerate "
            "calls `.to()` on the quantized model. Install the pinned compatible versions "
            "with: .venv/bin/python -m pip install 'bitsandbytes>=0.43.3,<0.48' "
            "'accelerate>=0.28,<1'"
        )
    import bitsandbytes as bnb

    linear_state = bnb.nn.Linear8bitLt(1, 1, bias=False).state
    if not hasattr(linear_state, "memory_efficient_backward"):
        raise RuntimeError(
            f"BitsAndBytes {bitsandbytes_version} is incompatible with PEFT 0.10 LoRA "
            "injection because MatmulLtState.memory_efficient_backward is missing. "
            "Install the pinned version with: .venv/bin/python -m pip install "
            "'bitsandbytes==0.43.3'"
        )


def load_pipeline(
    *,
    config: StudyConfig,
    quantization: str,
    requested_dtype: str,
    device: str,
    cpu_offload: bool,
    torch_module: Any,
    diffusers_module: Any,
    transformers_module: Any,
) -> tuple[Any, str]:
    """Load either the regular pipeline or quantized FLUX transformer and T5 encoder."""
    validate_runtime_options(quantization=quantization, cpu_offload=cpu_offload)
    if quantization == "8bit":
        dtype = torch_module.bfloat16
        diffusers_quantization = diffusers_module.BitsAndBytesConfig(load_in_8bit=True)
        transformer = diffusers_module.FluxTransformer2DModel.from_pretrained(
            config.base_model_id,
            subfolder="transformer",
            quantization_config=diffusers_quantization,
            torch_dtype=dtype,
        )
        transformers_quantization = transformers_module.BitsAndBytesConfig(
            load_in_8bit=True
        )
        text_encoder_2 = transformers_module.T5EncoderModel.from_pretrained(
            config.base_model_id,
            subfolder="text_encoder_2",
            quantization_config=transformers_quantization,
            torch_dtype=dtype,
        )
        pipeline_kwargs = {
            "transformer": transformer,
            "text_encoder_2": text_encoder_2,
            "torch_dtype": dtype,
            "device_map": "auto",
        }
        try:
            pipe = diffusers_module.FluxPipeline.from_pretrained(
                config.base_model_id, **pipeline_kwargs
            )
        except NotImplementedError as exc:
            # Diffusers 0.33 only names this single-GPU placement strategy "balanced".
            # The BitsAndBytes quantizers have already placed both quantized components.
            if "auto not supported" not in str(exc):
                raise
            pipeline_kwargs["device_map"] = "balanced"
            pipe = diffusers_module.FluxPipeline.from_pretrained(
                config.base_model_id, **pipeline_kwargs
            )
        if not getattr(pipe.transformer, "is_loaded_in_8bit", False):
            raise RuntimeError("Flux transformer was not loaded in 8-bit mode")
        if not getattr(pipe.text_encoder_2, "is_loaded_in_8bit", False):
            raise RuntimeError("T5 text_encoder_2 was not loaded in 8-bit mode")
        return pipe, "bfloat16"

    dtype = getattr(torch_module, requested_dtype)
    pipe = diffusers_module.FluxPipeline.from_pretrained(
        config.base_model_id, torch_dtype=dtype
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe, requested_dtype


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "result root (default: evaluation/results_8bit for 8-bit, "
            "evaluation/results otherwise)"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantization", choices=("none", "8bit"), default="none")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_study_config(args.config)
    try:
        validate_runtime_options(
            quantization=args.quantization, cpu_offload=args.cpu_offload
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    checkpoint_dir = args.checkpoint_dir or config.default_checkpoint_dir(args.config)
    checkpoints = discover_checkpoints(checkpoint_dir, config)
    output_dir = args.output_dir or Path(
        "evaluation/results_8bit"
        if args.quantization == "8bit"
        else "evaluation/results"
    )

    import diffusers
    import torch
    import transformers

    if args.quantization == "8bit":
        validate_8bit_dependency_versions()
    pipe, dtype_name = load_pipeline(
        config=config,
        quantization=args.quantization,
        requested_dtype=args.dtype,
        device=args.device,
        cpu_offload=args.cpu_offload,
        torch_module=torch,
        diffusers_module=diffusers,
        transformers_module=transformers,
    )
    pipe.set_progress_bar_config(disable=False)
    gpu = detect_gpu(torch)

    with torch.inference_mode():
        run_comparison(
            pipe=pipe,
            torch_module=torch,
            config=config,
            config_path=args.config,
            checkpoints=checkpoints,
            output_dir=output_dir,
            quantization=args.quantization,
            dtype=dtype_name,
            gpu=gpu,
            resume=args.resume,
        )
    print(f"Wrote controlled comparisons to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
