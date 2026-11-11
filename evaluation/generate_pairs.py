"""Generate paired base-FLUX and Study 1 LoRA four-view grids.

Every prompt/seed is rendered once with the pristine base model and then once per
adapter arm, with a freshly seeded CPU generator for every image. The classic call
(``--lora PATH``) produces the two conditions ``base`` and ``lora``. Repeating
``--arm LABEL=PATH[@SCALE]`` evaluates several adapters (training replicates,
intermediate checkpoints, or one adapter at several scales) against a single shared
base phase, so base images are not regenerated per adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
SEEDS = (1001, 1002)
GUIDANCE = 3.5
STEPS = 20
WIDTH = 1024
HEIGHT = 1024
BASE_CONDITION = "base"
CONDITIONS = (BASE_CONDITION, "lora")
ARM_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]*")
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


def _canonical_scheduler_config(value: Any) -> Any:
    """Normalize unordered Diffusers bookkeeping without changing scheduler values."""
    config = _json_safe(value)
    if not isinstance(config, dict):
        return config
    default_values = config.get("_use_default_values")
    if isinstance(default_values, list) and all(isinstance(item, str) for item in default_values):
        config["_use_default_values"] = sorted(default_values)
    return config


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
            isinstance(item, str) and item.strip() for item in (prompt_id, prompt_group, prompt)
        ):
            raise ValueError(
                f"Prompt id, group, and text must be non-empty strings on {path}:{line_number}"
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
    arms: Sequence[str] = ("lora",),
) -> list[Generation]:
    validate_arm_labels(arms)
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
        for condition in (BASE_CONDITION, *arms)
        for prompt in prompts
        for seed in seeds
    ]


def validate_arm_labels(arms: Sequence[str]) -> None:
    if not arms:
        raise ValueError("At least one adapter arm is required")
    if len(set(arms)) != len(arms):
        raise ValueError(f"Adapter arm labels must be unique: {list(arms)}")
    for label in arms:
        if label == BASE_CONDITION or not ARM_LABEL_RE.fullmatch(label):
            raise ValueError(f"Invalid adapter arm label: {label!r}")


def plan_conditions(plan: Sequence[Generation]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.condition for item in plan))


def validate_paired_records(
    records: list[dict[str, Any]], conditions: Sequence[str] = CONDITIONS
) -> None:
    expected_conditions = set(conditions)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["prompt_id"]), int(record["seed"]))
        grouped.setdefault(key, []).append(record)
    for key, pair in grouped.items():
        conditions = {str(record["condition"]) for record in pair}
        if len(pair) != len(expected_conditions) or conditions != expected_conditions:
            raise ValueError(
                f"Expected one output per condition {sorted(expected_conditions)} for {key}, "
                f"got {sorted(conditions)}"
            )
        first = pair[0]
        controlled_fields = (
            "prompt_group",
            "prompt",
            "seed",
            "guidance_scale",
            "num_inference_steps",
            "width",
            "height",
        )
        if any(other[field] != first[field] for other in pair[1:] for field in controlled_fields):
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


def _generation_key(item: Generation) -> tuple[str, int, str]:
    return (item.prompt_id, item.seed, item.condition)


# Fields that differ between adapter arms in a multi-arm run. They are stored per
# output in metadata.json rather than as shared generation parameters.
ARM_FIELDS = (
    "lora_label",
    "lora_path",
    "lora_resolved_path",
    "lora_scale",
    "lora_sha256",
    "training_run_name",
    "training_config_path",
    "training_config_resolved_path",
    "training_config_sha256",
)


def _pair_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = records[0]
    parameter_exclusions = {
        "condition",
        "model",
        "adapter_active",
        "output_path",
        "output_sha256",
        "metadata_path",
        "elapsed_seconds",
        "peak_vram_bytes",
        "code_version",
        *ARM_FIELDS,
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
                **{field: record[field] for field in ARM_FIELDS if field in record},
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
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(_pair_metadata(pair), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_resume_records(
    *,
    plan: list[Generation],
    output_dir: Path,
    manifest_path: Path,
    run_config: dict[str, Any],
    arm_configs: Mapping[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load and verify records that are safe to reuse during a resumed run."""
    conditions = plan_conditions(plan)
    arm_configs = arm_configs or {}
    if not manifest_path.exists():
        existing = [path for item in plan if (path := _output_path(output_dir, item)).exists()]
        if existing:
            preview = ", ".join(path.as_posix() for path in existing[:3])
            raise ValueError(
                f"Cannot resume without {manifest_path}; existing outputs have no "
                f"verifiable provenance ({preview})."
            )
        return []

    plan_by_key = {_generation_key(item): item for item in plan}
    expected_paths = {_output_path(output_dir, item).resolve() for item in plan}
    images_dir = output_dir / "images"
    actual_paths = (
        {
            path.resolve()
            for condition in conditions
            for path in images_dir.rglob(f"{condition}.png")
        }
        if images_dir.is_dir()
        else set()
    )
    unexpected_paths = sorted(actual_paths - expected_paths)
    if unexpected_paths:
        preview = ", ".join(path.as_posix() for path in unexpected_paths[:3])
        raise ValueError(f"Cannot resume with outputs outside the current plan: {preview}")

    records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str]] = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in resume manifest {manifest_path}:{line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"Resume manifest record must be an object on {manifest_path}:{line_number}"
            )
        try:
            prompt_id = record["prompt_id"]
            seed = record["seed"]
            condition = record["condition"]
        except KeyError as exc:
            raise ValueError(
                f"Resume manifest record is missing {exc.args[0]!r} on "
                f"{manifest_path}:{line_number}"
            ) from exc
        if (
            not isinstance(prompt_id, str)
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or condition not in conditions
        ):
            raise ValueError(
                f"Invalid generation key in resume manifest {manifest_path}:{line_number}"
            )
        key = (prompt_id, seed, str(condition))
        if key in seen_keys:
            raise ValueError(f"Duplicate resume manifest record for {key}")
        try:
            item = plan_by_key[key]
        except KeyError as exc:
            raise ValueError(
                f"Resume manifest record {key} is outside the current generation plan"
            ) from exc

        output_path = _output_path(output_dir, item)
        metadata_path = _metadata_path(output_dir, item)
        expected_record = {
            **asdict(item),
            **run_config,
            **arm_configs.get(item.condition, {}),
            "generator_device": "cpu",
            "generator_reset_per_condition": True,
            "lora_fused": False,
            "adapter_active": item.condition != BASE_CONDITION,
            "model": item.condition,
            "output_path": _relative_or_absolute(output_path, manifest_path.parent),
            "metadata_path": _relative_or_absolute(metadata_path, manifest_path.parent),
        }
        mismatched_fields = [
            field
            for field, expected in expected_record.items()
            if field not in record
            or (
                _canonical_scheduler_config(record[field]) != _canonical_scheduler_config(expected)
                if field == "scheduler_config"
                else record[field] != expected
            )
        ]
        if mismatched_fields:
            raise ValueError(
                f"Resume manifest record {key} does not match the current run for "
                f"fields: {', '.join(mismatched_fields)}"
            )
        if not output_path.is_file():
            raise ValueError(
                f"Resume manifest record {key} references a missing output: {output_path}"
            )
        output_hash = record.get("output_sha256")
        if not isinstance(output_hash, str) or sha256_file(output_path) != output_hash:
            raise ValueError(
                f"Resume output hash does not match the manifest for {key}: {output_path}"
            )
        seen_keys.add(key)
        records.append(record)
    return records


def _restore_resume_metadata(
    *,
    records: list[dict[str, Any]],
    plan: list[Generation],
    output_dir: Path,
) -> None:
    """Recreate derived metadata files from verified manifest records."""
    recorded_pairs = {(str(record["prompt_id"]), int(record["seed"])) for record in records}
    restored_pairs: set[tuple[str, int]] = set()
    for item in plan:
        pair_key = (item.prompt_id, item.seed)
        if pair_key in recorded_pairs and pair_key not in restored_pairs:
            _write_pair_metadata(records=records, item=item, output_dir=output_dir)
            restored_pairs.add(pair_key)


def _default_condition_controller(pipe: Any, condition: str) -> None:
    if condition == "base":
        pipe.disable_lora()
    else:
        pipe.enable_lora()


@dataclass(frozen=True)
class AdapterArm:
    label: str
    path: Path
    scale: float
    adapter_name: str


def build_condition_controller(
    pipe: Any,
    *,
    lora_path: Path | None = None,
    lora_scale: float = 1.0,
    adapter_name: str = "study1_pilot",
    arms: Sequence[AdapterArm] | None = None,
    before_first_load: Callable[[], None] | None = None,
) -> Callable[[str], None]:
    """Keep the base phase pristine, then load and activate exact LoRA files.

    Nothing is loaded until the first adapter condition, so base images come from
    an untouched pipeline. Each distinct file is loaded once; arms that share a file
    (a scale sweep) only change the adapter weight. ``set_adapters`` activates the
    requested adapter and deactivates every other loaded one.
    """
    if arms is None:
        if lora_path is None:
            raise ValueError("Pass either lora_path or arms")
        arms = [AdapterArm("lora", lora_path, lora_scale, adapter_name)]
    by_label = {arm.label: arm for arm in arms}
    loaded: set[str] = set()
    disabled = False

    def select_condition(condition: str) -> None:
        nonlocal disabled
        if condition == BASE_CONDITION:
            if loaded:
                pipe.disable_lora()
                disabled = True
            return
        arm = by_label[condition]
        if disabled:
            pipe.enable_lora()
            disabled = False
        if arm.adapter_name not in loaded:
            if not loaded and before_first_load is not None:
                before_first_load()
            pipe.load_lora_weights(str(arm.path), adapter_name=arm.adapter_name)
            loaded.add(arm.adapter_name)
        pipe.set_adapters(arm.adapter_name, adapter_weights=[arm.scale])
        list_adapters = getattr(pipe, "get_list_adapters", None)
        if callable(list_adapters) and arm.adapter_name not in list_adapters().get(
            "transformer", []
        ):
            raise RuntimeError(f"LoRA {arm.label!r} did not attach to the transformer")

    return select_condition


def code_version() -> dict[str, Any]:
    """Commit and dirty flag of this checkout, recorded with every new output."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", "*.py"], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "python_sources_dirty": None}
    return {"git_commit": commit, "python_sources_dirty": dirty}


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
    resume: bool = False,
    arm_configs: Mapping[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run base then each adapter arm, recreating a CPU generator for every output.

    ``elapsed_seconds``, ``peak_vram_bytes`` and ``code_version`` are recorded for
    new outputs but never compared on resume: they describe how an image was made,
    not what it should be. The analysis reports runs whose records span commits.
    """
    arm_configs = arm_configs or {}
    conditions = plan_conditions(plan)
    version = code_version()
    cuda = getattr(torch_module, "cuda", None)
    timing = cuda is not None and cuda.is_available()
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    if resume:
        records = _load_resume_records(
            plan=plan,
            output_dir=output_dir,
            manifest_path=manifest_path,
            run_config=run_config,
            arm_configs=arm_configs,
        )
        if records and "scheduler_config" in run_config:
            run_config = {
                **run_config,
                "scheduler_config": records[0]["scheduler_config"],
            }
        _restore_resume_metadata(records=records, plan=plan, output_dir=output_dir)
    else:
        existing = [path for item in plan if (path := _output_path(output_dir, item)).exists()]
        if manifest_path.exists():
            existing.append(manifest_path)
        if existing and not overwrite:
            preview = ", ".join(path.as_posix() for path in existing[:3])
            raise FileExistsError(
                f"Paired-generation outputs already exist ({preview}). "
                "Use a new --output-dir, pass --resume, or pass --overwrite."
            )
        records = []

    controller = condition_controller or (
        lambda condition: _default_condition_controller(pipe, condition)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        manifest_path.write_text("", encoding="utf-8")
    completed_keys = {
        (str(record["prompt_id"]), int(record["seed"]), str(record["condition"]))
        for record in records
    }
    active_condition: str | None = None
    for item in plan:
        item_key = _generation_key(item)
        if item_key in completed_keys:
            continue
        if item.condition != active_condition:
            controller(item.condition)
            active_condition = item.condition
        generator = torch_module.Generator(device="cpu").manual_seed(item.seed)
        if timing:
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = pipe(
            item.prompt,
            generator=generator,
            guidance_scale=item.guidance_scale,
            num_inference_steps=item.num_inference_steps,
            width=item.width,
            height=item.height,
        )
        if timing:
            cuda.synchronize()
        elapsed = time.perf_counter() - started
        output_path = _output_path(output_dir, item)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output_path)
        metadata_path = _metadata_path(output_dir, item)
        record = {
            **asdict(item),
            **run_config,
            **arm_configs.get(item.condition, {}),
            "generator_device": "cpu",
            "generator_reset_per_condition": True,
            "lora_fused": False,
            "adapter_active": item.condition != BASE_CONDITION,
            "model": item.condition,
            "output_path": _relative_or_absolute(output_path, manifest_path.parent),
            "output_sha256": sha256_file(output_path),
            "metadata_path": _relative_or_absolute(metadata_path, manifest_path.parent),
            "elapsed_seconds": round(elapsed, 3),
            "peak_vram_bytes": int(cuda.max_memory_allocated()) if timing else None,
            "code_version": version,
        }
        records.append(record)
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        _write_pair_metadata(records=records, item=item, output_dir=output_dir)
        completed_keys.add(item_key)
    validate_paired_records(records, conditions)
    return records


def parse_arm(spec: str) -> tuple[str, Path, float]:
    """Parse ``LABEL=PATH`` or ``LABEL=PATH@SCALE``."""
    label, sep, rest = spec.partition("=")
    if not sep or not rest:
        raise argparse.ArgumentTypeError(f"--arm expects LABEL=PATH[@SCALE], got {spec!r}")
    path, at, scale = rest.rpartition("@") if "@" in rest else (rest, "", "")
    try:
        value = float(scale) if at else 1.0
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid scale in --arm {spec!r}") from exc
    return label, Path(path), value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    adapters = parser.add_mutually_exclusive_group(required=True)
    adapters.add_argument(
        "--lora",
        type=Path,
        help="Exact LoRA checkpoint file to evaluate (conditions: base, lora)",
    )
    adapters.add_argument(
        "--arm",
        action="append",
        type=parse_arm,
        dest="arms",
        metavar="LABEL=PATH[@SCALE]",
        help=(
            "Adapter arm, repeatable. Its training config is the config.yaml AI Toolkit "
            "writes next to the checkpoint."
        ),
    )
    parser.add_argument("--prompts", type=Path, default=Path("evaluation/prompts.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-model-revision")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/study1_pilot.yaml"),
        help="Config used to train the --lora checkpoint (ignored with --arm)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--quantization",
        choices=("none", "8bit"),
        default="none",
        help="8bit loads the FLUX transformer and T5 with bitsandbytes (fits a 24 GB GPU)",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE)
    parser.add_argument("--num-inference-steps", type=int, default=STEPS)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    existing_output_mode = parser.add_mutually_exclusive_group()
    existing_output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse manifest-backed outputs, generating only missing records",
    )
    existing_output_mode.add_argument("--overwrite", action="store_true")
    return parser


def _arm_config(label: str, path: Path, scale: float, training_config: Path) -> dict[str, Any]:
    return {
        "lora_label": label,
        "lora_path": path.as_posix(),
        "lora_resolved_path": path.resolve().as_posix(),
        "lora_scale": scale,
        "lora_sha256": sha256_file(path),
        "training_run_name": validate_training_checkpoint(training_config, path),
        "training_config_path": training_config.as_posix(),
        "training_config_resolved_path": training_config.resolve().as_posix(),
        "training_config_sha256": sha256_file(training_config),
    }


def _adapter_name(path: Path, index: int) -> str:
    return f"a{index}_" + re.sub(r"\W", "_", path.stem)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    multi_arm = args.arms is not None
    if multi_arm:
        specs = args.arms
        for _, path, _ in specs:
            if not path.is_file():
                raise FileNotFoundError(f"Adapter checkpoint does not exist: {path}")
            if not (path.parent / "config.yaml").is_file():
                raise FileNotFoundError(f"No AI Toolkit config.yaml next to {path}")
        labels = [label for label, _, _ in specs]
        validate_arm_labels(labels)
        names: dict[Path, str] = {}
        arms = []
        for label, path, scale in specs:
            name = names.setdefault(path.resolve(), _adapter_name(path, len(names)))
            arms.append(AdapterArm(label, path, scale, name))
        arm_configs = {
            arm.label: _arm_config(arm.label, arm.path, arm.scale, arm.path.parent / "config.yaml")
            for arm in arms
        }
        arm_configs[BASE_CONDITION] = {field: None for field in ARM_FIELDS} | {"lora_scale": 0.0}
    else:
        if not args.lora.is_file():
            raise FileNotFoundError(f"--lora must name the exact LoRA checkpoint file: {args.lora}")
        if not args.training_config.is_file():
            raise FileNotFoundError(f"Training config does not exist: {args.training_config}")
        labels = ["lora"]
        training_run_name = validate_training_checkpoint(args.training_config, args.lora)
        arms = [AdapterArm("lora", args.lora, args.lora_scale, training_run_name)]
        arm_configs = {}

    import torch

    before_first_load = None
    if args.quantization == "8bit":
        import diffusers
        import transformers

        from evaluation.compare_checkpoints import (
            load_pipeline,
            prepare_quantized_transformer_for_peft,
            validate_8bit_dependency_versions,
        )

        validate_8bit_dependency_versions()
        if args.base_model_revision:
            raise ValueError("--base-model-revision is not supported with --quantization 8bit")
        pipe, _ = load_pipeline(
            config=SimpleNamespace(base_model_id=args.base_model),
            quantization="8bit",
            requested_dtype=args.dtype,
            device=args.device,
            cpu_offload=args.cpu_offload,
            torch_module=torch,
            diffusers_module=diffusers,
            transformers_module=transformers,
        )
        before_first_load = lambda: prepare_quantized_transformer_for_peft(pipe.transformer)  # noqa: E731
    else:
        from diffusers import FluxPipeline

        pipeline_kwargs: dict[str, Any] = {"torch_dtype": getattr(torch, args.dtype)}
        if args.base_model_revision:
            pipeline_kwargs["revision"] = args.base_model_revision
        pipe = FluxPipeline.from_pretrained(args.base_model, **pipeline_kwargs)
        if args.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(args.device)
    pipe.set_progress_bar_config(disable=False)

    manifest_path = args.output_dir / "generation_manifest.jsonl"
    scheduler_config = _canonical_scheduler_config(getattr(pipe.scheduler, "config", {}))
    run_config: dict[str, Any] = {
        "base_model": args.base_model,
        "base_model_revision": args.base_model_revision,
        "device": args.device,
        "dtype": args.dtype,
        "prompts_path": args.prompts.as_posix(),
        "prompts_resolved_path": args.prompts.resolve().as_posix(),
        "prompts_sha256": sha256_file(args.prompts),
        "model_cpu_offload": args.cpu_offload,
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_config": scheduler_config,
        "condition_order": [BASE_CONDITION, *labels],
        "base_generated_before_lora_load": True,
        "software_versions": {
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", None),
            "diffusers": _installed_version("diffusers"),
            "transformers": _installed_version("transformers"),
            "peft": _installed_version("peft"),
        },
    }
    if args.quantization != "none":
        # Only written when used, so manifests from earlier bf16 runs still resume.
        run_config["quantization"] = args.quantization
        run_config["software_versions"]["bitsandbytes"] = _installed_version("bitsandbytes")
    if not multi_arm:
        run_config |= {
            "training_run_name": training_run_name,
            "training_config_path": args.training_config.as_posix(),
            "training_config_resolved_path": args.training_config.resolve().as_posix(),
            "training_config_sha256": sha256_file(args.training_config),
            "lora_path": args.lora.as_posix(),
            "lora_resolved_path": args.lora.resolve().as_posix(),
            "lora_scale": args.lora_scale,
            "lora_sha256": sha256_file(args.lora),
        }

    plan = build_generation_plan(
        load_prompts(args.prompts),
        seeds=args.seeds,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        width=args.width,
        height=args.height,
        arms=labels,
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
                pipe, arms=arms, before_first_load=before_first_load
            ),
            overwrite=args.overwrite,
            resume=args.resume,
            arm_configs=arm_configs,
        )
    print(f"Run contains {len(records)} outputs and wrote {manifest_path}")


if __name__ == "__main__":
    main()
