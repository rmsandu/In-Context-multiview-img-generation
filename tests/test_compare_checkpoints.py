import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from evaluation.compare_checkpoints import (
    PROMPT_IDS,
    build_manifest,
    discover_checkpoints,
    load_study_config,
    output_paths,
    run_comparison,
    write_manifest,
)


class FlowMatchEulerDiscreteScheduler:
    pass


class _FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


class _FakeTorch:
    def __init__(self) -> None:
        self.generators: list[_FakeGenerator] = []

    def Generator(self, device: str) -> _FakeGenerator:  # noqa: N802
        generator = _FakeGenerator(device)
        self.generators.append(generator)
        return generator


class _FakePipe:
    def __init__(self) -> None:
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.condition = "base"
        self.events: list[str] = []

    def load_lora_weights(self, path: str, *, adapter_name: str) -> None:
        self.condition = adapter_name
        self.events.append(f"load:{Path(path).name}")

    def set_adapters(self, adapter_name: str, *, adapter_weights: float) -> None:
        assert adapter_name == self.condition
        assert adapter_weights == 1.0

    def unload_lora_weights(self) -> None:
        self.events.append(f"unload:{self.condition}")
        self.condition = "base"

    def __call__(self, *, prompt: str, generator: _FakeGenerator, **kwargs):
        self.events.append(f"generate:{self.condition}:{prompt}")
        assert kwargs["num_inference_steps"] == 20
        assert kwargs["guidance_scale"] == 3.5
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])


def test_load_current_study_config_uses_fixed_sampling_values() -> None:
    config = load_study_config(Path("configs/study1_pilot.yaml"))

    assert config.name == "study1_pilot"
    assert config.base_model_id == "black-forest-labs/FLUX.1-dev"
    assert len(config.prompts) == 2
    assert config.prompts[0].startswith(
        "[FOUR-VIEWS] Four views of A clear plastic container"
    )
    assert config.prompts[1].startswith(
        "[FOUR-VIEWS] Four views of A light beige long-sleeved"
    )
    assert config.seed == 17
    assert config.inference_steps == 20
    assert config.guidance_scale == 3.5
    assert (config.width, config.height) == (1024, 1024)
    assert config.scheduler == "flowmatch"
    assert config.scheduler_class == "FlowMatchEulerDiscreteScheduler"
    assert config.training_steps == 500


def test_checkpoint_discovery_maps_unnumbered_final_and_reports_missing(
    tmp_path: Path,
) -> None:
    config = load_study_config(Path("configs/study1_pilot.yaml"))
    (tmp_path / "study1_pilot_000000100.safetensors").write_bytes(b"100")
    (tmp_path / "study1_pilot_000000300.safetensors").write_bytes(b"300")
    final = tmp_path / "study1_pilot.safetensors"
    final.write_bytes(b"500")

    checkpoints = discover_checkpoints(tmp_path, config)

    assert checkpoints[500] == final
    final.unlink()
    with pytest.raises(
        FileNotFoundError, match=r"Missing required checkpoint steps: 500"
    ) as exc:
        discover_checkpoints(tmp_path, config)
    assert "step 100:" in str(exc.value)
    assert "step 300:" in str(exc.value)


def test_manifest_records_controlled_settings_hashes_and_paths(tmp_path: Path) -> None:
    config_path = Path("configs/study1_pilot.yaml")
    config = load_study_config(config_path)
    checkpoints = {}
    for step in (100, 300, 500):
        checkpoint = tmp_path / f"checkpoint_{step}.safetensors"
        checkpoint.write_bytes(f"weights-{step}".encode())
        checkpoints[step] = checkpoint

    paths = output_paths(tmp_path / "results", PROMPT_IDS[0])
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        prompt_id=PROMPT_IDS[0],
        prompt=config.prompts[0],
        checkpoint_paths=checkpoints,
        paths=paths,
        scheduler_class="FlowMatchEulerDiscreteScheduler",
    )
    write_manifest(paths["manifest"], manifest)
    saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert saved["prompt"] == config.prompts[0]
    assert saved["seed"] == 17
    assert saved["inference_steps"] == 20
    assert saved["guidance_scale"] == 3.5
    assert saved["resolution"] == {"width": 1024, "height": 1024}
    assert saved["scheduler_class"] == "FlowMatchEulerDiscreteScheduler"
    assert saved["base_model_id"] == "black-forest-labs/FLUX.1-dev"
    assert set(saved["checkpoints"]) == {"100", "300", "500"}
    assert all(len(item["sha256"]) == 64 for item in saved["checkpoints"].values())
    assert saved["output_paths"]["base"].endswith("snack_container/base.png")
    assert saved["output_paths"]["comparison"].endswith(
        "snack_container/comparison.png"
    )
    assert saved["output_paths"]["manifest"].endswith("snack_container/manifest.json")
    assert saved["generator"] == {
        "device": "cpu",
        "reset_before_every_generation": True,
    }
    assert saved["lora_fused"] is False


def test_generation_order_resets_seed_and_unloads_each_checkpoint(
    tmp_path: Path,
) -> None:
    config = replace(
        load_study_config(Path("configs/study1_pilot.yaml")),
        prompts=("prompt-a", "prompt-b"),
        width=8,
        height=8,
    )
    checkpoints = {}
    for step in (100, 300, 500):
        checkpoint = tmp_path / f"study1_pilot_{step}.safetensors"
        checkpoint.write_bytes(str(step).encode())
        checkpoints[step] = checkpoint
    pipe = _FakePipe()
    torch_module = _FakeTorch()

    run_comparison(
        pipe=pipe,
        torch_module=torch_module,
        config=config,
        config_path=Path("configs/study1_pilot.yaml"),
        checkpoints=checkpoints,
        output_dir=tmp_path / "results",
    )

    assert pipe.events[:2] == [
        "generate:base:prompt-a",
        "generate:base:prompt-b",
    ]
    assert pipe.events[2:] == [
        "load:study1_pilot_100.safetensors",
        "generate:study1_step_100:prompt-a",
        "generate:study1_step_100:prompt-b",
        "unload:study1_step_100",
        "load:study1_pilot_300.safetensors",
        "generate:study1_step_300:prompt-a",
        "generate:study1_step_300:prompt-b",
        "unload:study1_step_300",
        "load:study1_pilot_500.safetensors",
        "generate:study1_step_500:prompt-a",
        "generate:study1_step_500:prompt-b",
        "unload:study1_step_500",
    ]
    assert len(torch_module.generators) == 8
    assert all(generator.device == "cpu" for generator in torch_module.generators)
    assert all(generator.seed == 17 for generator in torch_module.generators)
