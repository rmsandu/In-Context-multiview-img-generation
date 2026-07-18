import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from evaluation.compare_checkpoints import (
    PROMPT_ID,
    SNACK_CONTAINER_PROMPT,
    build_manifest,
    build_parser,
    discover_checkpoints,
    load_and_verify_lora,
    load_pipeline,
    load_study_config,
    output_paths,
    prepare_quantized_transformer_for_peft,
    reusable_image_record,
    run_comparison,
    validate_8bit_dependency_versions,
    validate_runtime_options,
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
    bfloat16 = "torch.bfloat16"
    float16 = "torch.float16"

    def __init__(self) -> None:
        self.generators: list[_FakeGenerator] = []

    def Generator(self, device: str) -> _FakeGenerator:  # noqa: N802
        generator = _FakeGenerator(device)
        self.generators.append(generator)
        return generator


class _FakePipe:
    def __init__(self) -> None:
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.transformer = SimpleNamespace(is_loaded_in_8bit=False)
        self.condition = "base"
        self.events: list[str] = []
        self.adapters: list[str] = []

    def load_lora_weights(self, path: str, *, adapter_name: str) -> None:
        self.condition = adapter_name
        self.adapters.append(adapter_name)
        self.events.append(f"load:{Path(path).name}")

    def set_adapters(self, adapter_name: str, *, adapter_weights: float) -> None:
        assert adapter_name == self.condition
        assert adapter_weights == 1.0

    def unload_lora_weights(self) -> None:
        self.events.append(f"unload:{self.condition}")
        self.adapters.clear()
        self.condition = "base"

    def get_list_adapters(self) -> dict[str, list[str]]:
        return {"transformer": list(self.adapters)}

    def __call__(self, *, prompt: str, generator: _FakeGenerator, **kwargs):
        self.events.append(f"generate:{self.condition}:{prompt}")
        assert kwargs["num_inference_steps"] == 20
        assert kwargs["guidance_scale"] == 3.5
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])


def test_load_current_study_config_uses_fixed_sampling_values() -> None:
    config = load_study_config(Path("configs/study1_pilot.yaml"))

    assert config.name == "study1_pilot"
    assert config.base_model_id == "black-forest-labs/FLUX.1-dev"
    assert config.prompt == SNACK_CONTAINER_PROMPT
    assert config.seed == 17
    assert config.inference_steps == 20
    assert config.guidance_scale == 3.5
    assert (config.width, config.height) == (1024, 1024)
    assert config.scheduler == "flowmatch"
    assert config.scheduler_class == "FlowMatchEulerDiscreteScheduler"
    assert config.training_steps == 500
    assert config.train_text_encoder is False


def test_config_monitor_prompts_are_independent_from_comparison_prompt(
    tmp_path: Path,
) -> None:
    source = Path("configs/study1_pilot.yaml").read_text(encoding="utf-8")
    start = source.index("        prompts:\n")
    end = source.index("        neg:", start)
    changed = source[:start] + "        prompts:\n          - a new monitor prompt\n" + source[end:]
    config_path = tmp_path / "changed.yaml"
    config_path.write_text(changed, encoding="utf-8")

    config = load_study_config(config_path)

    assert config.prompt == SNACK_CONTAINER_PROMPT


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

    paths = output_paths(tmp_path / "results")
    manifest = build_manifest(
        config=config,
        config_path=config_path,
        checkpoint_paths=checkpoints,
        paths=paths,
        scheduler_class="FlowMatchEulerDiscreteScheduler",
        quantization="8bit",
        dtype="bfloat16",
        gpu="NVIDIA GeForce RTX 4090",
    )
    write_manifest(paths["manifest"], manifest)
    saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert saved["prompt_id"] == PROMPT_ID
    assert saved["prompt"] == SNACK_CONTAINER_PROMPT
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
    assert saved["quantization"] == "8bit"
    assert saved["dtype"] == "bfloat16"
    assert saved["gpu"] == "NVIDIA GeForce RTX 4090"
    assert saved["train_text_encoder"] is False
    assert saved["images"] == {}


def test_generation_order_resets_seed_and_unloads_each_checkpoint(
    tmp_path: Path,
) -> None:
    config = replace(
        load_study_config(Path("configs/study1_pilot.yaml")),
        prompt="prompt-a",
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
        quantization="none",
        dtype="bfloat16",
        gpu="fake-gpu",
        resume=False,
    )

    assert pipe.events == [
        "generate:base:prompt-a",
        "load:study1_pilot_100.safetensors",
        "generate:study1_step_100:prompt-a",
        "unload:study1_step_100",
        "load:study1_pilot_300.safetensors",
        "generate:study1_step_300:prompt-a",
        "unload:study1_step_300",
        "load:study1_pilot_500.safetensors",
        "generate:study1_step_500:prompt-a",
        "unload:study1_step_500",
    ]
    assert len(torch_module.generators) == 4
    assert all(generator.device == "cpu" for generator in torch_module.generators)
    assert all(generator.seed == 17 for generator in torch_module.generators)
    result_dir = tmp_path / "results" / "snack_container"
    assert {path.name for path in result_dir.iterdir()} == {
        "base.png",
        "checkpoint_100.png",
        "checkpoint_300.png",
        "checkpoint_500.png",
        "comparison.png",
        "manifest.json",
        "generation.log",
    }
    with Image.open(result_dir / "comparison.png") as comparison:
        assert comparison.size == (32, 72)
    assert not (tmp_path / "results" / "baby_sweatshirt").exists()


def test_generation_rejects_scheduler_mismatch_before_creating_outputs(
    tmp_path: Path,
) -> None:
    config = replace(
        load_study_config(Path("configs/study1_pilot.yaml")), width=8, height=8
    )
    pipe = _FakePipe()
    pipe.scheduler = object()

    with pytest.raises(ValueError, match="loaded base pipeline uses object"):
        run_comparison(
            pipe=pipe,
            torch_module=_FakeTorch(),
            config=config,
            config_path=Path("configs/study1_pilot.yaml"),
            checkpoints={},
            output_dir=tmp_path / "results",
            quantization="none",
            dtype="bfloat16",
            gpu="fake-gpu",
            resume=False,
        )

    assert pipe.events == []
    assert not (tmp_path / "results").exists()


def test_cli_quantization_defaults_and_rejects_8bit_cpu_offload() -> None:
    parser = build_parser()

    assert parser.parse_args([]).quantization == "none"
    assert parser.parse_args([]).output_dir is None
    assert parser.parse_args(["--quantization", "8bit", "--resume"]).resume is True
    with pytest.raises(ValueError, match="cannot be combined with --cpu-offload"):
        validate_runtime_options(quantization="8bit", cpu_offload=True)


def test_installed_8bit_dependencies_are_compatible() -> None:
    validate_8bit_dependency_versions()


def test_8bit_pipeline_avoids_dispatching_quantized_components_with_to() -> None:
    calls: list[tuple[str, str, dict]] = []

    class _BitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FluxTransformer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls.append(("transformer", model_id, kwargs))
            return SimpleNamespace(is_loaded_in_8bit=True)

    class _T5Encoder:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls.append(("text_encoder_2", model_id, kwargs))
            return SimpleNamespace(is_loaded_in_8bit=True)

    class _FluxPipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls.append(("pipeline", model_id, kwargs))
            return SimpleNamespace(
                transformer=kwargs["transformer"],
                text_encoder_2=kwargs["text_encoder_2"],
            )

    diffusers_module = SimpleNamespace(
        BitsAndBytesConfig=_BitsAndBytesConfig,
        FluxTransformer2DModel=_FluxTransformer,
        FluxPipeline=_FluxPipeline,
    )
    transformers_module = SimpleNamespace(
        BitsAndBytesConfig=_BitsAndBytesConfig,
        T5EncoderModel=_T5Encoder,
    )

    pipe, dtype = load_pipeline(
        config=load_study_config(Path("configs/study1_pilot.yaml")),
        quantization="8bit",
        requested_dtype="float16",
        device="cuda",
        cpu_offload=False,
        torch_module=_FakeTorch(),
        diffusers_module=diffusers_module,
        transformers_module=transformers_module,
    )

    assert dtype == "bfloat16"
    assert pipe.transformer.is_loaded_in_8bit is True
    assert pipe.text_encoder_2.is_loaded_in_8bit is True
    assert [call[0] for call in calls] == ["transformer", "text_encoder_2", "pipeline"]
    for _, model_id, kwargs in calls:
        assert model_id == "black-forest-labs/FLUX.1-dev"
        assert kwargs["torch_dtype"] == "torch.bfloat16"
    assert "device_map" not in calls[0][2]
    assert "device_map" not in calls[1][2]
    assert calls[2][2]["device_map"] == "auto"
    assert calls[0][2]["subfolder"] == "transformer"
    assert calls[1][2]["subfolder"] == "text_encoder_2"
    assert calls[0][2]["quantization_config"].kwargs == {"load_in_8bit": True}
    assert calls[1][2]["quantization_config"].kwargs == {"load_in_8bit": True}


def test_quantized_lora_must_attach_to_transformer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"weights")
    pipe = _FakePipe()
    pipe.transformer.is_loaded_in_8bit = True

    load_and_verify_lora(
        pipe,
        checkpoint,
        adapter_name="step_100",
        quantization="8bit",
    )

    assert pipe.get_list_adapters() == {"transformer": ["step_100"]}

    class _MissingTransformerAdapterPipe(_FakePipe):
        def get_list_adapters(self) -> dict[str, list[str]]:
            return {"transformer": []}

    missing = _MissingTransformerAdapterPipe()
    missing.transformer.is_loaded_in_8bit = True
    with pytest.raises(RuntimeError, match="did not attach to the transformer"):
        load_and_verify_lora(
            missing,
            checkpoint,
            adapter_name="step_100",
            quantization="8bit",
        )
    assert missing.adapters == []


def test_peft_compatibility_restores_missing_bnb_state_field() -> None:
    Linear8bitLt = type(
        "Linear8bitLt",
        (),
        {"state": SimpleNamespace(), "index": None},
    )
    layer = Linear8bitLt()
    transformer = SimpleNamespace(modules=lambda: [layer])

    assert prepare_quantized_transformer_for_peft(transformer) == 1
    assert layer.state.memory_efficient_backward is False
    assert prepare_quantized_transformer_for_peft(transformer) == 0


def test_resume_reuses_only_matching_outputs(tmp_path: Path) -> None:
    config_path = Path("configs/study1_pilot.yaml")
    config = replace(load_study_config(config_path), width=8, height=8)
    checkpoints = {}
    for step in (100, 300, 500):
        path = tmp_path / f"study1_pilot_{step}.safetensors"
        path.write_bytes(str(step).encode())
        checkpoints[step] = path
    output_dir = tmp_path / "results"

    first_pipe = _FakePipe()
    first_torch = _FakeTorch()
    run_comparison(
        pipe=first_pipe,
        torch_module=first_torch,
        config=config,
        config_path=config_path,
        checkpoints=checkpoints,
        output_dir=output_dir,
        quantization="none",
        dtype="bfloat16",
        gpu="fake-gpu",
        resume=False,
    )
    assert len(first_torch.generators) == 4

    resumed_pipe = _FakePipe()
    resumed_torch = _FakeTorch()
    run_comparison(
        pipe=resumed_pipe,
        torch_module=resumed_torch,
        config=config,
        config_path=config_path,
        checkpoints=checkpoints,
        output_dir=output_dir,
        quantization="none",
        dtype="bfloat16",
        gpu="fake-gpu",
        resume=True,
    )
    assert resumed_torch.generators == []
    manifest_path = output_dir / "snack_container" / "manifest.json"
    resumed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert all(record["reused"] for record in resumed_manifest["images"].values())

    checkpoints[100].write_bytes(b"changed-100")
    changed_pipe = _FakePipe()
    changed_torch = _FakeTorch()
    run_comparison(
        pipe=changed_pipe,
        torch_module=changed_torch,
        config=config,
        config_path=config_path,
        checkpoints=checkpoints,
        output_dir=output_dir,
        quantization="none",
        dtype="bfloat16",
        gpu="fake-gpu",
        resume=True,
    )
    assert len(changed_torch.generators) == 1
    changed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert changed_manifest["images"]["checkpoint_100"]["reused"] is False
    assert changed_manifest["images"]["base"]["reused"] is True
    assert changed_manifest["images"]["checkpoint_300"]["reused"] is True
    assert changed_manifest["images"]["checkpoint_500"]["reused"] is True
    assert len(changed_manifest["images"]["checkpoint_100"]["checkpoint_sha256"]) == 64

    log_records = [
        json.loads(line)
        for line in (output_dir / "snack_container" / "generation.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    image_records = [
        record for record in log_records if record["event"].startswith("image_")
    ]
    assert image_records
    assert all("elapsed_seconds" in record for record in image_records)
    assert all("peak_vram_bytes" in record for record in image_records)
    assert all(record["gpu"] == "fake-gpu" for record in image_records)

    expected_8bit = build_manifest(
        config=config,
        config_path=config_path,
        checkpoint_paths=checkpoints,
        paths=output_paths(output_dir),
        scheduler_class="FlowMatchEulerDiscreteScheduler",
        quantization="8bit",
        dtype="bfloat16",
        gpu="fake-gpu",
    )
    assert (
        reusable_image_record(
            existing_manifest=changed_manifest,
            expected_manifest=expected_8bit,
            condition="base",
            output_path=output_paths(output_dir)["base"],
            expected_size=(8, 8),
        )
        is None
    )

    expected_none = build_manifest(
        config=config,
        config_path=config_path,
        checkpoint_paths=checkpoints,
        paths=output_paths(output_dir),
        scheduler_class="FlowMatchEulerDiscreteScheduler",
        quantization="none",
        dtype="bfloat16",
        gpu="fake-gpu",
    )
    controlled_mismatches = {
        "prompt": "different prompt",
        "seed": 999,
        "inference_steps": 99,
        "guidance_scale": 1.0,
        "resolution": {"width": 16, "height": 16},
    }
    for field, mismatched_value in controlled_mismatches.items():
        mismatched_manifest = {**changed_manifest, field: mismatched_value}
        assert (
            reusable_image_record(
                existing_manifest=mismatched_manifest,
                expected_manifest=expected_none,
                condition="base",
                output_path=output_paths(output_dir)["base"],
                expected_size=(8, 8),
            )
            is None
        )
