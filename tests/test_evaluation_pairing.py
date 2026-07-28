import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.create_blinded_scorecard import SCORE_FIELDS, create_scorecard
from evaluation.generate_pairs import (
    build_condition_controller,
    build_generation_plan,
    generate,
    load_prompts,
    validate_paired_records,
    validate_training_checkpoint,
)


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


class _FakeImage:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def save(self, path: Path) -> None:
        path.write_bytes(self.payload)


class _FakePipe:
    def __init__(self) -> None:
        self.condition = "base"
        self.events: list[str] = []

    def disable_lora(self) -> None:
        self.events.append("disable_lora")
        self.condition = "base"

    def enable_lora(self) -> None:
        self.events.append("enable_lora")
        self.condition = "lora"

    def load_lora_weights(self, path: str, *, adapter_name: str) -> None:
        self.events.append(f"load_lora:{path}:{adapter_name}")

    def set_adapters(self, adapter_name: str, *, adapter_weights: list[float]) -> None:
        self.events.append(f"set_adapter:{adapter_name}:{adapter_weights[0]}")
        self.condition = "lora"

    def __call__(self, prompt: str, *, generator: _FakeGenerator, **kwargs):
        payload = f"{self.condition}:{generator.seed}:{prompt}:{kwargs}".encode()
        return SimpleNamespace(images=[_FakeImage(payload)])


def _prompt(prompt_id: str = "object") -> dict[str, str]:
    return {
        "id": prompt_id,
        "group": "synthetic_pose_conditioned",
        "prompt": (
            "[FOUR-VIEWS] object; [TOP-LEFT] front; [TOP-RIGHT] side; "
            "[BOTTOM-LEFT] rear; [BOTTOM-RIGHT] other side"
        ),
    }


def test_prompt_file_has_eight_structured_prompts() -> None:
    prompts = load_prompts(Path("evaluation/prompts.jsonl"))

    assert len(prompts) == 8
    assert len({prompt["id"] for prompt in prompts}) == 8
    assert {prompt["group"] for prompt in prompts} == {"synthetic_pose_conditioned"}


def test_holdout_prompt_file_has_grouped_prompts() -> None:
    prompts = load_prompts(Path("evaluation/prompts_holdout.jsonl"))

    assert len(prompts) == 42
    assert len({prompt["id"] for prompt in prompts}) == 42
    assert {prompt["group"] for prompt in prompts} == {
        "holdout_packaging_containers",
        "holdout_soft_goods_decor",
        "holdout_tools_sports",
        "holdout_toys_vehicles",
    }


def test_generation_plan_is_paired_and_resets_generator(tmp_path: Path) -> None:
    plan = build_generation_plan([_prompt()])
    fake_torch = _FakeTorch()
    manifest = tmp_path / "generation_manifest.jsonl"

    records = generate(
        _FakePipe(),
        plan,
        torch_module=fake_torch,
        output_dir=tmp_path,
        manifest_path=manifest,
        run_config={"base_model": "test", "lora_path": "test.safetensors"},
    )

    assert len(records) == 4
    assert [generator.seed for generator in fake_torch.generators] == [1001, 1002, 1001, 1002]
    assert len({id(generator) for generator in fake_torch.generators}) == 4
    assert all(record["generator_reset_per_condition"] for record in records)
    assert {record["model"] for record in records} == {"base", "lora"}
    assert len(manifest.read_text().splitlines()) == 4
    pair_dir = tmp_path / "images" / "object" / "1001"
    assert (pair_dir / "base.png").is_file()
    assert (pair_dir / "lora.png").is_file()
    metadata = json.loads((pair_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["prompt_group"] == "synthetic_pose_conditioned"
    assert metadata["generation_parameters"]["guidance_scale"] == 3.5
    assert set(metadata["outputs"]) == {"base", "lora"}
    assert metadata["outputs"]["base"]["adapter_active"] is False
    assert metadata["outputs"]["lora"]["adapter_active"] is True


def test_seed_zero_is_deterministic_and_not_treated_as_random(tmp_path: Path) -> None:
    plan = build_generation_plan([_prompt()], seeds=[0])
    fake_torch = _FakeTorch()

    generate(
        _FakePipe(),
        plan,
        torch_module=fake_torch,
        output_dir=tmp_path,
        manifest_path=tmp_path / "generation_manifest.jsonl",
        run_config={"base_model": "test", "lora_path": "test.safetensors"},
    )

    assert [generator.seed for generator in fake_torch.generators] == [0, 0]
    assert (tmp_path / "images" / "object" / "0" / "metadata.json").is_file()


def test_condition_controller_generates_base_before_loading_exact_lora() -> None:
    pipe = _FakePipe()
    controller = build_condition_controller(
        pipe,
        lora_path=Path("output/study1_pilot/study1_pilot.safetensors"),
        lora_scale=0.75,
    )

    controller("base")
    assert pipe.events == []

    controller("lora")
    assert pipe.events == [
        "load_lora:output/study1_pilot/study1_pilot.safetensors:study1_pilot",
        "set_adapter:study1_pilot:0.75",
    ]


def test_training_config_must_match_exact_checkpoint_name(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("config:\n  name: study1_pilot\n", encoding="utf-8")

    assert (
        validate_training_checkpoint(
            config, Path("output/study1_pilot/study1_pilot_000000300.safetensors")
        )
        == "study1_pilot"
    )
    with pytest.raises(ValueError, match="does not match training run"):
        validate_training_checkpoint(
            config, Path("output/study1_pilot_v2/study1_pilot_v2.safetensors")
        )


def test_generation_refuses_to_overwrite_existing_results(tmp_path: Path) -> None:
    plan = build_generation_plan([_prompt()], seeds=[1001])
    kwargs = {
        "torch_module": _FakeTorch(),
        "output_dir": tmp_path,
        "manifest_path": tmp_path / "generation_manifest.jsonl",
        "run_config": {"base_model": "test", "lora_path": "test.safetensors"},
    }
    generate(_FakePipe(), plan, **kwargs)

    with pytest.raises(FileExistsError, match="--overwrite"):
        generate(_FakePipe(), plan, **kwargs)


def test_pair_validation_rejects_missing_condition() -> None:
    record = {
        "prompt_id": "object",
        "prompt_group": "synthetic_pose_conditioned",
        "prompt": "prompt",
        "seed": 1001,
        "condition": "base",
        "guidance_scale": 3.5,
        "num_inference_steps": 20,
        "width": 1024,
        "height": 1024,
    }

    with pytest.raises(ValueError, match="one base and one LoRA"):
        validate_paired_records([record])


def test_blinded_scorecard_hides_condition_and_has_scoring_fields(tmp_path: Path) -> None:
    images = tmp_path / "generated"
    images.mkdir()
    records = []
    for condition in ("base", "lora"):
        path = images / f"object_{condition}.png"
        path.write_bytes(condition.encode())
        records.append(
            {
                "prompt_id": "object",
                "prompt_group": "synthetic_pose_conditioned",
                "prompt": "attribute-rich prompt",
                "seed": 1001,
                "condition": condition,
                "guidance_scale": 3.5,
                "num_inference_steps": 20,
                "width": 1024,
                "height": 1024,
                "output_path": path.name,
            }
        )
    generation_manifest = images / "generation_manifest.jsonl"
    generation_manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    scorecard, key = create_scorecard(
        records,
        generation_manifest=generation_manifest,
        output_dir=tmp_path / "blinded",
        blind_seed=17,
    )
    with scorecard.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert {row["blind_label"] for row in rows} == {"A", "B"}
    assert {row["prompt_group"] for row in rows} == {"synthetic_pose_conditioned"}
    assert all("base" not in row["image_path"] and "lora" not in row["image_path"] for row in rows)
    assert all(field in rows[0] for field in SCORE_FIELDS)
    assert "condition" not in rows[0]
    assert len(key.read_text().splitlines()) == 2
