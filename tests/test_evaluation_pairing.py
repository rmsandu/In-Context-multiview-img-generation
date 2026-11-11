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


class _InterruptingPipe(_FakePipe):
    def __init__(self, successful_calls: int) -> None:
        super().__init__()
        self.successful_calls = successful_calls
        self.calls = 0

    def __call__(self, prompt: str, *, generator: _FakeGenerator, **kwargs):
        if self.calls >= self.successful_calls:
            raise RuntimeError("simulated interruption")
        self.calls += 1
        return super().__call__(prompt, generator=generator, **kwargs)


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


def test_generation_resumes_only_missing_verified_outputs(tmp_path: Path) -> None:
    plan = build_generation_plan([_prompt()], seeds=[1001, 1002])
    manifest = tmp_path / "generation_manifest.jsonl"
    run_config = {"base_model": "test", "lora_path": "test.safetensors"}

    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate(
            _InterruptingPipe(successful_calls=3),
            plan,
            torch_module=_FakeTorch(),
            output_dir=tmp_path,
            manifest_path=manifest,
            run_config=run_config,
        )

    existing_paths = [
        tmp_path / "images" / "object" / "1001" / "base.png",
        tmp_path / "images" / "object" / "1002" / "base.png",
        tmp_path / "images" / "object" / "1001" / "lora.png",
    ]
    existing_payloads = {path: path.read_bytes() for path in existing_paths}
    orphan = tmp_path / "images" / "object" / "1002" / "lora.png"
    orphan.write_bytes(b"unmanifested partial output")
    fake_torch = _FakeTorch()

    records = generate(
        _FakePipe(),
        plan,
        torch_module=fake_torch,
        output_dir=tmp_path,
        manifest_path=manifest,
        run_config=run_config,
        resume=True,
    )

    assert len(records) == 4
    assert len(fake_torch.generators) == 1
    assert fake_torch.generators[0].seed == 1002
    assert all(path.read_bytes() == payload for path, payload in existing_payloads.items())
    assert orphan.read_bytes() != b"unmanifested partial output"
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 4
    metadata = json.loads(
        (tmp_path / "images" / "object" / "1002" / "metadata.json").read_text(encoding="utf-8")
    )
    assert set(metadata["outputs"]) == {"base", "lora"}


def test_resume_rejects_changed_run_configuration(tmp_path: Path) -> None:
    plan = build_generation_plan([_prompt()], seeds=[1001])
    manifest = tmp_path / "generation_manifest.jsonl"
    generate(
        _FakePipe(),
        plan,
        torch_module=_FakeTorch(),
        output_dir=tmp_path,
        manifest_path=manifest,
        run_config={"base_model": "test", "lora_path": "first.safetensors"},
    )

    with pytest.raises(ValueError, match="does not match the current run"):
        generate(
            _FakePipe(),
            plan,
            torch_module=_FakeTorch(),
            output_dir=tmp_path,
            manifest_path=manifest,
            run_config={"base_model": "test", "lora_path": "different.safetensors"},
            resume=True,
        )


def test_resume_accepts_reordered_scheduler_default_metadata(tmp_path: Path) -> None:
    plan = build_generation_plan([_prompt()], seeds=[1001])
    manifest = tmp_path / "generation_manifest.jsonl"
    saved_scheduler = {
        "_use_default_values": ["use_karras_sigmas", "invert_sigmas"],
        "shift": 3.0,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate(
            _InterruptingPipe(successful_calls=1),
            plan,
            torch_module=_FakeTorch(),
            output_dir=tmp_path,
            manifest_path=manifest,
            run_config={
                "base_model": "test",
                "lora_path": "test.safetensors",
                "scheduler_config": saved_scheduler,
            },
        )

    records = generate(
        _FakePipe(),
        plan,
        torch_module=_FakeTorch(),
        output_dir=tmp_path,
        manifest_path=manifest,
        run_config={
            "base_model": "test",
            "lora_path": "test.safetensors",
            "scheduler_config": {
                "_use_default_values": ["invert_sigmas", "use_karras_sigmas"],
                "shift": 3.0,
            },
        },
        resume=True,
    )

    assert len(records) == 2
    assert all(record["scheduler_config"] == saved_scheduler for record in records)
    with pytest.raises(ValueError, match="scheduler_config"):
        generate(
            _FakePipe(),
            plan,
            torch_module=_FakeTorch(),
            output_dir=tmp_path,
            manifest_path=manifest,
            run_config={
                "base_model": "test",
                "lora_path": "test.safetensors",
                "scheduler_config": {
                    "_use_default_values": ["invert_sigmas", "use_karras_sigmas"],
                    "shift": 4.0,
                },
            },
            resume=True,
        )


def test_resume_rejects_modified_manifest_backed_output(tmp_path: Path) -> None:
    plan = build_generation_plan([_prompt()], seeds=[1001])
    manifest = tmp_path / "generation_manifest.jsonl"
    run_config = {"base_model": "test", "lora_path": "test.safetensors"}
    generate(
        _FakePipe(),
        plan,
        torch_module=_FakeTorch(),
        output_dir=tmp_path,
        manifest_path=manifest,
        run_config=run_config,
    )
    (tmp_path / "images" / "object" / "1001" / "base.png").write_bytes(b"modified")

    with pytest.raises(ValueError, match="output hash does not match"):
        generate(
            _FakePipe(),
            plan,
            torch_module=_FakeTorch(),
            output_dir=tmp_path,
            manifest_path=manifest,
            run_config=run_config,
            resume=True,
        )


def test_resume_and_overwrite_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate(
            _FakePipe(),
            build_generation_plan([_prompt()], seeds=[1001]),
            torch_module=_FakeTorch(),
            output_dir=tmp_path,
            manifest_path=tmp_path / "generation_manifest.jsonl",
            run_config={"base_model": "test", "lora_path": "test.safetensors"},
            resume=True,
            overwrite=True,
        )


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

    with pytest.raises(ValueError, match="one output per condition"):
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


def test_multi_arm_plan_shares_one_base_phase(tmp_path: Path) -> None:
    from evaluation.generate_pairs import AdapterArm

    plan = build_generation_plan([_prompt()], seeds=[1001, 1002], arms=["r1", "s0.5"])
    assert [item.condition for item in plan] == ["base", "base", "r1", "r1", "s0.5", "s0.5"]
    pipe = _FakePipe()
    arms = [
        AdapterArm("r1", Path("run1/run1.safetensors"), 1.0, "a0_run1"),
        AdapterArm("s0.5", Path("run1/run1.safetensors"), 0.5, "a0_run1"),
    ]
    arm_configs = {
        "base": {"lora_label": None, "lora_scale": 0.0},
        "r1": {"lora_label": "r1", "lora_scale": 1.0},
        "s0.5": {"lora_label": "s0.5", "lora_scale": 0.5},
    }
    records = generate(
        pipe,
        plan,
        torch_module=_FakeTorch(),
        output_dir=tmp_path,
        manifest_path=tmp_path / "generation_manifest.jsonl",
        run_config={"base_model": "test"},
        condition_controller=build_condition_controller(pipe, arms=arms),
        arm_configs=arm_configs,
    )
    # The shared file is loaded once; the sweep only changes the adapter weight.
    assert pipe.events == [
        "load_lora:run1/run1.safetensors:a0_run1",
        "set_adapter:a0_run1:1.0",
        "set_adapter:a0_run1:0.5",
    ]
    assert len(records) == 6
    assert {r["condition"]: r["lora_scale"] for r in records} == {
        "base": 0.0,
        "r1": 1.0,
        "s0.5": 0.5,
    }
    assert all("code_version" in r and "elapsed_seconds" in r for r in records)
    metadata = json.loads((tmp_path / "images/object/1001/metadata.json").read_text())
    assert set(metadata["outputs"]) == {"base", "r1", "s0.5"}
    assert metadata["outputs"]["s0.5"]["lora_scale"] == 0.5
    assert "lora_scale" not in metadata["generation_parameters"]


def test_multi_arm_resume_ignores_timing_and_code_version(tmp_path: Path) -> None:
    plan = build_generation_plan([_prompt()], seeds=[1001], arms=["r1", "r2"])
    arm_configs = {
        "base": {"lora_label": None},
        "r1": {"lora_label": "r1"},
        "r2": {"lora_label": "r2"},
    }
    kwargs = {
        "torch_module": _FakeTorch(),
        "output_dir": tmp_path,
        "manifest_path": tmp_path / "generation_manifest.jsonl",
        "run_config": {"base_model": "test"},
        "arm_configs": arm_configs,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate(_InterruptingPipe(successful_calls=2), plan, **kwargs)
    manifest = kwargs["manifest_path"]
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:  # a later commit and a different machine speed must not block resume
        row["code_version"] = {"git_commit": "other", "python_sources_dirty": True}
        row["elapsed_seconds"] = 999.0
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    pipe = _FakePipe()
    records = generate(pipe, plan, resume=True, **kwargs)
    assert sorted(r["condition"] for r in records) == ["base", "r1", "r2"]
    # Changing an arm's identity is still caught.
    with pytest.raises(ValueError, match="lora_label"):
        generate(
            _FakePipe(),
            plan,
            resume=True,
            **(kwargs | {"arm_configs": {**arm_configs, "r2": {"lora_label": "x"}}}),
        )


def test_controller_reenables_adapters_after_base() -> None:
    from evaluation.generate_pairs import AdapterArm

    pipe = _FakePipe()
    controller = build_condition_controller(
        pipe,
        arms=[AdapterArm("r1", Path("a.safetensors"), 1.0, "a0")],
        before_first_load=lambda: pipe.events.append("prep"),
    )
    controller("base")
    controller("r1")
    controller("base")
    controller("r1")
    assert pipe.events == [
        "prep",
        "load_lora:a.safetensors:a0",
        "set_adapter:a0:1.0",
        "disable_lora",
        "enable_lora",
        "set_adapter:a0:1.0",
    ]


@pytest.mark.parametrize("labels", [[], ["base"], ["r1", "r1"], ["bad/label"]])
def test_invalid_arm_labels_are_rejected(labels) -> None:
    with pytest.raises(ValueError):
        build_generation_plan([_prompt()], arms=labels)


def test_parse_arm_spec() -> None:
    from evaluation.generate_pairs import parse_arm

    assert parse_arm("r1=output/run/run.safetensors") == (
        "r1",
        Path("output/run/run.safetensors"),
        1.0,
    )
    assert parse_arm("s0.25=output/run/run.safetensors@0.25") == (
        "s0.25",
        Path("output/run/run.safetensors"),
        0.25,
    )


def test_blinded_scorecard_selects_one_arm_and_writes_pairwise_sheet(tmp_path: Path) -> None:
    from evaluation.create_blinded_scorecard import load_generation_manifest

    images = tmp_path / "generated"
    images.mkdir()
    records = []
    for prompt_id in ("a", "b", "c"):
        for seed in (1001, 1002):
            for condition in ("base", "r1", "r2"):
                path = images / f"{prompt_id}_{seed}_{condition}.png"
                path.write_bytes(f"{prompt_id}{seed}{condition}".encode())
                records.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_group": "g",
                        "prompt": f"p-{prompt_id}",
                        "seed": seed,
                        "condition": condition,
                        "guidance_scale": 3.5,
                        "num_inference_steps": 20,
                        "width": 1024,
                        "height": 1024,
                        "output_path": path.name,
                    }
                )
    manifest = images / "generation_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    selected = load_generation_manifest(manifest, arm="r2", seeds=[1001])
    assert {(r["condition"], r["seed"]) for r in selected} == {("base", 1001), ("r2", 1001)}
    with pytest.raises(ValueError, match="available"):
        load_generation_manifest(manifest, arm="r9")

    _, key = create_scorecard(
        selected, generation_manifest=manifest, output_dir=tmp_path / "blind", arm="r2"
    )
    with (tmp_path / "blind" / "pairwise.csv").open(newline="") as handle:
        pairwise = list(csv.DictReader(handle))
    assert len(pairwise) == 3 and pairwise[0]["same_object_choice"] == ""
    keys = [json.loads(line) for line in key.read_text().splitlines()]
    assert {k["condition"] for k in keys} == {"base", "r2"}
    assert all((tmp_path / "blind" / row["image_A"]).is_file() for row in pairwise)
