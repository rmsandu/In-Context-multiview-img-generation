import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.create_blinded_scorecard import SCORE_FIELDS, create_scorecard
from evaluation.generate_pairs import (
    build_generation_plan,
    generate,
    load_prompts,
    validate_paired_records,
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

    def disable_lora(self) -> None:
        self.condition = "base"

    def enable_lora(self) -> None:
        self.condition = "lora"

    def __call__(self, prompt: str, *, generator: _FakeGenerator, **kwargs):
        payload = f"{self.condition}:{generator.seed}:{prompt}:{kwargs}".encode()
        return SimpleNamespace(images=[_FakeImage(payload)])


def _prompt(prompt_id: str = "object") -> dict[str, str]:
    return {
        "id": prompt_id,
        "prompt": (
            "[FOUR-VIEWS] object; [TOP-LEFT] front; [TOP-RIGHT] side; "
            "[BOTTOM-LEFT] rear; [BOTTOM-RIGHT] other side"
        ),
    }


def test_prompt_file_has_eight_structured_prompts() -> None:
    prompts = load_prompts(Path("evaluation/prompts.jsonl"))

    assert len(prompts) == 8
    assert len({prompt["id"] for prompt in prompts}) == 8


def test_generation_plan_is_paired_and_resets_generator(tmp_path: Path) -> None:
    plan = build_generation_plan([_prompt()])
    fake_torch = _FakeTorch()
    manifest = tmp_path / "generation_manifest.jsonl"

    records = generate(
        _FakePipe(),
        plan,
        torch_module=fake_torch,
        image_dir=tmp_path / "images",
        manifest_path=manifest,
        run_config={"base_model": "test", "lora_path": "test.safetensors"},
    )

    assert len(records) == 4
    assert [generator.seed for generator in fake_torch.generators] == [1001, 1001, 1002, 1002]
    assert len({id(generator) for generator in fake_torch.generators}) == 4
    assert all(record["generator_reset_per_condition"] for record in records)
    assert len(manifest.read_text().splitlines()) == 4


def test_pair_validation_rejects_missing_condition() -> None:
    record = {
        "prompt_id": "object",
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
    assert all("base" not in row["image_path"] and "lora" not in row["image_path"] for row in rows)
    assert all(field in rows[0] for field in SCORE_FIELDS)
    assert "condition" not in rows[0]
    assert len(key.read_text().splitlines()) == 2
