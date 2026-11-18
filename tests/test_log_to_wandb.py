import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image

pd = pytest.importorskip("pandas")

from analysis.log_to_wandb import (  # noqa: E402
    arm_means,
    generation_config,
    grid_rows,
    is_generated,
    thumbnail,
)

MANIFEST = Path("evaluation/results/study1_8bit/main/generation_manifest.jsonl")


def _png(path: Path, size=(64, 64)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _record(condition, seed=1001, prompt="p0", sha="h", **extra):
    record = {
        "condition": condition,
        "prompt_id": prompt,
        "prompt_group": "g",
        "seed": seed,
        "output_path": f"images/{prompt}/{seed}/{condition}.png",
        "output_sha256": sha,
        "elapsed_seconds": 25.0,
        "num_inference_steps": 20,
        "code_version": {"git_commit": "abcdef1234567890", "python_sources_dirty": False},
        "adapter_active": condition != "base",
        "lora_path": "output/run/run.safetensors",
        "lora_sha256": "lora",
        "lora_scale": 1.0,
        "training_run_name": "study1_pilot_r3",
        "quantization": "8bit",
    }
    record.update(extra)
    return record


def _grid(condition, seed=1001, prompt="p0", sha="h", **metrics):
    return {
        "grid_id": f"{prompt}/{seed}/{condition}",
        "image_path": (MANIFEST.parent / f"images/{prompt}/{seed}/{condition}.png").as_posix(),
        "prompt_id": prompt,
        "seed": seed,
        "arm": condition,
        "image_sha256": sha,
        "identity_dino": 0.5,
        "valid_grid": True,
        **metrics,
    }


def test_only_generation_trees_count_as_generated(tmp_path):
    assert is_generated(Path("evaluation/results/study1_8bit/main/images/a.png"), tmp_path)
    assert is_generated(Path("evaluation/results/holdout/images/a.png"), tmp_path)
    for real in (
        "evaluation/results/controls/images/real/a.png",
        "evaluation/results/controls_crop/images/real/a.png",
        "training/composites_4view_grid/a.png",
        "data/mvi_32/a.jpg",
        "evaluation/results/study1_8bit/../controls/images/real/a.png",
    ):
        assert not is_generated(Path(real), tmp_path), real


def test_thumbnail_checks_location_and_hash(tmp_path):
    generated = Path("evaluation/results/study1b_8bit/main/images/p0/1001/c1.png")
    sha = _png(tmp_path / generated, size=(1024, 1024))
    jpeg = thumbnail(generated, sha, repo=tmp_path)
    image = Image.open(io.BytesIO(jpeg))
    assert image.format == "JPEG" and max(image.size) == 384

    with pytest.raises(ValueError, match="hash"):
        thumbnail(generated, "0" * 64, repo=tmp_path)

    real = Path("evaluation/results/controls/images/real/p0.png")
    real_sha = _png(tmp_path / real)
    with pytest.raises(ValueError, match="not inside a generation output tree"):
        thumbnail(real, real_sha, repo=tmp_path)


def test_generation_config_drops_adapter_details_from_base_records():
    records = [
        _record("base", seed=1001, elapsed_seconds=20.0),
        _record("base", seed=1002, elapsed_seconds=30.0),
        _record("r3", seed=1001),
        _record("r3", seed=1002, elapsed_seconds=None),
    ]
    config = generation_config(records)
    assert config["seeds"] == [1001, 1002]
    assert config["n_images"] == 4 and config["n_prompts"] == 1
    base, r3 = config["arms"]["base"], config["arms"]["r3"]
    assert base["lora_path"] is None and base["training_run_name"] is None
    assert base["lora_scale"] == 0.0 and base["median_seconds_per_image"] == 25.0
    assert r3["wandb_training_run"] == "ki1yqnpk"
    assert r3["n_images"] == 2  # counted even without a recorded time
    assert config["code_commits"] == {"abcdef123456": 4}


def test_generation_config_flags_settings_that_vary():
    config = generation_config([_record("base"), _record("r3", num_inference_steps=28)])
    assert config["num_inference_steps"].startswith("varies")
    assert generation_config([_record("base", quantization=None)])["quantization"] == "none (bf16)"


def test_rows_join_on_path_when_images_share_a_hash():
    # An adapter at scale 0 can render exactly the base image.
    records = [
        _record("base", sha="same", elapsed_seconds=20.0),
        _record("s0", sha="same", elapsed_seconds=31.0),
    ]
    grids = pd.DataFrame([_grid("base", sha="same"), _grid("s0", sha="same")])
    rows = grid_rows(grids, records, MANIFEST)
    assert [(r["arm"], r["seconds"]) for r in rows] == [("base", 20.0), ("s0", 31.0)]
    assert rows[0]["valid_grid"] == 1.0


def test_rows_refuse_images_missing_from_or_differing_from_the_manifest():
    records = [_record("base", sha="a")]
    with pytest.raises(ValueError, match="not in the generation manifest"):
        grid_rows(pd.DataFrame([_grid("r3", sha="b")]), records, MANIFEST)
    with pytest.raises(ValueError, match="hashes differ"):
        grid_rows(pd.DataFrame([_grid("base", sha="b")]), records, MANIFEST)


def test_rows_without_manifest_and_missing_metrics():
    grids = pd.DataFrame([_grid("real", identity_dino=float("nan"))])
    (row,) = grid_rows(grids, None)
    assert row["seconds"] is None and row["identity_dino"] is None


def test_arm_means_treat_booleans_as_rates():
    grids = pd.DataFrame(
        [
            _grid("base", seed=1001, valid_grid=True, identity_dino=0.4),
            _grid("base", seed=1002, valid_grid=False, identity_dino=0.6),
        ]
    )
    means = arm_means(grids)
    assert means["valid_grid"] == {"base": 0.5}
    assert means["identity_dino"]["base"] == pytest.approx(0.5)
