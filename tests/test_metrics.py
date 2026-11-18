import json

import numpy as np
import pytest
from PIL import Image

from evaluation import controls
from evaluation.metrics import stats
from evaluation.metrics.run_metrics import Thresholds, derive_grid_metrics, load_grid_list
from evaluation.metrics.views import (
    TILE_PAIRS,
    count_distinct_views,
    expected_relative_azimuth,
    join_tiles,
    parse_tile_views,
    split_grid,
    wrap_degrees,
)

HOLDOUT_STYLE = (
    "[FOUR-VIEWS] Four views of a red kettle; [TOP-LEFT] Front, spout; "
    "[TOP-RIGHT] Right side, handle; [BOTTOM-LEFT] Back, lid; [BOTTOM-RIGHT] Left side, base."
)


def colour_grid(colours):
    return join_tiles([Image.new("RGB", (8, 8), c) for c in colours])


def test_split_and_join_round_trip():
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (9, 9, 9)]
    tiles = split_grid(colour_grid(colours))
    assert [t.getpixel((0, 0)) for t in tiles] == colours
    with pytest.raises(ValueError):
        split_grid(Image.new("RGB", (7, 8)))


def test_parse_tile_views_reads_label_and_elevation():
    views = parse_tile_views(
        "[TOP-LEFT] Left side, high angle; lid; [TOP-RIGHT] Front-right three-quarter, eye-level; "
        "[BOTTOM-LEFT] Top, overhead; [BOTTOM-RIGHT] Rear, low angle"
    )
    assert views["TOP-LEFT"].azimuth == 270 and views["TOP-LEFT"].elevation == "high angle"
    assert views["TOP-RIGHT"].azimuth == 45
    assert views["BOTTOM-LEFT"].label == "top" and views["BOTTOM-LEFT"].azimuth is None
    assert views["BOTTOM-RIGHT"].azimuth == 180


def test_missing_positions_are_unscored_not_guessed():
    views = parse_tile_views("[TOP-LEFT] Front")
    assert views["BOTTOM-RIGHT"].azimuth is None
    assert expected_relative_azimuth(views, 0, 3) is None


def test_expected_relative_azimuth_wraps():
    views = parse_tile_views(HOLDOUT_STYLE)
    assert expected_relative_azimuth(views, 0, 1) == 90  # front -> right side
    assert expected_relative_azimuth(views, 0, 3) == -90  # front -> left side
    assert abs(expected_relative_azimuth(views, 1, 3)) == 180
    assert wrap_degrees(-180) == 180 and wrap_degrees(370) == 10


def test_distinct_views_is_transitive():
    none_same = {pair: False for pair in TILE_PAIRS}
    assert count_distinct_views(none_same) == 4
    chain = dict(none_same) | {(0, 1): True, (1, 2): True}
    assert count_distinct_views(chain) == 2
    assert count_distinct_views({pair: True for pair in TILE_PAIRS}) == 1


def fake_raw(rel_by_pair, dino=0.8):
    return {
        "tiles": [
            {
                "foreground": 0.3,
                "abs_azimuth": 0,
                "abs_elevation": 10,
                "abs_azimuth_spread": 2,
                "symmetry_alpha": 1,
            }
            for _ in range(4)
        ],
        "pairs": [
            {
                "i": i,
                "j": j,
                "dino_masked": dino,
                "dino_unmasked": dino + 0.05,
                "dreamsim_distance": 0.3,
                "phash_distance": 20,
                "rel_azimuth": rel_by_pair[(i, j)],
                "rel_elevation": 0.0,
                "rel_inconsistency": 4.0,
            }
            for i, j in TILE_PAIRS
        ],
    }


def test_derive_metrics_perfect_canonical_grid():
    camera = [0, 90, 180, 270]
    rel = {(i, j): wrap_degrees(camera[j] - camera[i]) for i, j in TILE_PAIRS}
    grid, pairs = derive_grid_metrics(fake_raw(rel), HOLDOUT_STYLE, Thresholds(azimuth_sign=1))
    assert grid["distinct_views"] == 4 and not grid["any_same_view"]
    assert grid["pose_accuracy"] == 1.0 and grid["pose_pairs"] == 6
    assert grid["identity_dino"] == pytest.approx(0.8)
    assert grid["identity_dreamsim"] == pytest.approx(0.7)
    assert grid["valid_grid"]


def test_derive_metrics_sign_flip_breaks_left_right_only():
    camera = [0, 90, 180, 270]
    rel = {(i, j): -wrap_degrees(camera[j] - camera[i]) for i, j in TILE_PAIRS}
    grid, _ = derive_grid_metrics(fake_raw(rel), HOLDOUT_STYLE, Thresholds(azimuth_sign=1))
    # front/back and right/left pairs are 180 apart, so they survive a sign flip.
    assert grid["pose_accuracy"] == pytest.approx(2 / 6)
    flipped, _ = derive_grid_metrics(fake_raw(rel), HOLDOUT_STYLE, Thresholds(azimuth_sign=-1))
    assert flipped["pose_accuracy"] == 1.0
    # The frozen default is the sign calibrated on real holdout photos.
    assert Thresholds().azimuth_sign == -1


def test_derive_metrics_duplicated_view():
    rel = {pair: 5.0 for pair in TILE_PAIRS}
    grid, pairs = derive_grid_metrics(fake_raw(rel, dino=0.99), HOLDOUT_STYLE)
    assert grid["distinct_views"] == 1 and grid["any_same_view"]
    assert grid["pose_accuracy"] == 0.0
    assert all(p["same_view"] for p in pairs)


def test_load_grid_list_from_generation_manifest(tmp_path):
    rows = [
        {
            "prompt_id": "kettle",
            "seed": 1001,
            "condition": "base",
            "output_path": "images/k/1001/base.png",
            "prompt": HOLDOUT_STYLE,
            "output_sha256": "abc",
        },
        {
            "prompt_id": "kettle",
            "seed": 1001,
            "condition": "lora",
            "lora_scale": 0.5,
            "output_path": "images/k/1001/lora.png",
            "prompt": HOLDOUT_STYLE,
            "lora_sha256": "def",
        },
    ]
    manifest = tmp_path / "generation_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    grids = load_grid_list(manifest)
    assert [g["arm"] for g in grids] == ["base", "lora@0.5"]
    assert grids[0]["image_path"] == str(tmp_path / "images/k/1001/base.png")
    assert grids[0]["lora_scale"] == 0.0 and grids[1]["lora_scale"] == 0.5


def test_bootstrap_recovers_known_shift():
    rng = np.random.default_rng(0)
    prompt_effect = rng.normal(0, 1, size=(40, 1))
    a = prompt_effect + rng.normal(0, 0.1, size=(40, 4))
    b = a + 0.5
    est = stats.paired_cluster_bootstrap(a, b, n_boot=2000)
    assert est.diff == pytest.approx(0.5)
    assert est.ci_low == pytest.approx(0.5) and est.ci_high == pytest.approx(0.5)
    noisy = stats.paired_cluster_bootstrap(a, a + rng.normal(0, 0.2, size=a.shape), n_boot=2000)
    assert noisy.ci_low < 0 < noisy.ci_high


def test_bootstrap_ignores_nan_prompts_and_is_seeded():
    a = np.array([[0.1, 0.2], [np.nan, np.nan], [0.3, 0.4]])
    b = a + 0.1
    one = stats.paired_cluster_bootstrap(a, b, n_boot=500, seed=3)
    two = stats.paired_cluster_bootstrap(a, b, n_boot=500, seed=3)
    assert one == two and one.n_prompts == 2


def test_holm_is_monotone_and_capped():
    adjusted = stats.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.08, "c": 0.5})
    assert stats.holm({"x": 0.9, "y": 0.8})["x"] == 1.0


def test_mde_shrinks_with_more_seeds():
    null = np.random.default_rng(1).normal(0, 0.1, size=42)
    one = stats.minimum_detectable_effect(null, n_prompts=42, seeds_per_condition=1)
    four = stats.minimum_detectable_effect(null, n_prompts=42, seeds_per_condition=4)
    assert four == pytest.approx(one / 2)


def make_split(tmp_path, n=6):
    root = tmp_path / "split"
    (root / "holdout").mkdir(parents=True)
    rows = []
    for k in range(n):
        stem = f"obj{k}_{100 + k}_x"
        colours = [(40 * k, q * 60, 10) for q in range(4)]
        colour_grid(colours).save(root / "holdout" / f"{stem}.png")
        (root / "holdout" / f"{stem}.txt").write_text(HOLDOUT_STYLE)
        rows.append(
            {
                "split": "holdout",
                "stem": stem,
                "image": f"holdout/{stem}.png",
                "caption": f"holdout/{stem}.txt",
                "image_sha256": f"h{k}",
                "source_instance": f"mvi_00/{100 + k}/x",
            }
        )
    rows.append(
        {
            "split": "train",
            "stem": "ignored",
            "image": "x",
            "caption": "y",
            "image_sha256": "t",
            "source_instance": "mvi_00/1/y",
        }
    )
    (root / "split_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_controls_build_variants(tmp_path):
    root = make_split(tmp_path)
    grid_list = controls.build(root, tmp_path / "controls")
    records = [json.loads(line) for line in grid_list.read_text().splitlines()]
    assert len(records) == 6 * len(controls.KINDS)
    by_id = {r["grid_id"]: r for r in records}
    base = tmp_path / "controls"

    def tiles(kind):
        return split_grid(Image.open(base / by_id[f"obj0_100_x/0/{kind}"]["image_path"]))

    real, dup, shuffled, mixed = tiles("real"), tiles("dup4"), tiles("shuffled"), tiles("mixed")
    assert len({t.getpixel((0, 0)) for t in dup}) == 1
    assert [t.getpixel((0, 0)) for t in shuffled] == [t.getpixel((0, 0)) for t in real][::-1]
    sources = by_id["obj0_100_x/0/mixed"]["mixed_sources"]
    assert sources[0] == "obj0_100_x" and len(set(sources)) == 4
    assert mixed[0].getpixel((0, 0)) == real[0].getpixel((0, 0))
    # Rebuilding is deterministic.
    again = controls.build(root, tmp_path / "controls2")
    assert [json.loads(x)["mixed_sources"] for x in again.read_text().splitlines()] == [
        r["mixed_sources"] for r in records
    ]


def test_load_grid_list_from_multi_arm_manifest(tmp_path):
    rows = [
        {
            "prompt_id": "k",
            "seed": 1,
            "condition": "base",
            "lora_label": None,
            "lora_scale": 0.0,
            "output_path": "images/k/1/base.png",
            "prompt": HOLDOUT_STYLE,
            "quantization": "8bit",
        },
        {
            "prompt_id": "k",
            "seed": 1,
            "condition": "r2",
            "lora_label": "r2",
            "lora_scale": 1.0,
            "training_run_name": "study1_pilot_v2",
            "output_path": "images/k/1/r2.png",
            "prompt": HOLDOUT_STYLE,
            "quantization": "8bit",
        },
    ]
    manifest = tmp_path / "generation_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    grids = load_grid_list(manifest)
    assert [g["arm"] for g in grids] == ["base", "r2"]
    assert grids[1]["lora_scale"] == 1.0 and grids[1]["training_run_name"] == "study1_pilot_v2"
    assert grids[0]["quantization"] == "8bit"


def test_letterbox_fraction_counts_edge_bars_only():
    from evaluation.metrics.views import letterbox_fraction

    tile = Image.new("RGB", (100, 100), (0, 0, 0))
    tile.paste(Image.new("RGB", (60, 100), (200, 150, 100)), (20, 0))  # pillarbox 20 px each side
    assert letterbox_fraction(tile) == pytest.approx(0.4)
    dark_object = Image.new("RGB", (100, 100), (200, 200, 200))
    dark_object.paste(Image.new("RGB", (30, 30), (0, 0, 0)), (35, 35))  # dark object, no bars
    assert letterbox_fraction(dark_object) == 0.0
