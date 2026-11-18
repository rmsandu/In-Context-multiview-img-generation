import importlib.util

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
# The primary analysis uses scipy (Wilcoxon), which only .venv-eval has.
needs_scipy = pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None, reason="needs scipy (.venv-eval)"
)

from analysis.study1_report import REPORTED  # noqa: E402
from analysis.study1b_effect import tile_distances  # noqa: E402
from analysis.study1b_report import (  # noqa: E402
    analyse,
    base_identical,
    median_ratio,
    reading,
)

PROMPTS = [f"p{i:02d}" for i in range(12)]
SEEDS = [1001, 1002]


def _grids(arms: dict[str, float], *, base_hash: str = "h") -> pd.DataFrame:
    """One row per prompt x seed x arm; every metric = arm offset + prompt noise."""
    rng = np.random.default_rng(0)
    noise = {p: rng.normal(0, 0.05) for p in PROMPTS}
    rows = []
    for arm, offset in {"base": 0.0, **arms}.items():
        for p in PROMPTS:
            for s in SEEDS:
                row = {"arm": arm, "prompt_id": p, "seed": s, "grid_id": f"{p}/{s}/{arm}"}
                row |= {m: 0.6 + offset + noise[p] for m in REPORTED}
                row |= {"valid_grid": 1.0, "max_letterbox_fraction": 0.0}
                row["image_sha256"] = f"{base_hash}-{p}-{s}" if arm == "base" else f"{arm}-{p}-{s}"
                rows.append(row)
    return pd.DataFrame(rows)


def _distances(scale: dict[str, float]) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for arm, value in scale.items():
        family = "crop" if arm.startswith("c") else "letterbox"
        for p in PROMPTS:
            for s in SEEDS:
                for tile in range(4):
                    d = value * rng.uniform(0.8, 1.2)
                    rows.append(
                        {
                            "family": family,
                            "arm": arm,
                            "prompt_id": p,
                            "seed": s,
                            "tile": tile,
                            "dreamsim_to_base": d,
                        }
                    )
    return pd.DataFrame(rows)


def _bars(value: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"family": f, "arm": a, "prompt_id": "p00", "seed": 1001, "soft_bar_fraction": value}
            for f, arms in (("crop", ("c1", "c2", "c3")), ("letterbox", ("r1", "r2", "r3")))
            for a in arms
        ]
    )


def test_tile_distance_is_one_minus_cosine():
    a = np.eye(4)
    assert tile_distances(a, a).tolist() == [0.0] * 4
    assert tile_distances(a, -a).tolist() == [2.0] * 4
    with pytest.raises(ValueError):
        tile_distances(a, a[:3])


def test_median_ratio_recovers_a_doubling():
    d = _distances({"c1": 0.08, "c2": 0.08, "c3": 0.08, "r1": 0.04, "r2": 0.04, "r3": 0.04})
    s2 = median_ratio(d, ("c1", "c2", "c3"), ("r1", "r2", "r3"), n_boot=500)
    assert s2["ratio"] == pytest.approx(2.0, rel=0.05)
    assert 1.8 < s2["ci_low"] < s2["ratio"] < s2["ci_high"] < 2.2


def test_median_ratio_needs_every_prompt():
    d = _distances({"c1": 0.1, "r1": 0.1})
    d = d[~((d["arm"] == "c1") & (d["prompt_id"] == "p03"))]
    with pytest.raises(ValueError, match="No distances"):
        median_ratio(d, ("c1",), ("r1",), n_boot=10)


def test_base_identical_detects_a_changed_base():
    letterbox = _grids({"r1": 0.0})
    assert base_identical(_grids({"c1": 0.0}), letterbox)["identical"]
    changed = _grids({"c1": 0.0}, base_hash="other")
    assert not base_identical(changed, letterbox)["identical"]


def _gates(a1: bool) -> dict:
    return {"gates": {"A1_identity": {"pass": a1}}}


@pytest.mark.parametrize(
    ("a1", "ci", "bug", "start"),
    [
        (True, (0.9, 1.1), False, "removing the bars was enough"),
        (False, (0.9, 1.1), False, "the bars were not what kept"),
        (False, (1.2, 1.6), False, "cropping makes the adapter stronger"),
        (False, (0.5, 0.8), False, "not anticipated"),
        (True, (1.2, 1.6), True, "a crop arm draws bars"),
    ],
)
def test_reading_follows_the_protocol(a1, ci, bug, start):
    s2 = {"ci_low": ci[0], "ci_high": ci[1]}
    assert reading(_gates(a1), s2, {"bug": bug}).startswith(start)


@needs_scipy
def test_analyse_end_to_end():
    crop = _grids({"c1": 0.05, "c2": 0.06, "c3": 0.04})
    letterbox = _grids({"r1": 0.01, "r2": 0.0, "r3": 0.01})
    d = _distances({"c1": 0.09, "c2": 0.09, "c3": 0.09, "r1": 0.03, "r2": 0.03, "r3": 0.03})
    out = analyse(crop, letterbox, d, _bars())

    assert out["decision"]["gates"]["A1_identity"]["pass"]
    assert out["base_identical"]["identical"]
    s1 = out["S1_crop_minus_letterbox"]["identity_dino"]
    assert s1["diff"] == pytest.approx(0.05 - 0.02 / 3)
    assert out["S3"]["identity"]["identity_dino"]["diff"] == pytest.approx(0.03)
    assert out["S2_adapter_strength"]["ratio"] == pytest.approx(3.0, rel=0.1)
    assert not out["S4_bars"]["bug"]
    assert out["reading"].startswith("removing the bars was enough")


@needs_scipy
def test_analyse_flags_bars_in_a_crop_arm():
    crop = _grids({"c1": 0.0, "c2": 0.0, "c3": 0.0})
    letterbox = _grids({"r1": 0.0, "r2": 0.0, "r3": 0.0})
    d = _distances({a: 0.03 for a in ("c1", "c2", "c3", "r1", "r2", "r3")})
    out = analyse(crop, letterbox, d, _bars(0.2))
    assert out["S4_bars"]["bug"]
    assert out["reading"].startswith("a crop arm draws bars")
