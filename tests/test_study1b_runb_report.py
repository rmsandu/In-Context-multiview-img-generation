import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")  # the primary analysis (Wilcoxon); only .venv-eval has it

from analysis.study1_report import REPORTED  # noqa: E402
from analysis.study1b_runb_report import analyse, reading, write_report  # noqa: E402

PROMPTS = [f"p{i:02d}" for i in range(12)]
SEEDS = [1001, 1002]
RUNB = {"b1": 0.05, "b2": 0.05, "b3": 0.05}
RUNA = {"c1": 0.01, "c2": 0.01, "c3": 0.01}
SWEEP = {"b1_s2000": 0.02, "b2_s2000": 0.02, "b3_s2000": 0.02, **RUNB}


def _grids(arms: dict[str, float], *, base_hash: str = "h", bars: float = 0.0) -> pd.DataFrame:
    """One row per prompt x seed x arm; every metric = arm offset + prompt noise."""
    rng = np.random.default_rng(0)
    noise = {p: rng.normal(0, 0.05) for p in PROMPTS}
    rows = []
    for arm, offset in {"base": 0.0, **arms}.items():
        for p in PROMPTS:
            for s in SEEDS:
                row = {"arm": arm, "prompt_id": p, "seed": s, "grid_id": f"{p}/{s}/{arm}"}
                row |= {m: 0.6 + offset + noise[p] for m in REPORTED}
                row |= {"valid_grid": 1.0, "max_letterbox_fraction": 0.0 if arm == "base" else bars}
                row["image_sha256"] = f"{base_hash}-{p}-{s}" if arm == "base" else f"{arm}-{p}-{s}"
                rows.append(row)
    return pd.DataFrame(rows)


def _distances(family: str, scale: dict[str, float]) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        [
            {
                "family": family,
                "arm": arm,
                "prompt_id": p,
                "seed": s,
                "tile": tile,
                "dreamsim_to_base": value * rng.uniform(0.8, 1.2),
            }
            for arm, value in scale.items()
            for p in PROMPTS
            for s in SEEDS
            for tile in range(4)
        ]
    )


def _bars(value: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"family": f, "arm": a, "prompt_id": "p00", "seed": 1001, "soft_bar_fraction": value}
            for f, arms in (("runb", RUNB), ("runb_sweep", SWEEP))
            for a in ("base", *arms)
        ]
    )


def _inputs(*, study1_hash="h", bars=0.0, runb=RUNB, runb_distance=0.08):
    distances = pd.concat(
        [
            _distances("runb", {a: runb_distance for a in RUNB}),
            _distances("crop", {a: 0.04 for a in RUNA}),
            # The sweep reuses the names b1-b3; its distances must not leak into T2.
            _distances("runb_sweep", {**{f"b{i}_s2000": 0.02 for i in (1, 2, 3)}, **{
                f"b{i}": 0.06 for i in (1, 2, 3)
            }}),
        ],
        ignore_index=True,
    )
    return dict(
        runb=_grids(runb, bars=bars),
        runa=_grids(RUNA),
        sweep=_grids(SWEEP),
        study1_main=_grids({"r1": 0.0}, base_hash=study1_hash),
        study1_sweep=_grids({}, base_hash=study1_hash),
        distances=distances,
        bars=_bars(bars),
    )


def test_analysis_recovers_planted_effects():
    s = analyse(**_inputs())
    assert s["decision"]["gates"]["A1_identity"]["pass"]
    assert s["base_identical"]["identical"]
    assert s["T1_runb_minus_runa"]["identity_dino"]["diff"] == pytest.approx(0.04)
    assert s["T2_adapter_strength"]["ratio"] == pytest.approx(2.0, rel=0.05)
    assert s["T2_adapter_strength"]["ci_low"] > 1.0
    assert s["T2_adapter_strength"]["per_adapter"]["runb_sweep b1"] == pytest.approx(0.06, rel=0.1)
    t3 = s["T3_length"]
    assert t3["identity"]["step4000_minus_step2000"]["diff"] == pytest.approx(0.03)
    assert t3["identity"]["step2000_minus_base"]["diff"] == pytest.approx(0.02)
    assert t3["strength"]["ratio"] == pytest.approx(3.0, rel=0.05)
    assert s["identity_vs_base"]["runa"]["pooled"]["diff"] == pytest.approx(0.01)
    assert not s["T4_bars"]["bug"]
    assert s["reading"].startswith("more data and longer training give a measurable")


def test_weak_adapter_reading_and_no_gain():
    s = analyse(**_inputs(runb={"b1": 0.0, "b2": 0.0, "b3": 0.0}, runb_distance=0.04))
    assert not s["decision"]["gates"]["A1_identity"]["pass"]
    assert s["T2_adapter_strength"]["ci_low"] <= 1.0
    assert s["reading"].startswith("three and a half times the data")


def test_t1_is_withheld_when_the_base_differs(tmp_path):
    s = analyse(**_inputs(study1_hash="other"))
    assert not s["base_identical"]["identical"]
    assert s["T1_runb_minus_runa"] is None
    s["code_versions"] = {}
    write_report(s, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text()
    assert "T1 | not reported" in text and "0/24" in text
    json.dumps(s, default=float)  # the summary must serialise


def test_bars_in_a_run_b_arm_stop_the_reading():
    s = analyse(**_inputs(bars=0.2))
    assert s["T4_bars"]["bug"]
    assert s["reading"].startswith("a run B arm draws bars")


def test_a_missing_arm_is_an_error_not_nan():
    inputs = _inputs()
    inputs["sweep"] = inputs["sweep"][inputs["sweep"]["arm"] != "b2_s2000"]
    with pytest.raises(ValueError, match="b2_s2000"):
        analyse(**inputs)


@pytest.mark.parametrize(
    "a1, ci_low, bug, start",
    [
        (False, 1.2, False, "run B's adapters move images further"),
        (False, 0.9, False, "three and a half times"),
        (True, 0.9, False, "more data and longer training"),
        (True, 1.2, True, "a run B arm draws bars"),
    ],
)
def test_reading_follows_the_protocol(a1, ci_low, bug, start):
    gates = {"gates": {"A1_identity": {"pass": a1}}}
    assert reading(gates, {"ci_low": ci_low}, {"bug": bug}).startswith(start)


def test_figure_renders(tmp_path):
    pytest.importorskip("matplotlib")
    from analysis.study1b_runb_report import figure

    s = analyse(**_inputs())
    figure(s, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 10_000
