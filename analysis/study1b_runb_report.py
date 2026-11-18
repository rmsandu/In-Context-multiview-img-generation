"""Study 1b run B analysis exactly as fixed in docs/study1b_protocol.md (run B).

Reads the metric tables of run B (``results/metrics/runb_main``, ``runb_sweep``),
run A (``results/metrics/study1b_main``) and Study 1 (``main``, ``sweep``, for the
base gate), run B's per-tile tables from ``analysis.study1b_runb_effect`` and run
A's from ``analysis.study1b_effect``. Writes ``results/study1b/runb/summary.json``,
``report.md`` and ``figures/runb_vs_runa.png``. Nothing here runs a model.

    .venv-eval/bin/python -m analysis.study1b_runb_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.study1_report import (
    INK,
    INK_2,
    N_BOOT,
    SEED,
    SURFACE,
    _axes,
    _style,
    cell_matrix,
    code_versions,
    evaluate_gates,
    fmt,
    load_grids,
    primary_analysis,
)
from analysis.study1b_report import (
    BAR_LIMIT,
    CROP,
    CROP_COLOR,
    crop_vs_letterbox,
    median_ratio,
    pooled_matrix,
)
from analysis.study1b_runb_effect import RUNB, RUNB_2000
from evaluation.metrics.stats import paired_cluster_bootstrap

# Categorical slot 3 of the validated palette; run A keeps slot 1 (blue) as in its
# own figure. Slot 3 is below 3:1 contrast on the light surface, so bars carry
# direct value labels.
RUNB_COLOR = "#1baf7a"


# ---------------------------------------------------------------------- analysis


def same_base(new: pd.DataFrame, study1: pd.DataFrame) -> dict:
    """Base gate: every base image of ``new`` equals Study 1's, byte for byte."""

    def hashes(grids: pd.DataFrame) -> pd.Series:
        return grids[grids["arm"] == "base"].set_index(["prompt_id", "seed"])["image_sha256"]

    a, b = hashes(new), hashes(study1)
    shared = a.index.intersection(b.index)
    same = int((a.loc[shared] == b.loc[shared]).sum())
    return {
        "identical": bool(len(a) == len(b) == len(shared) == same),
        "identical_images": same,
        "new_run_base_images": int(len(a)),
        "study1_base_images": int(len(b)),
    }


def identity_vs_base(grids: pd.DataFrame, arms) -> dict:
    """Delta identity_dino against the run's own base, per adapter and pooled."""
    prompts, seeds = _axes(grids[grids["arm"] == "base"])
    base = cell_matrix(grids, "base", "identity_dino", prompts, seeds)

    def boot(matrix: np.ndarray) -> dict:
        return paired_cluster_bootstrap(base, matrix, n_boot=N_BOOT, seed=SEED).as_dict()

    out = {arm: boot(cell_matrix(grids, arm, "identity_dino", prompts, seeds)) for arm in arms}
    out["pooled"] = boot(pooled_matrix(grids, arms, "identity_dino", prompts, seeds))
    return out


def length_effect(sweep: pd.DataFrame) -> dict:
    """T3 identity: pooled step 4,000 - pooled step 2,000, and each against base."""
    prompts, seeds = _axes(sweep[sweep["arm"] == "base"])
    base = cell_matrix(sweep, "base", "identity_dino", prompts, seeds)
    early = pooled_matrix(sweep, RUNB_2000, "identity_dino", prompts, seeds)
    late = pooled_matrix(sweep, RUNB, "identity_dino", prompts, seeds)

    def boot(a: np.ndarray, b: np.ndarray) -> dict:
        return paired_cluster_bootstrap(a, b, n_boot=N_BOOT, seed=SEED).as_dict()

    return {
        "step4000_minus_step2000": boot(early, late),
        "step2000_minus_base": boot(base, early),
        "step4000_minus_base": boot(base, late),
    }


def adapter_medians(distances: pd.DataFrame) -> dict:
    medians = distances.groupby(["family", "arm"])["dreamsim_to_base"].median()
    return {f"{family} {arm}": float(v) for (family, arm), v in medians.items()}


def bar_summary(runb: pd.DataFrame, sweep: pd.DataFrame, bars: pd.DataFrame) -> dict:
    """T4: mean bar fraction per arm by the registered and the looser detector."""
    registered = {}
    for family, grids in (("runb", runb), ("runb_sweep", sweep)):
        for arm, value in grids.groupby("arm")["max_letterbox_fraction"].mean().items():
            registered[f"{family} {arm}"] = float(value)
    loose = {
        f"{family} {arm}": float(v)
        for (family, arm), v in bars.groupby(["family", "arm"])["soft_bar_fraction"].mean().items()
    }
    adapter_keys = [f"runb {a}" for a in RUNB] + [f"runb_sweep {a}" for a in RUNB_2000 + RUNB]
    missing = [k for k in adapter_keys if k not in registered or k not in loose]
    if missing:
        raise ValueError(f"No bar measurements for {missing}")
    worst = max(max(registered[k], loose[k]) for k in adapter_keys)
    return {
        "max_letterbox_fraction": registered,
        "soft_bar_fraction": loose,
        "worst_runb_arm": worst,
        "bug": worst > BAR_LIMIT,
    }


def reading(gates: dict, t2: dict, bars: dict) -> str:
    """The pre-written run B readings in docs/study1b_protocol.md, applied mechanically."""
    if bars["bug"]:
        return "a run B arm draws bars: a bug in the data; stop and investigate"
    if gates["gates"]["A1_identity"]["pass"]:
        return (
            "more data and longer training give a measurable identity gain with this recipe; "
            "H1 decides the full claim"
        )
    if t2["ci_low"] > 1.0:
        return (
            "run B's adapters move images further than run A's, but that still does not show "
            "as an identity gain with 42 prompts"
        )
    return (
        "three and a half times the data and eight times the steps did not make the "
        "adapter stronger"
    )


def analyse(
    runb: pd.DataFrame,
    runa: pd.DataFrame,
    sweep: pd.DataFrame,
    study1_main: pd.DataFrame,
    study1_sweep: pd.DataFrame,
    distances: pd.DataFrame,
    bars: pd.DataFrame,
    human=None,
) -> dict:
    for name, grids, arms in (
        ("run B main", runb, RUNB),
        ("run A main", runa, CROP),
        ("run B sweep", sweep, RUNB_2000 + RUNB),
    ):
        missing = sorted({"base", *arms} - set(grids["arm"]))
        if missing:  # a missing arm would otherwise become NaN cells
            raise ValueError(f"{name} metrics lack arms {missing}")
    primary = primary_analysis(runb, replicates=RUNB)
    decision = evaluate_gates(primary, human)
    gate = {"main": same_base(runb, study1_main), "sweep": same_base(sweep, study1_sweep)}
    gate["identical"] = gate["main"]["identical"] and gate["sweep"]["identical"]
    # Arm names repeat across runs (b1 is in the main and the sweep run): filter by family.
    main_distances = distances[distances["family"].isin(["runb", "crop"])]
    sweep_distances = distances[distances["family"] == "runb_sweep"]
    t2 = median_ratio(main_distances, RUNB, CROP)
    bar_result = bar_summary(runb, sweep, bars)
    return {
        "primary": primary,
        "decision": decision,
        "base_identical": gate,
        "identity_vs_base": {
            "runa": identity_vs_base(runa, CROP),
            "runb": identity_vs_base(runb, RUNB),
        },
        "T1_runb_minus_runa": (
            crop_vs_letterbox(runb, runa, RUNB, CROP) if gate["identical"] else None
        ),
        "T2_adapter_strength": t2 | {"per_adapter": adapter_medians(distances)},
        "T3_length": {
            "identity": length_effect(sweep),
            "strength": median_ratio(sweep_distances, RUNB, RUNB_2000),
        },
        "T4_bars": bar_result,
        "reading": reading(decision, t2, bar_result),
    }


# ----------------------------------------------------------------------- figure


def figure(summary: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    identity = summary["identity_vs_base"]
    rows = [("run A", arm, identity["runa"][arm]) for arm in (*CROP, "pooled")] + [
        ("run B", arm, identity["runb"][arm]) for arm in (*RUNB, "pooled")
    ]
    for y, (run, arm, est) in enumerate(rows):
        color = RUNB_COLOR if run == "run B" else CROP_COLOR
        size = 9 if arm == "pooled" else 6
        left.plot([est["ci_low"], est["ci_high"]], [y, y], color=color, linewidth=2)
        left.plot(est["diff"], y, "o", color=color, markersize=size, markeredgecolor=SURFACE)
    left.axvline(0, color=INK_2, linewidth=0.8)
    left.set_yticks(range(len(rows)), [f"{run} {arm}" for run, arm, _ in rows])
    left.invert_yaxis()
    _style(left, "Identity vs base (95% CI)", "delta identity_dino", "")

    per_adapter = summary["T2_adapter_strength"]["per_adapter"]
    bars = [("run A", a, per_adapter[f"crop {a}"]) for a in CROP] + [
        ("run B", a, per_adapter[f"runb {a}"]) for a in RUNB
    ]
    for x, (run, arm, value) in enumerate(bars):
        color = RUNB_COLOR if run == "run B" else CROP_COLOR
        right.bar(x, value, width=0.6, color=color, edgecolor=SURFACE)
        right.annotate(
            f"{value:.3f}",
            (x, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
            color=INK,
        )
    right.set_xticks(range(len(bars)), [arm for _, arm, _ in bars])
    t2 = summary["T2_adapter_strength"]
    _style(
        right, "How far each adapter moves a tile", "adapter", "median DreamSim distance to base"
    )
    right.text(
        0.02,
        0.97,
        f"run B / run A: {t2['ratio']:.2f} [{t2['ci_low']:.2f}, {t2['ci_high']:.2f}]",
        transform=right.transAxes,
        va="top",
        fontsize=9,
        color=INK_2,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2},
    )
    right.margins(y=0.15)
    handles = [
        plt.Line2D([], [], color=CROP_COLOR, marker="o", linewidth=2, label="run A (500 steps)"),
        plt.Line2D([], [], color=RUNB_COLOR, marker="o", linewidth=2, label="run B (4,000 steps)"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- report


def write_report(summary: dict, out: Path) -> None:
    p = summary["primary"]
    lines = [
        "# Study 1b run B results",
        "",
        "Run A's 381 crops plus 936 new ones (1,317), trained for 4,000 steps instead of 500, "
        "everything else as in run A ([protocol](../../../docs/study1b_protocol.md)).",
        "",
        f"Verdict (pre-registered rule): **{summary['decision']['verdict']}**.",
        "",
        f"Reading (written in advance): **{summary['reading']}**.",
        "",
        "| Gate | Rule | Pass |",
        "| --- | --- | :-: |",
    ]
    for name, gate in summary["decision"]["gates"].items():
        mark = {True: "yes", False: "no", None: "pending"}[gate["pass"]]
        lines.append(f"| {name} | {gate['rule']} | {mark} |")
    lines += [
        "",
        f"Run B adapter minus base, {p['n_prompts']} prompts x {len(p['seeds'])} seeds, "
        "95% cluster-bootstrap CI:",
        "",
        "| Metric | Base | Pooled run B | Delta (pooled) | " + " | ".join(p["replicates"]) + " |",
        "| --- | --: | --: | --: | " + " | ".join("--:" for _ in p["replicates"]) + " |",
    ]
    for metric, entry in p["metrics"].items():
        pooled = entry["pooled"]
        reps = " | ".join(fmt(entry["replicates"][r]) for r in p["replicates"])
        lines.append(
            f"| {metric} | {pooled['mean_a']:.3f} | {pooled['mean_b']:.3f} "
            f"| {fmt(pooled)} | {reps} |"
        )
    holm = json.dumps({k: round(v, 4) for k, v in p["holm_p"].items()})
    lines += ["", f"Holm-adjusted Wilcoxon p: {holm}", "", "## Run B vs run A", ""]
    gate = summary["base_identical"]
    for name in ("main", "sweep"):
        g = gate[name]
        lines.append(
            f"* {name} base images identical to Study 1: {g['identical_images']}/"
            f"{g['study1_base_images']} (**{'yes' if g['identical'] else 'no'}**)"
        )
    t2, t3 = summary["T2_adapter_strength"], summary["T3_length"]
    lines += ["", "| Analysis | Estimate, 95% CI |", "| --- | --: |"]
    if summary["T1_runb_minus_runa"] is None:
        lines.append("| T1 | not reported: run B does not share Study 1's base (protocol) |")
    else:
        lines += [
            f"| T1 delta {m}, pooled run B - pooled run A | {fmt(e)} |"
            for m, e in summary["T1_runb_minus_runa"].items()
        ]
    lines += [
        f"| T2 median distance to base: run B / run A | {t2['ratio']:.2f} "
        f"[{t2['ci_low']:.2f}, {t2['ci_high']:.2f}] "
        f"({t2['median_numerator']:.3f} / {t2['median_denominator']:.3f}) |",
        *(
            f"| T3 delta identity_dino, {name.replace('_', ' ')} (sweep prompts) | {fmt(e)} |"
            for name, e in t3["identity"].items()
        ),
        f"| T3 median distance to base: step 4,000 / step 2,000 (sweep prompts) | "
        f"{t3['strength']['ratio']:.2f} "
        f"[{t3['strength']['ci_low']:.2f}, {t3['strength']['ci_high']:.2f}] |",
        "",
        "Median distance to base per adapter (T2):",
        "",
        "| Adapter | Median |",
        "| --- | --: |",
        *(f"| {k} | {v:.3f} |" for k, v in t2["per_adapter"].items()),
        "",
        "![Run B vs run A](figures/runb_vs_runa.png)",
    ]
    bars = summary["T4_bars"]
    lines += [
        "",
        "## T4: bars in generated grids",
        "",
        "| Arm | max_letterbox_fraction | soft_bar_fraction |",
        "| --- | --: | --: |",
    ]
    for key, value in bars["max_letterbox_fraction"].items():
        loose = bars["soft_bar_fraction"].get(key)
        lines.append(f"| {key} | {value:.3f} | {'' if loose is None else f'{loose:.3f}'} |")
    lines += [
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(summary["code_versions"], indent=2),
        "```",
        "",
    ]
    out.write_text("\n".join(lines))


# ------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    metrics = Path("results/metrics")
    parser.add_argument("--runb-metrics", type=Path, default=metrics / "runb_main")
    parser.add_argument("--sweep-metrics", type=Path, default=metrics / "runb_sweep")
    parser.add_argument("--runa-metrics", type=Path, default=metrics / "study1b_main")
    parser.add_argument("--study1-metrics", type=Path, default=metrics / "main")
    parser.add_argument("--study1-sweep-metrics", type=Path, default=metrics / "sweep")
    parser.add_argument("--tables", type=Path, default=Path("results/study1b/runb"))
    parser.add_argument("--runa-tables", type=Path, default=Path("results/study1b"))
    parser.add_argument("--humans", type=Path, default=Path("results/study1b/runb/humans.json"))
    parser.add_argument(
        "--generation", type=Path, default=Path("evaluation/results/study1b_8bit")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/study1b/runb"))
    args = parser.parse_args(argv)

    runb_distances = pd.read_csv(args.tables / "adapter_distance.csv")
    runa_distances = pd.read_csv(args.runa_tables / "adapter_distance.csv")
    distances = pd.concat(
        [runb_distances, runa_distances[runa_distances["family"] == "crop"]], ignore_index=True
    )
    human = json.loads(args.humans.read_text()) if args.humans.is_file() else None
    summary = analyse(
        load_grids(args.runb_metrics),
        load_grids(args.runa_metrics),
        load_grids(args.sweep_metrics),
        load_grids(args.study1_metrics),
        load_grids(args.study1_sweep_metrics),
        distances,
        pd.read_csv(args.tables / "bars.csv"),
        human,
    )
    summary["code_versions"] = {
        run: code_versions(args.generation / f"runb_{run}" / "generation_manifest.jsonl")
        for run in ("main", "sweep")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    figure(summary, args.output_dir / "figures" / "runb_vs_runa.png")
    write_report(summary, args.output_dir / "report.md")
    print(f"verdict: {summary['decision']['verdict']}\nreading: {summary['reading']}")


if __name__ == "__main__":
    main()
