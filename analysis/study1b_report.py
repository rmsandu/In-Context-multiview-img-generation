"""Study 1b run A analysis exactly as fixed in docs/study1b_protocol.md.

Reads the metric tables of both runs (``results/metrics/main`` for the Study 1
letterbox adapters, ``results/metrics/study1b_main`` for the crop adapters) and
the per-tile tables from ``analysis.study1b_effect``. Writes
``results/study1b/summary.json``, ``results/study1b/report.md`` and
``results/study1b/figures/crop_vs_letterbox.png``. Nothing here runs a model.

    .venv-eval/bin/python -m analysis.study1b_report
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
from evaluation.metrics.stats import paired_cluster_bootstrap

LETTERBOX = ("r1", "r2", "r3")
CROP = ("c1", "c2", "c3")
S1_METRICS = ("identity_dino", "identity_dreamsim")
BAR_LIMIT = 0.05  # S4: a crop arm averaging more bars than this means a build bug
# Categorical slots 1 and 2 of the validated palette (light surface, all checks pass).
CROP_COLOR, LETTERBOX_COLOR = "#2a78d6", "#eb6834"


# ---------------------------------------------------------------------- analysis


def base_identical(crop: pd.DataFrame, letterbox: pd.DataFrame) -> dict:
    """The gate for S1-S3: both runs must share byte-identical base images."""

    def hashes(grids: pd.DataFrame) -> pd.Series:
        return grids[grids["arm"] == "base"].set_index(["prompt_id", "seed"])["image_sha256"]

    a, b = hashes(crop), hashes(letterbox)
    shared = a.index.intersection(b.index)
    same = int((a.loc[shared] == b.loc[shared]).sum())
    return {
        "identical": bool(len(a) == len(b) == len(shared) == same),
        "identical_images": same,
        "crop_run_base_images": int(len(a)),
        "letterbox_run_base_images": int(len(b)),
    }


def pooled_matrix(grids: pd.DataFrame, arms, metric: str, prompts, seeds) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmean(
            np.stack([cell_matrix(grids, arm, metric, prompts, seeds) for arm in arms]), axis=0
        )


def crop_vs_letterbox(
    crop: pd.DataFrame, letterbox: pd.DataFrame, crop_arms, letterbox_arms
) -> dict:
    """S1 (and S3 with one arm each): crop minus letterbox, paired by prompt and seed."""
    prompts, seeds = _axes(crop[crop["arm"] == "base"])
    out = {}
    for metric in S1_METRICS:
        a = pooled_matrix(letterbox, letterbox_arms, metric, prompts, seeds)
        b = pooled_matrix(crop, crop_arms, metric, prompts, seeds)
        out[metric] = paired_cluster_bootstrap(a, b, n_boot=N_BOOT, seed=SEED).as_dict()
    return out


def median_ratio(
    distances: pd.DataFrame, numerator, denominator, *, n_boot: int = N_BOOT, seed: int = SEED
) -> dict:
    """S2: median per-tile distance to base, numerator arms / denominator arms.

    The CI resamples prompts with replacement (the protocol's unit) and recomputes
    both medians over all tiles of the resampled prompts.
    """
    prompts = sorted(distances["prompt_id"].unique())

    def per_prompt(arms) -> list[np.ndarray]:
        rows = distances[distances["arm"].isin(arms)]
        grouped = rows.groupby("prompt_id")["dreamsim_to_base"]
        values = {p: g.to_numpy() for p, g in grouped}
        missing = set(prompts) - set(values)
        if missing:
            raise ValueError(f"No distances for {sorted(missing)[:3]} in arms {list(arms)}")
        return [values[p] for p in prompts]

    num, den = per_prompt(numerator), per_prompt(denominator)
    point_num = float(np.median(np.concatenate(num)))
    point_den = float(np.median(np.concatenate(den)))
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    for i, idx in enumerate(rng.integers(0, len(prompts), size=(n_boot, len(prompts)))):
        ratios[i] = np.median(np.concatenate([num[j] for j in idx])) / np.median(
            np.concatenate([den[j] for j in idx])
        )
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {
        "median_numerator": point_num,
        "median_denominator": point_den,
        "ratio": point_num / point_den,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_prompts": len(prompts),
        "n_boot": n_boot,
    }


def bar_summary(crop: pd.DataFrame, letterbox: pd.DataFrame, bars: pd.DataFrame) -> dict:
    """S4: mean bar fraction per arm, by the registered and the looser detector."""
    registered = pd.concat(
        [
            crop.groupby("arm")["max_letterbox_fraction"].mean().rename(lambda a: f"crop {a}"),
            letterbox.groupby("arm")["max_letterbox_fraction"]
            .mean()
            .rename(lambda a: f"letterbox {a}"),
        ]
    )
    loose = bars.groupby(["family", "arm"])["soft_bar_fraction"].mean()
    worst_crop = max(
        float(crop[crop["arm"].isin(CROP)].groupby("arm")["max_letterbox_fraction"].mean().max()),
        float(loose.loc["crop"].reindex(CROP).max()),
    )
    return {
        "max_letterbox_fraction": {k: float(v) for k, v in registered.items()},
        "soft_bar_fraction": {f"{f} {a}": float(v) for (f, a), v in loose.items()},
        "worst_crop_arm": worst_crop,
        "bug": worst_crop > BAR_LIMIT,
    }


def reading(gates: dict, s2: dict, bars: dict) -> str:
    """The pre-written readings in docs/study1b_protocol.md, applied mechanically."""
    if bars["bug"]:
        return "a crop arm draws bars: a bug in the crop build; stop and investigate"
    if gates["gates"]["A1_identity"]["pass"]:
        return (
            "removing the bars was enough for a measurable identity gain with this recipe; "
            "H1 decides the full claim"
        )
    if s2["ci_low"] <= 1.0 <= s2["ci_high"]:
        return (
            "the bars were not what kept the adapter weak; the recipe (500 steps at lr 1e-4) "
            "is, which run B tests"
        )
    if s2["ci_low"] > 1.0:
        return (
            "cropping makes the adapter stronger, but not enough to show on identity "
            "with 42 prompts"
        )
    return "not anticipated by the protocol: cropping made the adapter weaker (S2 below 1)"


# ----------------------------------------------------------------------- figure


def figure(summary: dict, distances: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    identity = summary["identity_vs_base"]
    rows = [("letterbox", arm, identity["letterbox"][arm]) for arm in (*LETTERBOX, "pooled")] + [
        ("crop", arm, identity["crop"][arm]) for arm in (*CROP, "pooled")
    ]
    for y, (family, arm, est) in enumerate(rows):
        color = CROP_COLOR if family == "crop" else LETTERBOX_COLOR
        size = 9 if arm == "pooled" else 6
        left.plot([est["ci_low"], est["ci_high"]], [y, y], color=color, linewidth=2)
        left.plot(est["diff"], y, "o", color=color, markersize=size, markeredgecolor=SURFACE)
    left.axvline(0, color=INK_2, linewidth=0.8)
    left.set_yticks(range(len(rows)), [f"{f} {a}" for f, a, _ in rows])
    left.invert_yaxis()
    _style(left, "Identity vs base (95% CI)", "delta identity_dino", "")

    medians = distances.groupby(["family", "arm"])["dreamsim_to_base"].median()
    arms = [("letterbox", a) for a in LETTERBOX] + [("crop", a) for a in CROP]
    for x, (family, arm) in enumerate(arms):
        color = CROP_COLOR if family == "crop" else LETTERBOX_COLOR
        right.bar(x, medians.loc[family, arm], width=0.6, color=color, edgecolor=SURFACE)
        right.annotate(
            f"{medians.loc[family, arm]:.3f}",
            (x, medians.loc[family, arm]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
            color=INK,
        )
    right.set_xticks(range(len(arms)), [arm for _, arm in arms])
    s2 = summary["S2_adapter_strength"]
    _style(
        right, "How far each adapter moves a tile", "adapter", "median DreamSim distance to base"
    )
    right.text(
        0.02,
        0.97,
        f"crop / letterbox: {s2['ratio']:.2f} [{s2['ci_low']:.2f}, {s2['ci_high']:.2f}]",
        transform=right.transAxes,
        va="top",
        fontsize=9,
        color=INK_2,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2},
    )
    right.margins(y=0.15)
    handles = [
        plt.Line2D([], [], color=LETTERBOX_COLOR, marker="o", linewidth=2, label="letterbox"),
        plt.Line2D([], [], color=CROP_COLOR, marker="o", linewidth=2, label="crop"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------- report


def write_report(summary: dict, out: Path) -> None:
    p = summary["primary"]
    s1, s2, s3 = summary["S1_crop_minus_letterbox"], summary["S2_adapter_strength"], summary["S3"]
    lines = [
        "# Study 1b run A results",
        "",
        "Object-centred square crops instead of letterboxed tiles, everything else as in "
        "Study 1 ([protocol](../../docs/study1b_protocol.md)).",
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
        f"Crop adapter minus base, {p['n_prompts']} prompts x {len(p['seeds'])} seeds, "
        "95% cluster-bootstrap CI:",
        "",
        "| Metric | Base | Pooled crop | Delta (pooled) | " + " | ".join(p["replicates"]) + " |",
        "| --- | --: | --: | --: | " + " | ".join("--:" for _ in p["replicates"]) + " |",
    ]
    for metric, entry in p["metrics"].items():
        pooled = entry["pooled"]
        reps = " | ".join(fmt(entry["replicates"][r]) for r in p["replicates"])
        lines.append(
            f"| {metric} | {pooled['mean_a']:.3f} | {pooled['mean_b']:.3f} "
            f"| {fmt(pooled)} | {reps} |"
        )
    lines += [
        "",
        f"Holm-adjusted Wilcoxon p: {json.dumps({k: round(v, 4) for k, v in p['holm_p'].items()})}",
        "",
        "## Crop vs letterbox",
        "",
    ]
    gate = summary["base_identical"]
    lines.append(
        f"Base images identical to Study 1: {gate['identical_images']}/"
        f"{gate['letterbox_run_base_images']} (**{'yes' if gate['identical'] else 'no'}**)."
    )
    if gate["identical"]:
        lines += [
            "",
            "| Analysis | Estimate, 95% CI |",
            "| --- | --: |",
            *(
                f"| S1 delta {m}, pooled crop - pooled letterbox | {fmt(e)} |"
                for m, e in s1.items()
            ),
            *(
                f"| S3 delta {m}, c3 - r3 (same seed) | {fmt(e)} |"
                for m, e in s3["identity"].items()
            ),
            f"| S2 median distance to base: crop / letterbox | {s2['ratio']:.2f} "
            f"[{s2['ci_low']:.2f}, {s2['ci_high']:.2f}] "
            f"({s2['median_numerator']:.3f} / {s2['median_denominator']:.3f}) |",
            f"| S3 median distance to base: c3 / r3 | {s3['strength']['ratio']:.2f} "
            f"[{s3['strength']['ci_low']:.2f}, {s3['strength']['ci_high']:.2f}] |",
            "",
            "![Crop vs letterbox](figures/crop_vs_letterbox.png)",
        ]
    else:
        lines += ["", "S1-S3 are not reported: the two runs do not share a base (protocol)."]
    bars = summary["S4_bars"]
    lines += [
        "",
        "## S4: bars in generated grids",
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


def analyse(crop: pd.DataFrame, letterbox: pd.DataFrame, distances, bars, human=None) -> dict:
    primary = primary_analysis(crop, replicates=CROP)
    decision = evaluate_gates(primary, human)
    prompts, seeds = _axes(crop)
    identity = {}
    for family, grids, arms in (("letterbox", letterbox, LETTERBOX), ("crop", crop, CROP)):
        base = cell_matrix(grids, "base", "identity_dino", prompts, seeds)
        identity[family] = {
            arm: paired_cluster_bootstrap(
                base,
                cell_matrix(grids, arm, "identity_dino", prompts, seeds),
                n_boot=N_BOOT,
                seed=SEED,
            ).as_dict()
            for arm in arms
        } | {
            "pooled": paired_cluster_bootstrap(
                base,
                pooled_matrix(grids, arms, "identity_dino", prompts, seeds),
                n_boot=N_BOOT,
                seed=SEED,
            ).as_dict()
        }
    s2 = median_ratio(distances, CROP, LETTERBOX)
    bar_result = bar_summary(crop, letterbox, bars)
    return {
        "primary": primary,
        "decision": decision,
        "base_identical": base_identical(crop, letterbox),
        "identity_vs_base": identity,
        "S1_crop_minus_letterbox": crop_vs_letterbox(crop, letterbox, CROP, LETTERBOX),
        "S2_adapter_strength": s2,
        "S3": {
            "identity": crop_vs_letterbox(crop, letterbox, ("c3",), ("r3",)),
            "strength": median_ratio(distances, ("c3",), ("r3",)),
        },
        "S4_bars": bar_result,
        "reading": reading(decision, s2, bar_result),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crop-metrics", type=Path, default=Path("results/metrics/study1b_main"))
    parser.add_argument("--letterbox-metrics", type=Path, default=Path("results/metrics/main"))
    parser.add_argument("--tables", type=Path, default=Path("results/study1b"))
    parser.add_argument("--humans", type=Path, default=Path("results/study1b/humans.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/results/study1b_8bit/main/generation_manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/study1b"))
    args = parser.parse_args(argv)

    crop, letterbox = load_grids(args.crop_metrics), load_grids(args.letterbox_metrics)
    distances = pd.read_csv(args.tables / "adapter_distance.csv")
    bars = pd.read_csv(args.tables / "bars.csv")
    human = json.loads(args.humans.read_text()) if args.humans.is_file() else None
    summary = analyse(crop, letterbox, distances, bars, human)
    summary["code_versions"] = code_versions(args.manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    figure(summary, distances, args.output_dir / "figures" / "crop_vs_letterbox.png")
    write_report(summary, args.output_dir / "report.md")
    print(f"verdict: {summary['decision']['verdict']}\nreading: {summary['reading']}")


if __name__ == "__main__":
    main()
