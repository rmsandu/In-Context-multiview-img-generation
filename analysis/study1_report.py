"""Study 1 analysis exactly as fixed in docs/study1_protocol.md.

Reads the metric tables written by ``evaluation.metrics.run_metrics`` and writes
``results/study1/summary.json``, ``results/study1/report.md`` and the figures in
``results/study1/figures``. Nothing here re-runs a model.

    python -m analysis.study1_report
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics.stats import holm, paired_cluster_bootstrap, wilcoxon_signed_rank
from evaluation.metrics.views import count_distinct_views

REPLICATES = ("r1", "r2", "r3")
PRIMARY = {"A1": "identity_dino", "A2": "distinct_views"}
REPORTED = (
    "identity_dino",
    "identity_dreamsim",
    "identity_dino_unmasked",
    "distinct_views",
    "distinct_views_cal",
    "any_same_view",
    "valid_grid",
    "max_letterbox_fraction",
    "pose_accuracy",
)
# Protocol deviation 1: Orient Anything reads camera angles at ~0.70x their COLMAP
# value, so 30 deg of real camera motion shows up as ~21 deg. A2-cal recounts
# distinct views with this threshold as a sensitivity analysis; A2 is unchanged.
CALIBRATED_SAME_VIEW_DEG = 21.0
N_BOOT = 10_000
SEED = 17

# Light-mode chart tokens (validated: blue/orange/aqua pass all-pairs CVD checks;
# aqua is below 3:1 contrast, so every aqua mark carries a direct label).
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
LORA, BASE, REAL = "#2a78d6", "#eb6834", "#1baf7a"


# ----------------------------------------------------------------------- loading


def load_grids(path: Path) -> pd.DataFrame:
    grids = pd.read_csv(path / "grids.csv")
    for column in ("any_same_view", "valid_grid"):
        grids[column] = grids[column].astype(float)
    if (path / "pairs.csv").is_file():
        grids["distinct_views_cal"] = grids["grid_id"].map(
            calibrated_distinct_views(pd.read_csv(path / "pairs.csv"))
        )
    return grids


def calibrated_distinct_views(pairs: pd.DataFrame) -> pd.Series:
    """Distinct views per grid at CALIBRATED_SAME_VIEW_DEG. A missing angle is 'not same'."""
    wrapped = (pairs["rel_azimuth"] + 180.0) % 360.0 - 180.0
    pairs = pairs.assign(same=wrapped.abs() < CALIBRATED_SAME_VIEW_DEG)
    return pairs.groupby("grid_id")[["i", "j", "same"]].apply(
        lambda g: count_distinct_views({(int(r.i), int(r.j)): bool(r.same) for r in g.itertuples()})
    )


def cell_matrix(grids: pd.DataFrame, arm: str, metric: str, prompts, seeds) -> np.ndarray:
    """Values for one arm as a prompts x seeds array aligned across arms."""
    table = grids[grids["arm"] == arm].pivot_table(
        index="prompt_id", columns="seed", values=metric, aggfunc="mean"
    )
    return table.reindex(index=prompts, columns=seeds).to_numpy(dtype=float)


def _axes(grids: pd.DataFrame) -> tuple[list[str], list[int]]:
    return sorted(grids["prompt_id"].unique()), sorted(grids["seed"].unique())


# ---------------------------------------------------------------------- analysis


def primary_analysis(grids: pd.DataFrame) -> dict:
    prompts, seeds = _axes(grids)
    reps = [arm for arm in REPLICATES if arm in set(grids["arm"])]
    out: dict = {"replicates": reps, "n_prompts": len(prompts), "seeds": seeds, "metrics": {}}
    for metric in REPORTED:
        base = cell_matrix(grids, "base", metric, prompts, seeds)
        per_rep = {r: cell_matrix(grids, r, metric, prompts, seeds) for r in reps}
        with np.errstate(all="ignore"):
            pooled = np.nanmean(np.stack(list(per_rep.values())), axis=0)
        entry = {
            "pooled": paired_cluster_bootstrap(base, pooled, n_boot=N_BOOT, seed=SEED).as_dict(),
            "replicates": {
                r: paired_cluster_bootstrap(base, v, n_boot=N_BOOT, seed=SEED).as_dict()
                for r, v in per_rep.items()
            },
        }
        if metric in PRIMARY.values():
            entry["wilcoxon_p"] = wilcoxon_signed_rank(base, pooled)
        out["metrics"][metric] = entry
    pvals = {k: out["metrics"][m]["wilcoxon_p"] for k, m in PRIMARY.items()}
    out["holm_p"] = holm(pvals)
    return out


def evaluate_gates(primary: dict, human: dict | None) -> dict:
    m = primary["metrics"]
    a1 = m["identity_dino"]
    a1_signs = {np.sign(v["diff"]) for v in a1["replicates"].values()}
    gates = {
        "A1_identity": {
            "pass": a1["pooled"]["ci_low"] > 0 and a1_signs == {1.0},
            "rule": "pooled CI lower bound > 0 and positive in every replicate",
        },
        "A2_distinct_views": {
            "pass": m["distinct_views"]["pooled"]["ci_low"] > -0.25,
            "rule": "pooled CI lower bound > -0.25",
        },
        "V_valid_grid": {
            "pass": m["valid_grid"]["pooled"]["diff"] > -0.05,
            "rule": "pooled difference > -0.05",
        },
    }
    if human is None:
        gates["H1_humans"] = {"pass": None, "rule": "pending: no human scores yet"}
    else:
        gates["H1_humans"] = {
            "pass": human["lora_preferred"]["ci_low"] > 0.5,
            "rule": "2AFC LoRA-preferred share, CI lower bound > 0.5",
        }
    decided = [g["pass"] for g in gates.values() if g["pass"] is not None]
    if not all(decided):
        verdict = "no evidence that the LoRA improves identity without a diversity cost"
    elif human is None:
        verdict = "automated gates pass; verdict waits for the human endpoint"
    else:
        verdict = "the LoRA improves identity without a diversity cost"
    return {"gates": gates, "verdict": verdict}


def sweep_analysis(grids: pd.DataFrame) -> dict:
    prompts, seeds = _axes(grids)
    arms = sorted(set(grids["arm"]) - {"base"})
    out = {"scale": [], "checkpoint": []}
    for arm in arms:
        if match := re.fullmatch(r"s([\d.]+)", arm):
            kind, x = "scale", float(match.group(1))
        elif match := re.fullmatch(r"c(\d+)", arm):
            kind, x = "checkpoint", int(match.group(1))
        else:
            continue
        row = {"arm": arm, "x": x}
        for metric in ("identity_dino", "distinct_views", "max_letterbox_fraction"):
            base = cell_matrix(grids, "base", metric, prompts, seeds)
            value = cell_matrix(grids, arm, metric, prompts, seeds)
            row[metric] = paired_cluster_bootstrap(base, value, n_boot=N_BOOT, seed=SEED).as_dict()
        out[kind].append(row)
    # Final checkpoint at scale 1 is both the end of the checkpoint curve and s1.0.
    final = next((r for r in out["scale"] if r["x"] == 1.0), None)
    if final:
        out["checkpoint"].append({**final, "arm": "s1.0", "x": 500})
    for kind in out:
        out[kind].sort(key=lambda r: r["x"])

    base_hash = grids[grids["arm"] == "base"].set_index(["prompt_id", "seed"])["image_sha256"]
    if "s0" in set(grids["arm"]):
        zero = grids[grids["arm"] == "s0"].set_index(["prompt_id", "seed"])["image_sha256"]
        out["scale0_identical_to_base"] = bool((zero.sort_index() == base_hash.sort_index()).all())
    return out


def quantization_check(main: pd.DataFrame, bf16: pd.DataFrame) -> dict:
    """LoRA-minus-base on seeds 1001/1002: bf16 July run vs this 8-bit run."""
    out = {}
    bf16_lora = sorted(set(bf16["arm"]) - {"base"})
    if len(bf16_lora) != 1:
        return {"skipped": f"expected one adapter arm in the bf16 run, got {bf16_lora}"}
    shared_seeds = sorted(set(bf16["seed"]) & set(main["seed"]))
    prompts = sorted(set(bf16["prompt_id"]) & set(main["prompt_id"]))
    for metric in ("identity_dino", "distinct_views", "identity_dreamsim"):
        deltas = {}
        for name, frame, arm in (("bf16", bf16, bf16_lora[0]), ("8bit", main, "r1")):
            base = cell_matrix(frame, "base", metric, prompts, shared_seeds)
            lora = cell_matrix(frame, arm, metric, prompts, shared_seeds)
            deltas[name] = paired_cluster_bootstrap(base, lora, n_boot=N_BOOT, seed=SEED).as_dict()
        base_8 = cell_matrix(main, "base", metric, prompts, shared_seeds)
        base_16 = cell_matrix(bf16, "base", metric, prompts, shared_seeds)
        deltas["base_8bit_minus_bf16"] = paired_cluster_bootstrap(
            base_16, base_8, n_boot=N_BOOT, seed=SEED
        ).as_dict()
        out[metric] = deltas
    return out


def code_versions(manifest: Path) -> dict:
    if not manifest.is_file():
        return {"manifest": f"{manifest} not available (generation outputs are not in git)"}
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    commits = {}
    for row in rows:
        version = row.get("code_version") or {}
        dirty = " (dirty)" if version.get("python_sources_dirty") else ""
        key = f"{version.get('git_commit', 'unknown')}{dirty}"
        commits[key] = commits.get(key, 0) + 1
    seconds = [r["elapsed_seconds"] for r in rows if r.get("elapsed_seconds")]
    return {
        "records": len(rows),
        "commits": commits,
        "single_commit": len(commits) == 1,
        "median_seconds_per_image": float(np.median(seconds)) if seconds else None,
    }


# ----------------------------------------------------------------------- figures


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def _mean_ci(values: np.ndarray, rng) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    boots = [rng.choice(values, size=len(values)).mean() for _ in range(2000)]
    return float(values.mean()), *np.quantile(boots, [0.025, 0.975])


def figure_identity_vs_views(main: pd.DataFrame, controls: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200, facecolor=SURFACE)
    _style(
        ax,
        "Identity vs. viewpoint diversity (per-prompt means, 95% CI)",
        "distinct viewpoints in the grid (of 4)",
        "identity: masked DINOv2 cosine between tiles",
    )

    def point(frame, colour, label, size=42, label_dx=0.04, label_dy=0.0, ink=INK):
        per_prompt = frame.groupby("prompt_id")[["identity_dino", "distinct_views"]].mean()
        x = _mean_ci(per_prompt["distinct_views"].to_numpy(), rng)
        y = _mean_ci(per_prompt["identity_dino"].to_numpy(), rng)
        ax.errorbar(
            x[0],
            y[0],
            xerr=[[x[0] - x[1]], [x[2] - x[0]]],
            yerr=[[y[0] - y[1]], [y[2] - y[0]]],
            fmt="none",
            ecolor=colour,
            elinewidth=1,
            capsize=0,
            alpha=0.7,
        )
        ax.scatter(x[0], y[0], s=size, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(
            label,
            (x[0], y[0]),
            xytext=(x[0] + label_dx, y[0] + label_dy),
            fontsize=8,
            color=ink,
            va="center",
        )

    for arm, label in (("dup4", "same tile x4"), ("mixed", "4 different objects")):
        point(controls[controls["arm"] == arm], MUTED, label, size=30, ink=INK_2)
    point(controls[controls["arm"] == "real"], REAL, "real photos (holdout)")
    point(main[main["arm"] == "base"], BASE, "FLUX base")
    for rep in REPLICATES:
        if rep in set(main["arm"]):
            point(main[main["arm"] == rep], LORA, f"LoRA {rep}", size=34)
    ax.set_xlim(0.8, 4.0)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def figure_paired_deltas(main: pd.DataFrame, primary: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    prompts, seeds = _axes(main)
    base = cell_matrix(main, "base", "identity_dino", prompts, seeds)
    reps = [cell_matrix(main, r, "identity_dino", prompts, seeds) for r in primary["replicates"]]
    pooled = np.nanmean(np.stack(reps), axis=0)
    delta = np.nanmean(pooled, axis=1) - np.nanmean(base, axis=1)
    order = np.argsort(delta)
    est = primary["metrics"]["identity_dino"]["pooled"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200, facecolor=SURFACE)
    _style(
        ax,
        "Per-prompt change in identity (LoRA - base)",
        "holdout prompt (sorted)",
        "delta identity",
    )
    ax.axhspan(est["ci_low"], est["ci_high"], color=LORA, alpha=0.12, linewidth=0)
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.axhline(est["diff"], color=LORA, linewidth=1.2)
    ax.scatter(np.arange(len(delta)), delta[order], s=16, color=LORA, zorder=3)
    ax.annotate(
        f"mean {est['diff']:+.3f}  [{est['ci_low']:+.3f}, {est['ci_high']:+.3f}]",
        (0.5, est["ci_high"]),
        xytext=(0.5, est["ci_high"] + 0.01),
        fontsize=8,
        color=INK,
    )
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def figure_sweep(sweep: dict, kind: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    rows = sweep[kind]
    if not rows:
        return
    metrics = (
        ("identity_dino", "delta identity"),
        ("distinct_views", "delta distinct views"),
        ("max_letterbox_fraction", "delta letterbox share"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=200, facecolor=SURFACE)
    x = [r["x"] for r in rows]
    xlabel = "LoRA scale" if kind == "scale" else "training step (scale 1.0)"
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        _style(ax, label, xlabel, "")
        d = np.array([r[metric]["diff"] for r in rows])
        lo = np.array([r[metric]["ci_low"] for r in rows])
        hi = np.array([r[metric]["ci_high"] for r in rows])
        ax.fill_between(x, lo, hi, color=LORA, alpha=0.15, linewidth=0)
        ax.plot(x, d, color=LORA, linewidth=2, marker="o", markersize=4)
        ax.axhline(0, color="#c3c2b7", linewidth=1)
    fig.suptitle(
        f"{kind.capitalize()} sweep: change vs. base, 12 prompts x 2 seeds",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- report


def fmt(est: dict) -> str:
    return f"{est['diff']:+.3f} [{est['ci_low']:+.3f}, {est['ci_high']:+.3f}]"


def write_report(summary: dict, out: Path) -> None:
    p = summary["primary"]
    lines = [
        "# Study 1 results",
        "",
        f"Verdict (pre-registered rule): **{summary['decision']['verdict']}**.",
        "",
        "| Gate | Rule | Pass |",
        "| --- | --- | :-: |",
    ]
    for name, gate in summary["decision"]["gates"].items():
        mark = {True: "yes", False: "no", None: "pending"}[gate["pass"]]
        lines.append(f"| {name} | {gate['rule']} | {mark} |")
    lines += [
        "",
        f"LoRA minus base, {p['n_prompts']} prompts x {len(p['seeds'])} seeds, "
        "95% cluster-bootstrap CI:",
        "",
        "| Metric | Base | Pooled LoRA | Delta (pooled) | " + " | ".join(p["replicates"]) + " |",
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
    ]
    if "sweep" in summary:
        s = summary["sweep"]
        lines += ["", "## Sweeps (12 prompts x 2 seeds)", ""]
        if "scale0_identical_to_base" in s:
            lines.append(
                f"Scale-0 arm byte-identical to base: **{s['scale0_identical_to_base']}**."
            )
        for kind in ("scale", "checkpoint"):
            lines += [
                "",
                f"| {kind} | delta identity | delta distinct views | delta letterbox |",
                "| --: | --: | --: | --: |",
            ]
            for r in s[kind]:
                lines.append(
                    f"| {r['x']} | {fmt(r['identity_dino'])} | {fmt(r['distinct_views'])} | "
                    f"{fmt(r['max_letterbox_fraction'])} |"
                )
    if "quantization" in summary and "skipped" not in summary["quantization"]:
        lines += [
            "",
            "## Quantization check (r1, seeds 1001-1002)",
            "",
            "| Metric | bf16 delta | 8-bit delta | base 8-bit - bf16 |",
            "| --- | --: | --: | --: |",
        ]
        for metric, d in summary["quantization"].items():
            lines.append(
                f"| {metric} | {fmt(d['bf16'])} | {fmt(d['8bit'])} "
                f"| {fmt(d['base_8bit_minus_bf16'])} |"
            )
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics"))
    parser.add_argument("--generations", type=Path, default=Path("evaluation/results/study1_8bit"))
    parser.add_argument("--humans", type=Path, default=Path("results/humans/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/study1"))
    args = parser.parse_args(argv)

    main_grids = load_grids(args.metrics / "main")
    controls = load_grids(args.metrics / "controls")
    human = json.loads(args.humans.read_text()) if args.humans.is_file() else None
    primary = primary_analysis(main_grids)
    summary = {
        "protocol": "docs/study1_protocol.md",
        "primary": primary,
        "decision": evaluate_gates(primary, human),
        "code_versions": {
            "main": code_versions(args.generations / "main" / "generation_manifest.jsonl")
        },
    }
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if (args.metrics / "sweep" / "grids.csv").is_file():
        sweep_grids = load_grids(args.metrics / "sweep")
        summary["sweep"] = sweep_analysis(sweep_grids)
        summary["code_versions"]["sweep"] = code_versions(
            args.generations / "sweep" / "generation_manifest.jsonl"
        )
        figure_sweep(summary["sweep"], "scale", figures / "scale_sweep.png")
        figure_sweep(summary["sweep"], "checkpoint", figures / "checkpoint_sweep.png")
    if (args.metrics / "holdout_bf16" / "grids.csv").is_file():
        summary["quantization"] = quantization_check(
            main_grids, load_grids(args.metrics / "holdout_bf16")
        )
    if human:
        summary["humans"] = human

    figure_identity_vs_views(main_grids, controls, figures / "identity_vs_views.png")
    figure_paired_deltas(main_grids, primary, figures / "paired_identity_deltas.png")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    write_report(summary, args.output_dir / "report.md")
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
