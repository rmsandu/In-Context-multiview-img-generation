"""What did the LoRA actually change? Per-module anatomy of a trained adapter.

For a LoRA with down-projection A (r x in), up-projection B (out x r) and scale
alpha/r, the weight update is

    dW = (alpha / r) * B @ A          (out x in, rank <= r)

Everything below is computed without forming dW: with thin QR decompositions
B = Qb Rb and A^T = Qa Ra, we have B @ A = Qb (Rb Ra^T) Qa^T, so the singular
values of dW are those of the tiny r x r matrix Rb Ra^T. Likewise the Frobenius
inner product of two updates is trace((B1^T B2)(A2 A1^T)), again r x r.

Reported per module:
* ``norm``: ||dW||_F, and ``rel_norm`` = ||dW||_F / ||W||_F against the frozen
  FLUX weight (how far the module moved).
* ``eff_rank``: exp(entropy) of the normalised singular values (Roy & Vetterli,
  2007). 16 means the update uses every rank equally; 1 means one direction.
* ``cos_<a>_<b>``: cosine between the updates of two adapters.
* ``ovA_<a>_<b>`` / ``ovB_<a>_<b>``: overlap of the input (row space of A) and
  output (column space of B) subspaces, as the mean squared cosine of the
  principal angles: 1 for the same subspace, about r/d for unrelated ones.

Why both: LoRA initialises A randomly and B at zero, and with a small learning
rate A barely leaves its initial subspace. Two runs from different seeds then
write their updates into unrelated input subspaces, so their weight-space
cosine is near zero *even if they learned the same behaviour*. The B overlap
shows whether they at least agree on which output directions to move.

    python -m analysis.lora_anatomy --lora r1=output/study1_pilot/study1_pilot.safetensors \
        --lora r2=output/study1_pilot_v2/study1_pilot_v2.safetensors
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

ALPHA_OVER_RANK = 1.0  # configs/study1_pilot.yaml: linear 16, linear_alpha 16


def load_lora(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from safetensors import safe_open

    pairs: dict[str, dict[str, np.ndarray]] = {}
    with safe_open(str(path), "np") as handle:
        for key in handle.keys():
            match = re.fullmatch(r"transformer\.(.+)\.lora_([AB])\.weight", key)
            if match is None:
                raise ValueError(f"Unexpected tensor in {path}: {key}")
            pairs.setdefault(match.group(1), {})[match.group(2)] = handle.get_tensor(key).astype(
                np.float64
            )
    return {module: (ab["A"], ab["B"]) for module, ab in pairs.items()}


def classify(module: str) -> dict[str, object]:
    """Block type, index and a coarse role for one adapted linear layer."""
    stream = "double" if module.startswith("transformer_blocks.") else "single"
    index = int(module.split(".")[1])
    leaf = module.split(".", 2)[2]
    if "norm" in leaf:
        role = "modulation (adaLN)"
    elif stream == "double" and (leaf.startswith("attn.add_") or "to_add_out" in leaf):
        role = "text attention"
    elif leaf.startswith("attn."):
        role = "image attention" if stream == "double" else "joint attention"
    elif "ff_context" in leaf:
        role = "text MLP"
    elif leaf.startswith("ff.") or leaf == "proj_mlp":
        role = "image MLP" if stream == "double" else "joint MLP"
    elif leaf == "proj_out":
        role = "joint attn+MLP out"
    else:
        role = "other"
    return {"stream": stream, "block": index, "leaf": leaf, "role": role}


def _core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, rb = np.linalg.qr(b)
    _, ra = np.linalg.qr(a.T)
    return ALPHA_OVER_RANK * rb @ ra.T


def module_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    sv = np.linalg.svd(_core(a, b), compute_uv=False)
    norm = float(np.sqrt(np.sum(sv**2)))
    p = sv / sv.sum() if sv.sum() > 0 else sv
    p = p[p > 0]
    return {
        "norm": norm,
        "sigma_max": float(sv[0]),
        "eff_rank": float(np.exp(-np.sum(p * np.log(p)))),
        "top1_energy": float(sv[0] ** 2 / np.sum(sv**2)) if norm > 0 else float("nan"),
    }


def subspace_overlap(m1: np.ndarray, m2: np.ndarray) -> float:
    """Mean squared cosine of principal angles between the column spaces of m1, m2."""
    q1, _ = np.linalg.qr(m1)
    q2, _ = np.linalg.qr(m2)
    return float(np.mean(np.linalg.svd(q1.T @ q2, compute_uv=False) ** 2))


def inner(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray) -> float:
    return float(ALPHA_OVER_RANK**2 * np.trace((b1.T @ b2) @ (a2 @ a1.T)))


def base_weight_norms(modules: list[str]) -> dict[str, float]:
    """||W||_F of the frozen FLUX.1-dev weights, read one tensor at a time."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    repo = "black-forest-labs/FLUX.1-dev"
    index_path = hf_hub_download(
        repo, "transformer/diffusion_pytorch_model.safetensors.index.json", local_files_only=True
    )
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for module in modules:
        by_shard.setdefault(weight_map[f"{module}.weight"], []).append(module)
    norms = {}
    for shard, names in by_shard.items():
        shard_path = hf_hub_download(repo, f"transformer/{shard}", local_files_only=True)
        with safe_open(shard_path, "pt") as handle:
            for name in names:
                weight = handle.get_tensor(f"{name}.weight").float()
                norms[name] = float(weight.norm())
    return norms


def analyse(loras: dict[str, Path], with_base: bool):
    import pandas as pd

    loaded = {label: load_lora(path) for label, path in loras.items()}
    modules = sorted(next(iter(loaded.values())))
    for label, lora in loaded.items():
        if sorted(lora) != modules:
            raise ValueError(f"{label} adapts a different set of modules")
    base = base_weight_norms(modules) if with_base else {}

    rows = []
    totals = {label: 0.0 for label in loaded}
    cross = {pair: 0.0 for pair in itertools.combinations(loaded, 2)}
    for module in modules:
        row = {"module": module, **classify(module)}
        for label, lora in loaded.items():
            a, b = lora[module]
            stats = module_stats(a, b)
            totals[label] += stats["norm"] ** 2
            row |= {f"{key}_{label}": value for key, value in stats.items()}
            if base:
                row[f"rel_norm_{label}"] = stats["norm"] / base[module]
        for x, y in cross:
            ip = inner(*loaded[x][module], *loaded[y][module])
            cross[(x, y)] += ip
            denom = row[f"norm_{x}"] * row[f"norm_{y}"]
            row[f"cos_{x}_{y}"] = ip / denom if denom else float("nan")
            (ax, bx), (ay, by) = loaded[x][module], loaded[y][module]
            row[f"ovA_{x}_{y}"] = subspace_overlap(ax.T, ay.T)
            row[f"ovB_{x}_{y}"] = subspace_overlap(bx, by)
        if base:
            row["base_norm"] = base[module]
        a0, b0 = next(iter(loaded.values()))[module]
        row["chance_overlap_A"] = a0.shape[0] / a0.shape[1]
        row["chance_overlap_B"] = b0.shape[1] / b0.shape[0]
        rows.append(row)

    frame = pd.DataFrame(rows)
    summary = {
        "adapters": {label: str(path) for label, path in loras.items()},
        "n_modules": len(modules),
        "total_update_norm": {label: float(np.sqrt(v)) for label, v in totals.items()},
        "global_cosine": {
            f"{x}_{y}": cross[(x, y)] / np.sqrt(totals[x] * totals[y]) for x, y in cross
        },
        "median_subspace_overlap": {
            f"{x}_{y}": {
                "A": float(frame[f"ovA_{x}_{y}"].median()),
                "B": float(frame[f"ovB_{x}_{y}"].median()),
            }
            for x, y in cross
        },
        "median_chance_overlap": {
            "A": float(frame["chance_overlap_A"].median()),
            "B": float(frame["chance_overlap_B"].median()),
        },
    }
    return frame, summary


def figure(summary: dict, out: Path) -> None:
    """Subspace overlap and update cosine: same run at earlier steps vs other seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink_2, muted, grid, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    colour_a, colour_b = "#eb6834", "#2a78d6"
    overlap, cosine = summary["median_subspace_overlap"], summary["global_cosine"]
    rows = [
        (f"r1: step {step} vs final", f"r1_c{step}")
        for step in (400, 300, 200, 100)
        if f"r1_c{step}" in overlap
    ] + [
        (f"{x} vs {y} (other seed)", f"{x}_{y}")
        for x, y in (("r1", "r2"), ("r1", "r3"), ("r2", "r3"))
        if f"{x}_{y}" in overlap
    ]
    labels, keys = zip(*rows)
    y = np.arange(len(rows))[::-1]
    chance = summary["median_chance_overlap"]["A"]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9.6, 3.9), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]}
    )
    fig.patch.set_facecolor(surface)
    for ax in (left, right):
        ax.set_facecolor(surface)
        ax.grid(True, axis="x", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink_2, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    a_vals = [overlap[k]["A"] for k in keys]
    b_vals = [overlap[k]["B"] for k in keys]
    for yy, a, b in zip(y, a_vals, b_vals, strict=True):
        left.plot([a, b], [yy, yy], color=grid, linewidth=2, zorder=1)
    left.scatter(a_vals, y, s=64, color=colour_a, edgecolor=surface, linewidth=2, zorder=3)
    left.scatter(b_vals, y, s=64, color=colour_b, edgecolor=surface, linewidth=2, zorder=3)
    left.axvline(chance, color=muted, linestyle="--", linewidth=1)
    left.text(chance * 1.12, y[-1] - 0.75, f"chance = r/d = {chance:.4f}", color=muted, fontsize=8)
    first_seed = len([k for k in keys if "_c" in k])  # label the first cross-seed row
    left.text(
        a_vals[first_seed] * 1.15, y[first_seed] + 0.35, "A (input side)", color=ink, fontsize=8.5
    )
    left.text(
        b_vals[first_seed],
        y[first_seed] + 0.35,
        "B (output side)",
        color=ink,
        fontsize=8.5,
        ha="center",
    )
    left.set_xscale("log")
    left.set_xlim(0.003, 1.6)
    left.set_ylim(y[-1] - 1.1, y[0] + 0.9)
    left.set_yticks(y, labels)
    left.set_title(
        "Subspace overlap, median over 494 modules", loc="left", fontsize=10.5, color=ink
    )
    left.set_xlabel("mean squared cosine of principal angles (log)", color=ink_2, fontsize=9)

    c_vals = [cosine[k] for k in keys]
    right.barh(y, c_vals, height=0.5, color=ink_2, edgecolor=surface, linewidth=2)
    for yy, c in zip(y, c_vals, strict=True):
        right.text(c + 0.02, yy, f"{c:.2f}", va="center", color=ink, fontsize=8.5)
    right.set_xlim(0, 1.12)
    right.set_title("Cosine of the whole update", loc="left", fontsize=10.5, color=ink)
    right.set_xlabel("cos(dW_x, dW_y), all modules stacked", color=ink_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lora", action="append", metavar="LABEL=PATH")
    parser.add_argument("--no-base", action="store_true", help="skip relative norms")
    parser.add_argument("--output-dir", type=Path, default=Path("results/lora_anatomy"))
    parser.add_argument(
        "--plot-only", action="store_true", help="redraw the figure from summary.json"
    )
    args = parser.parse_args(argv)
    if args.plot_only:
        summary = json.loads((args.output_dir / "summary.json").read_text())
        figure(summary, args.output_dir / "subspace_overlap.png")
        return
    if not args.lora:
        parser.error("--lora is required unless --plot-only is given")
    loras = {}
    for spec in args.lora:
        label, _, path = spec.partition("=")
        loras[label] = Path(path)
    table, summary = analyse(loras, with_base=not args.no_base)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "modules.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    first = next(iter(loras))
    cols = [f"norm_{first}", f"eff_rank_{first}"] + (
        [f"rel_norm_{first}"] if not args.no_base else []
    )
    print(table.groupby(["stream", "role"])[cols].median().round(4).to_string())
    figure(summary, args.output_dir / "subspace_overlap.png")


if __name__ == "__main__":
    main()
