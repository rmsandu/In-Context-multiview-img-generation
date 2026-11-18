"""How much does the adapter change the picture at all?

The pre-registered metrics compare *properties* of base and LoRA grids. This
asks a more basic question: with prompt and seed fixed, how far is the LoRA
image from the base image, compared with how far two seeds of the base model
are from each other? It also checks the one feature every training composite
has: black side bars (portrait frames padded into square tiles).

Exploratory, not part of the pre-registered verdict. Needs the generated
images (not in git) and a GPU for DreamSim.

    .venv-eval/bin/python -m analysis.adapter_effect
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

RUNS = Path("evaluation/results/study1_8bit")
TRAIN = Path("training/study1_pilot/train")


def tiles(path: Path):
    from PIL import Image

    grid = Image.open(path).convert("RGB")
    w, h = grid.size
    return [grid.crop((x, y, x + w // 2, y + h // 2)) for y in (0, h // 2) for x in (0, w // 2)]


def soft_bar_fraction(path: Path, *, level: float = 40.0, q: float = 95.0) -> float:
    """Widest edge band (share of tile width) whose q-th percentile is below ``level``.

    Looser than ``evaluation.metrics.views.letterbox_fraction`` (which needs every
    pixel below 16), so it also catches grey or noisy bars a generator might draw.
    """
    best = 0.0
    for tile in tiles(path):
        grey = np.asarray(tile.convert("L").resize((256, 256)), dtype=float)
        for axis in (0, 1):
            dark = np.percentile(grey, q, axis=axis) < level
            run = int(np.argmin(dark)) if not dark.all() else len(dark)
            back = int(np.argmin(dark[::-1])) if not dark.all() else 0
            best = max(best, (run + back) / len(dark))
    return best


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def load(run: str) -> dict[tuple[str, int, str], Path]:
    root = RUNS / run
    rows = [json.loads(line) for line in (root / "generation_manifest.jsonl").open()]
    return {(r["prompt_id"], r["seed"], r["condition"]): root / r["output_path"] for r in rows}


def main(argv: list[str] | None = None) -> None:
    from evaluation.metrics.embeddings import DreamSimEmbedder

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("results/adapter_effect"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    model = DreamSimEmbedder(args.device)
    cache: dict[Path, np.ndarray] = {}

    def emb(path: Path) -> np.ndarray:
        if path not in cache:
            cache[path] = model(tiles(path))
        return cache[path]

    def dist(a: Path, b: Path) -> float:
        """Mean DreamSim distance between corresponding tiles of two grids."""
        return float(np.mean(1.0 - np.sum(emb(a) * emb(b), axis=1)))

    rows = []
    for run in ("main", "sweep"):
        grids = load(run)
        prompts = sorted({p for p, _, _ in grids})
        seeds = sorted({s for _, s, _ in grids})
        arms = sorted({c for _, _, c in grids} - {"base"})
        for p in prompts:
            for s in seeds:
                for arm in arms:
                    rows.append(
                        {
                            "run": run,
                            "prompt_id": p,
                            "seed": s,
                            "comparison": f"base vs {arm}",
                            "value": dist(grids[p, s, "base"], grids[p, s, arm]),
                        }
                    )
            for s1, s2 in itertools.combinations(seeds, 2):
                rows.append(
                    {
                        "run": run,
                        "prompt_id": p,
                        "seed": s1,
                        "comparison": "base seed vs seed",
                        "value": dist(grids[p, s1, "base"], grids[p, s2, "base"]),
                    }
                )
            if run == "main":
                for s in seeds:
                    base = emb(grids[p, s, "base"])
                    for a, b in itertools.combinations(("r1", "r2", "r3"), 2):
                        rows.append(
                            {
                                "run": run,
                                "prompt_id": p,
                                "seed": s,
                                "comparison": f"{a} vs {b}",
                                "value": dist(grids[p, s, a], grids[p, s, b]),
                            }
                        )
                        # do two replicates move the same tile in the same direction?
                        # null: replicate b's displacement on the next prompt instead,
                        # which keeps any direction every adapter shares on every image
                        q = prompts[(prompts.index(p) + 1) % len(prompts)]
                        da = emb(grids[p, s, a]) - base
                        for label, db in (
                            ("", emb(grids[p, s, b]) - base),
                            (" (other prompt)", emb(grids[q, s, b]) - emb(grids[q, s, "base"])),
                        ):
                            rows.append(
                                {
                                    "run": run,
                                    "prompt_id": p,
                                    "seed": s,
                                    "comparison": f"displacement cosine {a} {b}{label}",
                                    "value": float(np.mean(cosine_rows(da, db))),
                                }
                            )
        print(f"{run}: {len(cache)} grids embedded so far")

    import pandas as pd

    frame = pd.DataFrame(rows)
    # prompt is the unit: average within prompt first, then summarise across prompts
    per_prompt = frame.groupby(["run", "comparison", "prompt_id"])["value"].mean()
    summary = {
        f"{run} | {comp}": {
            "median_over_prompts": float(g.median()),
            "mean_over_prompts": float(g.mean()),
            "n_prompts": int(g.size),
        }
        for (run, comp), g in per_prompt.groupby(level=[0, 1])
    }

    main_grids = load("main")
    bars = {
        "training_composites": [soft_bar_fraction(p) for p in sorted(TRAIN.glob("*.png"))],
        **{
            arm: [soft_bar_fraction(path) for (_, _, c), path in main_grids.items() if c == arm]
            for arm in ("base", "r1", "r2", "r3")
        },
    }
    sweep_grids = load("sweep")
    bars["sweep s1.5"] = [
        soft_bar_fraction(path) for (_, _, c), path in sweep_grids.items() if c == "s1.5"
    ]
    bar_summary = {
        name: {
            "n": len(v),
            "share_with_bars_over_5pct": float(np.mean(np.array(v) > 0.05)),
            "median_fraction": float(np.median(v)),
        }
        for name, v in bars.items()
    }

    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "distances.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps({"dreamsim_tiles": summary, "side_bars": bar_summary}, indent=2) + "\n"
    )
    for key, value in summary.items():
        print(f"{key:32s} median {value['median_over_prompts']:.3f}")
    for key, value in bar_summary.items():
        print(f"bars {key:22s} {value['share_with_bars_over_5pct']:.2f} (n={value['n']})")


if __name__ == "__main__":
    main()
