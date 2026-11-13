"""Per-tile measurements for Study 1b run B's T2-T4 (docs/study1b_protocol.md, run B).

With prompt and seed fixed, the DreamSim distance between each tile of an
adapter's grid and the same tile of the base grid in the same run:

* ``runb``: the main run, ``b1``-``b3`` at step 4,000 (T2);
* ``runb_sweep``: the sweep run, ``b1_s2000``-``b3_s2000`` and ``b1``-``b3`` (T3).

Run A's distances (``crop``, T2's comparison) are reused from the committed
``results/study1b/adapter_distance.csv``, not recomputed. The looser bar detector
runs on every run B grid (T4). Every image is checked against its manifest hash.
Needs the generated images (not in git) and a GPU for DreamSim.

    .venv-eval/bin/python -m analysis.study1b_runb_effect
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.adapter_effect import soft_bar_fraction, tiles
from analysis.study1b_effect import load_manifest, tile_distances

RUNB = ("b1", "b2", "b3")
RUNB_2000 = ("b1_s2000", "b2_s2000", "b3_s2000")
RUNS = {
    "runb": (Path("evaluation/results/study1b_8bit/runb_main"), RUNB),
    "runb_sweep": (Path("evaluation/results/study1b_8bit/runb_sweep"), RUNB_2000 + RUNB),
}


def measure(runs: dict, embed) -> tuple[list[dict], list[dict]]:
    """Tile distances to base for each run's adapter arms, and bars on every grid."""
    cache: dict[Path, object] = {}

    def emb(path: Path):
        if path not in cache:
            cache[path] = embed(tiles(path))
        return cache[path]

    distances, bars = [], []
    for family, (root, arms) in runs.items():
        grids = load_manifest(root)
        conditions = {arm for _, _, arm in grids}
        if not set(arms) <= conditions:
            raise ValueError(f"{root} lacks arms {sorted(set(arms) - conditions)}")
        for (prompt, seed, arm), path in sorted(grids.items()):
            bars.append(
                {
                    "family": family,
                    "arm": arm,
                    "prompt_id": prompt,
                    "seed": seed,
                    "soft_bar_fraction": soft_bar_fraction(path),
                }
            )
            if arm not in arms:
                continue
            d = tile_distances(emb(grids[prompt, seed, "base"]), emb(path))
            distances += [
                {
                    "family": family,
                    "arm": arm,
                    "prompt_id": prompt,
                    "seed": seed,
                    "tile": tile,
                    "dreamsim_to_base": float(value),
                }
                for tile, value in enumerate(d)
            ]
        print(f"{family}: {len(cache)} grids embedded so far", flush=True)
    return distances, bars


def main(argv: list[str] | None = None) -> None:
    import pandas as pd

    from evaluation.metrics.embeddings import DreamSimEmbedder

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("results/study1b/runb"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    distances, bars = measure(RUNS, DreamSimEmbedder(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(distances).to_csv(args.output_dir / "adapter_distance.csv", index=False)
    pd.DataFrame(bars).to_csv(args.output_dir / "bars.csv", index=False)
    print(f"wrote {len(distances)} tile distances and {len(bars)} bar measurements")


if __name__ == "__main__":
    main()
