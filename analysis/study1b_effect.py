"""Per-tile measurements for Study 1b's S2 and S4 (docs/study1b_protocol.md).

S2: with prompt and seed fixed, the DreamSim distance between each tile of an
adapter's grid and the same tile of the base grid, for the letterbox adapters
(Study 1, r1-r3) and the crop adapters (run A, c1-c3). S4: the looser bar
detector from ``analysis.adapter_effect`` on every grid.

Every image is checked against its manifest hash first. Needs the generated
images (not in git) and a GPU for DreamSim. ``analysis.study1b_report`` reads the
tables this writes; it never runs a model.

    .venv-eval/bin/python -m analysis.study1b_effect
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.adapter_effect import soft_bar_fraction, tiles

LETTERBOX = ("r1", "r2", "r3")
CROP = ("c1", "c2", "c3")
RUNS = {
    "letterbox": Path("evaluation/results/study1_8bit/main"),
    "crop": Path("evaluation/results/study1b_8bit/main"),
}


def load_manifest(root: Path) -> dict[tuple[str, int, str], Path]:
    """(prompt, seed, arm) -> image path, after checking every image's hash."""
    out = {}
    for line in (root / "generation_manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        path = root / row["output_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["output_sha256"]:
            raise ValueError(f"{path} does not match its manifest hash")
        out[row["prompt_id"], row["seed"], row["condition"]] = path
    return out


def tile_distances(base: np.ndarray, adapter: np.ndarray) -> np.ndarray:
    """DreamSim distance per tile from unit-normalised embeddings (4 x d each)."""
    if base.shape != adapter.shape:
        raise ValueError(f"Embedding shapes differ: {base.shape} vs {adapter.shape}")
    return 1.0 - np.sum(base * adapter, axis=1)


def main(argv: list[str] | None = None) -> None:
    import pandas as pd

    from evaluation.metrics.embeddings import DreamSimEmbedder

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("results/study1b"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    grids = {family: load_manifest(root) for family, root in RUNS.items()}
    model = DreamSimEmbedder(args.device)
    cache: dict[Path, np.ndarray] = {}

    def emb(path: Path) -> np.ndarray:
        if path not in cache:
            cache[path] = model(tiles(path))
        return cache[path]

    distances, bars = [], []
    for family, arms in (("letterbox", LETTERBOX), ("crop", CROP)):
        run = grids[family]
        for (prompt, seed, arm), path in sorted(run.items()):
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
            d = tile_distances(emb(run[prompt, seed, "base"]), emb(path))
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(distances).to_csv(args.output_dir / "adapter_distance.csv", index=False)
    pd.DataFrame(bars).to_csv(args.output_dir / "bars.csv", index=False)
    print(f"wrote {len(distances)} tile distances and {len(bars)} bar measurements")


if __name__ == "__main__":
    main()
