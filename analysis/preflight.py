"""Numbers the Study 1 protocol fixes before any LoRA image is scored.

1. Azimuth sign: which sign maps Orient Anything's relative azimuth onto the
   caption convention, chosen on the real holdout composites only.
2. Minimum detectable effects from base-model seed-to-seed noise.

    python -m analysis.preflight \
        --controls results/metrics/controls --base results/metrics/holdout_bf16_base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics.stats import minimum_detectable_effect
from evaluation.metrics.views import angular_distance

MDE_METRICS = ("identity_dino", "identity_dreamsim", "any_same_view", "distinct_views")


def calibrate_sign(pairs: pd.DataFrame, tolerance: float = 45.0) -> dict[str, float]:
    real = pairs[pairs["grid_id"].str.endswith("/real") & pairs["expected_rel"].notna()]
    out = {"n_pairs": int(len(real))}
    for sign in (1, -1):
        hits = [
            angular_distance(sign * rel, exp) <= tolerance
            for rel, exp in zip(real["rel_azimuth"], real["expected_rel"], strict=True)
        ]
        out[f"accuracy_sign_{sign:+d}"] = float(np.mean(hits))
    out["chosen_sign"] = 1 if out["accuracy_sign_+1"] >= out["accuracy_sign_-1"] else -1
    return out


def mde_table(grids: pd.DataFrame, seeds_per_condition: int = 4) -> list[dict[str, float]]:
    rows = []
    for metric in MDE_METRICS:
        wide = grids.pivot_table(index="prompt_id", columns="seed", values=metric, aggfunc="mean")
        seed_a, seed_b = sorted(wide.columns)[:2]
        null = (wide[seed_a] - wide[seed_b]).to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "base_mean": float(grids[metric].astype(float).mean()),
                "null_sd_single_seed": float(np.nanstd(null, ddof=1)),
                "mde": minimum_detectable_effect(
                    null, n_prompts=len(wide), seeds_per_condition=seeds_per_condition
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/preflight.json"))
    args = parser.parse_args(argv)

    sign = calibrate_sign(pd.read_csv(args.controls / "pairs.csv"))
    mde = mde_table(pd.read_csv(args.base / "grids.csv"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"azimuth_sign": sign, "mde": mde}, indent=2) + "\n")
    print(json.dumps(sign, indent=2))
    print(pd.DataFrame(mde).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
