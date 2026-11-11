"""Control grids that every Study 1 metric must rank correctly before it is trusted.

Built from the 42 real MVImgNet holdout composites, which never entered training:

* ``real``         - the composite itself. The ceiling for a real four-view capture.
* ``dup4``         - the top-left tile copied four times. Perfect "identity", one view.
* ``dup4_jitter``  - the same copy with small crop/brightness jitter, so duplicate
                     detection cannot rely on pixel-identical tiles.
* ``shuffled``     - the real tiles in reversed order. Identity and diversity must not
                     change (they are permutation invariant); pose agreement with the
                     caption must collapse, because the labels now point at other tiles.
* ``mixed``        - four tiles from four objects in four different categories. Identity
                     must drop well below ``real``.

Pose agreement for controls is scored against each composite's own Gemini caption,
not against the concise evaluation prompt (whose F/R/B/L layout the real photos do
not follow).

    python -m evaluation.controls build --output-dir evaluation/results/controls
    python -m evaluation.controls check --metrics-dir results/metrics/controls
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance

from evaluation.metrics.views import TILE_POSITIONS, join_tiles, parse_tile_views, split_grid

CONTROL_SEED = 17
KINDS = ("real", "dup4", "dup4_jitter", "shuffled", "mixed")


def load_holdout(split_root: Path) -> list[dict]:
    manifest = split_root / "split_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["split"] == "holdout"]
    for row in rows:
        row["category"] = row["source_instance"].split("/")[1]
        row["caption_text"] = (split_root / row["caption"]).read_text().strip()
    return sorted(rows, key=lambda row: row["stem"])


def jitter(tile: Image.Image, rng: random.Random) -> Image.Image:
    w, h = tile.size
    scale = rng.uniform(0.90, 0.98)
    cw, ch = int(w * scale), int(h * scale)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    out = tile.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.Resampling.BICUBIC)
    return ImageEnhance.Brightness(out).enhance(rng.uniform(0.92, 1.08))


def caption_from_labels(labels: list[str | None]) -> str:
    parts = [
        f"[{pos}] {label or 'indeterminate'}"
        for pos, label in zip(TILE_POSITIONS, labels, strict=True)
    ]
    return "; ".join(parts)


def pick_mixed_partners(rows: list[dict], index: int, rng: random.Random) -> list[int]:
    """Three partners from three other categories, distinct from the anchor's."""
    used = {rows[index]["category"]}
    order = list(range(len(rows)))
    rng.shuffle(order)
    partners = []
    for candidate in order:
        if rows[candidate]["category"] not in used:
            partners.append(candidate)
            used.add(rows[candidate]["category"])
        if len(partners) == 3:
            return partners
    raise ValueError("Not enough categories to build a mixed control")


def build(split_root: Path, output_dir: Path) -> Path:
    rows = load_holdout(split_root)
    rng = random.Random(CONTROL_SEED)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    tiles = {
        row["stem"]: split_grid(Image.open(split_root / row["image"]).convert("RGB"))
        for row in rows
    }
    records = []
    for index, row in enumerate(rows):
        stem, own = row["stem"], tiles[row["stem"]]
        caption = row["caption_text"]
        partners = pick_mixed_partners(rows, index, rng)
        sources = [row] + [rows[p] for p in partners]
        mixed_tiles = [tiles[src["stem"]][k] for k, src in enumerate(sources)]
        mixed_labels = [
            parse_tile_views(src["caption_text"])[TILE_POSITIONS[k]].label
            for k, src in enumerate(sources)
        ]
        variants = {
            "real": (own, caption),
            "dup4": ([own[0]] * 4, caption),
            "dup4_jitter": ([own[0]] + [jitter(own[0], rng) for _ in range(3)], caption),
            "shuffled": (own[::-1], caption),
            "mixed": (mixed_tiles, caption_from_labels(mixed_labels)),
        }
        for kind, (grid_tiles, view_caption) in variants.items():
            path = images_dir / kind / f"{stem}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            join_tiles(grid_tiles).save(path)
            records.append(
                {
                    "grid_id": f"{stem}/0/{kind}",
                    "image_path": str(path.relative_to(output_dir)),
                    "prompt_id": stem,
                    "seed": 0,
                    "condition": kind,
                    "arm": kind,
                    "view_caption": view_caption,
                    "source_image_sha256": row["image_sha256"],
                    "mixed_sources": [src["stem"] for src in sources] if kind == "mixed" else None,
                }
            )
    grid_list = output_dir / "grids.jsonl"
    grid_list.write_text("".join(json.dumps(r) + "\n" for r in records))
    return grid_list


# Each check: (metric, description, predicate over per-arm means). Written before any
# control metrics were computed; a metric that fails is not used for conclusions.
CHECKS = [
    ("identity_dino", "dup4 > real > mixed", lambda m: m["dup4"] > m["real"] > m["mixed"]),
    ("identity_dino", "dup4_jitter > real", lambda m: m["dup4_jitter"] > m["real"]),
    ("identity_dino", "|shuffled - real| < 0.01", lambda m: abs(m["shuffled"] - m["real"]) < 0.01),
    ("identity_dreamsim", "dup4 > real > mixed", lambda m: m["dup4"] > m["real"] > m["mixed"]),
    ("identity_dreamsim", "dup4_jitter > real", lambda m: m["dup4_jitter"] > m["real"]),
    ("identity_dino_unmasked", "dup4 > real > mixed", lambda m: m["dup4"] > m["real"] > m["mixed"]),
    ("distinct_views", "real >= dup4 + 1.0", lambda m: m["real"] >= m["dup4"] + 1.0),
    ("distinct_views", "dup4_jitter <= 1.5", lambda m: m["dup4_jitter"] <= 1.5),
    ("any_same_view", "dup4 >= 0.9", lambda m: m["dup4"] >= 0.9),
    ("any_same_view", "dup4_jitter >= 0.9", lambda m: m["dup4_jitter"] >= 0.9),
    ("pose_accuracy", "real >= shuffled + 0.15", lambda m: m["real"] >= m["shuffled"] + 0.15),
    ("pose_accuracy", "real >= dup4 + 0.15", lambda m: m["real"] >= m["dup4"] + 0.15),
]


def check(metrics_dir: Path) -> list[dict]:
    import pandas as pd

    grids = pd.read_csv(metrics_dir / "grids.csv")
    columns = sorted({metric for metric, _, _ in CHECKS})
    means = grids.groupby("arm")[columns].mean(numeric_only=False)
    results = []
    for metric, description, predicate in CHECKS:
        per_arm = {arm: float(means.loc[arm, metric]) for arm in means.index}
        results.append(
            {"metric": metric, "check": description, "passed": bool(predicate(per_arm)), **per_arm}
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or check Study 1 metric controls")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split-root", type=Path, default=Path("training/study1_pilot"))
    b.add_argument("--output-dir", type=Path, default=Path("evaluation/results/controls"))
    c = sub.add_parser("check")
    c.add_argument("--metrics-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(build(args.split_root, args.output_dir))
        return
    results = check(args.metrics_dir)
    arms = [k for k in results[0] if k not in ("metric", "check", "passed")]
    print(f"{'metric':<24} {'check':<26} {'ok':<4} " + " ".join(f"{a:>11}" for a in arms))
    for r in results:
        values = " ".join(f"{r[a]:>11.3f}" for a in arms)
        print(f"{r['metric']:<24} {r['check']:<26} {'yes' if r['passed'] else 'NO':<4} {values}")
    (args.metrics_dir / "control_checks.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
