"""Create randomized, blinded Study 1 image copies and scoring sheets.

Each rater gets two sheets over the same blinded A/B pairs:

* ``scorecard.csv``, one row per image: identity 1-5, distinct views 1-4, which tile
  pairs look like the same view (tiles numbered 1=top-left, 2=top-right,
  3=bottom-left, 4=bottom-right, e.g. ``12 34``), prompt fidelity, and validity.
* ``pairwise.csv``, one row per pair: which grid shows the same object more
  consistently across its four tiles (``A``, ``B`` or ``tie``). This forced
  choice is the human primary endpoint in docs/study1_protocol.md.

The condition key is written separately and must stay with the organiser.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

from evaluation.generate_pairs import validate_paired_records

SCORE_FIELDS = (
    "identity_consistency_1_5",
    "distinct_views_1_4",
    "duplicate_tile_pairs",
    "prompt_fidelity_1_5",
    "valid_grid",
)


PAIRWISE_FIELDS = ("same_object_choice", "notes")


def load_generation_manifest(
    path: Path, *, arm: str = "lora", seeds: list[int] | None = None
) -> list[dict[str, Any]]:
    """Base and one adapter arm from a (possibly multi-arm) generation manifest."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    conditions = tuple(dict.fromkeys(str(record["condition"]) for record in records))
    validate_paired_records(records, conditions)
    if arm not in conditions:
        raise ValueError(f"Arm {arm!r} is not in {path}; available: {list(conditions)}")
    selected = [r for r in records if r["condition"] in ("base", arm)]
    if seeds:
        selected = [r for r in selected if int(r["seed"]) in set(seeds)]
    return selected


def resolve_output_path(record: dict[str, Any], manifest_path: Path) -> Path:
    path = Path(str(record["output_path"]))
    return path if path.is_absolute() else manifest_path.parent / path


def create_scorecard(
    records: list[dict[str, Any]],
    *,
    generation_manifest: Path,
    output_dir: Path,
    blind_seed: int = 17,
    arm: str = "lora",
) -> tuple[Path, Path]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        key = (str(record["prompt_id"]), int(record["seed"]))
        grouped.setdefault(key, {})[str(record["condition"])] = record

    rng = random.Random(blind_seed)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    pairs = sorted(grouped.items())
    rng.shuffle(pairs)  # raters should not see the seeds of one prompt back to back
    for pair_number, (key, conditions) in enumerate(pairs, 1):
        condition_order = ["base", arm]
        rng.shuffle(condition_order)
        prompt_id, seed = key
        pair_id = f"pair-{pair_number:03d}"
        pairwise_rows.append(
            {
                "pair_id": pair_id,
                "image_A": f"images/{pair_id}_A.png",
                "image_B": f"images/{pair_id}_B.png",
                "prompt": conditions["base"]["prompt"],
                **{field: "" for field in PAIRWISE_FIELDS},
            }
        )
        for blind_label, condition in zip(("A", "B"), condition_order, strict=True):
            record = conditions[condition]
            source = resolve_output_path(record, generation_manifest)
            if not source.is_file():
                raise FileNotFoundError(f"Generated output does not exist: {source}")
            blind_name = f"{pair_id}_{blind_label}.png"
            blind_path = image_dir / blind_name
            shutil.copy2(source, blind_path)
            score_rows.append(
                {
                    "pair_id": pair_id,
                    "prompt_id": prompt_id,
                    "prompt_group": record.get("prompt_group", ""),
                    "seed": seed,
                    "blind_label": blind_label,
                    "image_path": blind_path.relative_to(output_dir).as_posix(),
                    "prompt": record["prompt"],
                    **{field: "" for field in SCORE_FIELDS},
                }
            )
            key_rows.append(
                {
                    "blind_label": blind_label,
                    "condition": condition,
                    "original_output_path": record["output_path"],
                    "pair_id": pair_id,
                    "prompt_id": prompt_id,
                    "prompt_group": record.get("prompt_group", ""),
                    "seed": seed,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.csv"
    fieldnames = [
        "pair_id",
        "prompt_id",
        "prompt_group",
        "seed",
        "blind_label",
        "image_path",
        "prompt",
        *SCORE_FIELDS,
    ]
    with scorecard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(score_rows)

    with (output_dir / "pairwise.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pair_id", "image_A", "image_B", "prompt", *PAIRWISE_FIELDS]
        )
        writer.writeheader()
        writer.writerows(pairwise_rows)

    key_path = output_dir / "blind_key.jsonl"
    key_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in key_rows),
        encoding="utf-8",
    )
    return scorecard_path, key_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-manifest",
        type=Path,
        default=Path("evaluation/results/generation_manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/blinded"))
    parser.add_argument("--blind-seed", type=int, default=17)
    parser.add_argument("--arm", default="lora", help="adapter arm to compare with base")
    parser.add_argument("--seeds", nargs="+", type=int, help="only these generation seeds")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = load_generation_manifest(args.generation_manifest, arm=args.arm, seeds=args.seeds)
    scorecard, key = create_scorecard(
        records,
        generation_manifest=args.generation_manifest,
        output_dir=args.output_dir,
        blind_seed=args.blind_seed,
        arm=args.arm,
    )
    print(f"Wrote blinded scorecard: {scorecard}")
    print(f"Keep the condition key separate from raters: {key}")


if __name__ == "__main__":
    main()
