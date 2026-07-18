"""Create randomized, blinded Study 1 image copies and a scoring CSV."""

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
    "duplicate_pairs_0_6",
    "prompt_fidelity_1_5",
    "valid_grid",
)


def load_generation_manifest(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    validate_paired_records(records)
    return records


def resolve_output_path(record: dict[str, Any], manifest_path: Path) -> Path:
    path = Path(str(record["output_path"]))
    return path if path.is_absolute() else manifest_path.parent / path


def create_scorecard(
    records: list[dict[str, Any]],
    *,
    generation_manifest: Path,
    output_dir: Path,
    blind_seed: int = 17,
) -> tuple[Path, Path]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        key = (str(record["prompt_id"]), int(record["seed"]))
        grouped.setdefault(key, {})[str(record["condition"])] = record

    rng = random.Random(blind_seed)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for pair_number, (key, conditions) in enumerate(sorted(grouped.items()), 1):
        condition_order = ["base", "lora"]
        rng.shuffle(condition_order)
        prompt_id, seed = key
        pair_id = f"pair-{pair_number:03d}"
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
                    "seed": seed,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.csv"
    fieldnames = [
        "pair_id",
        "prompt_id",
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
        default=Path("evaluation/outputs/study1_pilot/generation_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/outputs/study1_pilot_blinded")
    )
    parser.add_argument("--blind-seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = load_generation_manifest(args.generation_manifest)
    scorecard, key = create_scorecard(
        records,
        generation_manifest=args.generation_manifest,
        output_dir=args.output_dir,
        blind_seed=args.blind_seed,
    )
    print(f"Wrote blinded scorecard: {scorecard}")
    print(f"Keep the condition key separate from raters: {key}")


if __name__ == "__main__":
    main()
