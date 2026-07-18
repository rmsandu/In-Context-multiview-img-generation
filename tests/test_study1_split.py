import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.study1_split import (
    Pair,
    load_pairs,
    materialize_split,
    split_pairs,
    validate_partition,
)


def _source_pair(root: Path, stem: str, instance: str, image: bytes) -> dict[str, object]:
    (root / f"{stem}.png").write_bytes(image)
    (root / f"{stem}.txt").write_text(f"caption {stem}", encoding="utf-8")
    return {
        "instance": instance,
        "output_image": f"{stem}.png",
        "output_caption": f"{stem}.txt",
        "views": [{"sha256": f"source-{stem}"}],
    }


def _write_manifest(root: Path, records: list[dict[str, object]]) -> None:
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_split_is_deterministic_and_keeps_instances_together(tmp_path: Path) -> None:
    records = []
    for index in range(10):
        records.append(_source_pair(tmp_path, f"item-{index}", f"instance-{index}", bytes([index])))
    records.append(_source_pair(tmp_path, "item-10", "instance-0", b"extra"))
    _write_manifest(tmp_path, records)
    pairs = load_pairs(tmp_path, expected_count=11)

    train_a, holdout_a = split_pairs(pairs, seed=17, holdout_fraction=0.2)
    train_b, holdout_b = split_pairs(pairs, seed=17, holdout_fraction=0.2)

    assert [pair.stem for pair in train_a] == [pair.stem for pair in train_b]
    assert [pair.stem for pair in holdout_a] == [pair.stem for pair in holdout_b]
    assert {pair.instance for pair in train_a}.isdisjoint({pair.instance for pair in holdout_a})
    assert sum(pair.instance == "instance-0" for pair in train_a + holdout_a) == 2


def test_load_pairs_rejects_unmatched_stems(tmp_path: Path) -> None:
    record = _source_pair(tmp_path, "matched", "one", b"png")
    (tmp_path / "orphan.png").write_bytes(b"orphan")
    _write_manifest(tmp_path, [record])

    with pytest.raises(ValueError, match="PNG/TXT stems do not match"):
        load_pairs(tmp_path)


def test_partition_rejects_cross_split_image_hash_overlap(tmp_path: Path) -> None:
    pair = Pair("a", "one", tmp_path / "a.png", tmp_path / "a.txt", "same", ())
    duplicate = replace(pair, stem="b", instance="two")

    with pytest.raises(ValueError, match="Composite image SHA-256 overlap"):
        validate_partition([pair], [duplicate], expected_total=2)


def test_materialized_pairs_and_manifest_match(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    records = [
        _source_pair(source, f"item-{index}", f"instance-{index}", bytes([index]))
        for index in range(4)
    ]
    _write_manifest(source, records)
    pairs = load_pairs(source, expected_count=4)
    train, holdout = split_pairs(pairs, seed=17, holdout_fraction=0.25)

    manifest = materialize_split(train, holdout, tmp_path / "pilot")
    manifest_records = [json.loads(line) for line in manifest.read_text().splitlines()]

    assert len(list((tmp_path / "pilot" / "train").glob("*.png"))) == 3
    assert len(list((tmp_path / "pilot" / "holdout").glob("*.png"))) == 1
    assert len(manifest_records) == 4
    assert {record["split"] for record in manifest_records} == {"train", "holdout"}
