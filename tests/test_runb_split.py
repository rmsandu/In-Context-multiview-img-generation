import hashlib
import json
from pathlib import Path

import pytest

from src.runb_split import build_new_split, combine


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _source(tmp_path: Path, objects: dict[str, str]) -> Path:
    """Letterboxed source folder: stem -> instance, four frame hashes derived from the stem."""
    source = tmp_path / "source"
    source.mkdir()
    records = []
    for stem, instance in objects.items():
        (source / f"{stem}.png").write_bytes(f"png {stem}".encode())
        (source / f"{stem}.txt").write_text(f"caption {stem}")
        records.append(
            {
                "instance": instance,
                "output_image": f"{stem}.png",
                "output_caption": f"{stem}.txt",
                "views": [{"sha256": _sha(f"{stem} frame {i}".encode())} for i in range(4)],
            }
        )
    _write_jsonl(source / "manifest.jsonl", records)
    return source


def _split_row(stem, instance, split, image_sha="x", frames=None):
    return {
        "stem": stem,
        "source_instance": instance,
        "split": split,
        "image": f"{split}/{stem}.png",
        "caption": f"{split}/{stem}.txt",
        "image_sha256": image_sha,
        "source_image_sha256": frames or [_sha(f"{stem} frame {i}".encode()) for i in range(4)],
        "seed": 17,
        "holdout_fraction": 0.1,
    }


STUDY1 = [
    _split_row("Old_1_a", "mvi_32/1/a", "train"),
    _split_row("Held_2_b", "mvi_32/2/b", "holdout"),
]


def test_new_split_is_train_only_and_records_no_random_seed(tmp_path):
    source = _source(tmp_path, {"New_3_c": "mvi_40/3/c", "New_4_d": "mvi_40/4/d"})
    _write_jsonl(tmp_path / "study1.jsonl", STUDY1)
    manifest = build_new_split(source, tmp_path / "study1.jsonl", tmp_path / "out", expected_count=2)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert [r["split"] for r in rows] == ["train", "train"]
    assert all(r["seed"] is None and r["holdout_fraction"] is None for r in rows)
    assert sorted(p.name for p in (tmp_path / "out/train").iterdir()) == [
        "New_3_c.png",
        "New_3_c.txt",
        "New_4_d.png",
        "New_4_d.txt",
    ]
    assert not (tmp_path / "out/holdout").exists()
    with pytest.raises(FileExistsError):
        build_new_split(source, tmp_path / "study1.jsonl", tmp_path / "out", expected_count=2)


@pytest.mark.parametrize(
    "study1, message",
    [
        # a holdout frame reappears in a new composite
        (
            [_split_row("Held_2_b", "mvi_32/2/b", "holdout", frames=[_sha(b"New_3_c frame 0")])],
            "Source image SHA-256 overlap",
        ),
        ([_split_row("Held_2_b", "mvi_40/3/c", "holdout")], "Source-instance overlap"),
        (
            [_split_row("Old_9_z", "mvi_40/3/c", "train"), STUDY1[1]],
            "repeat Study 1 objects",
        ),
        ([_split_row("New_3_c", "mvi_32/7/q", "train"), STUDY1[1]], "clash"),
    ],
)
def test_new_split_refuses_overlap_with_study1(tmp_path, study1, message):
    source = _source(tmp_path, {"New_3_c": "mvi_40/3/c"})
    _write_jsonl(tmp_path / "study1.jsonl", study1)
    with pytest.raises(ValueError, match=message):
        build_new_split(source, tmp_path / "study1.jsonl", tmp_path / "out", expected_count=1)
    assert not (tmp_path / "out").exists()


def _cropped_part(root: Path, stems: dict[str, str]) -> list[dict]:
    rows = []
    for stem, instance in stems.items():
        data = f"crop {stem}".encode()
        (root / "train").mkdir(parents=True, exist_ok=True)
        (root / "train" / f"{stem}.png").write_bytes(data)
        (root / "train" / f"{stem}.txt").write_text(f"caption {stem}")
        rows.append(_split_row(stem, instance, "train", image_sha=_sha(data)))
    return rows


def test_combine_checks_run_a_against_its_record(tmp_path):
    runa = tmp_path / "runa"
    runa_rows = _cropped_part(runa, {"Old_1_a": "mvi_32/1/a"}) + [STUDY1[1]]
    _write_jsonl(runa / "split_manifest.jsonl", runa_rows)
    new = tmp_path / "new"
    _write_jsonl(new / "split_manifest.jsonl", _cropped_part(new, {"New_3_c": "mvi_40/3/c"}))
    _write_jsonl(tmp_path / "record.jsonl", runa_rows)
    _write_jsonl(tmp_path / "study1.jsonl", STUDY1)

    info = combine(
        [runa, new],
        tmp_path / "out",
        runa_record=tmp_path / "record.jsonl",
        study1_split=tmp_path / "study1.jsonl",
    )
    assert info["train"] == 2
    rows = [json.loads(line) for line in (tmp_path / "out/split_manifest.jsonl").read_text().splitlines()]
    assert [(r["stem"], r["part"]) for r in rows] == [
        ("New_3_c", new.as_posix()),
        ("Old_1_a", runa.as_posix()),
    ]
    assert (tmp_path / "out/train/Old_1_a.txt").read_text() == "caption Old_1_a"

    (runa / "train/Old_1_a.png").write_bytes(b"changed")
    _write_jsonl(
        tmp_path / "record_changed.jsonl",
        [dict(runa_rows[0], image_sha256="different"), STUDY1[1]],
    )
    with pytest.raises(ValueError, match="does not match the run A build record"):
        combine(
            [runa, new],
            tmp_path / "out2",
            runa_record=tmp_path / "record_changed.jsonl",
            study1_split=tmp_path / "study1.jsonl",
        )


def test_combine_refuses_an_object_in_two_parts(tmp_path):
    runa = tmp_path / "runa"
    runa_rows = _cropped_part(runa, {"Old_1_a": "mvi_32/1/a"})
    _write_jsonl(runa / "split_manifest.jsonl", runa_rows)
    new = tmp_path / "new"
    _write_jsonl(new / "split_manifest.jsonl", _cropped_part(new, {"Other_1_a": "mvi_32/1/a"}))
    _write_jsonl(tmp_path / "record.jsonl", runa_rows)
    _write_jsonl(tmp_path / "study1.jsonl", STUDY1)
    with pytest.raises(ValueError, match="more than one part"):
        combine(
            [runa, new],
            tmp_path / "out",
            runa_record=tmp_path / "record.jsonl",
            study1_split=tmp_path / "study1.jsonl",
        )
