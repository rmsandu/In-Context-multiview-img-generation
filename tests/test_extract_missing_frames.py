import json
import zipfile

import pytest

from src.extract_missing_frames import FRAME, empty_objects, extract, main, plan


def test_frame_pattern_only_matches_jpeg_frames():
    assert FRAME.fullmatch("1/080182b5/images/001.jpg")
    for name in (
        "1/080182b5/images/001_bg_removed.png",
        "1/080182b5/sparse/0/cameras.bin",
        "1/080182b5/images/",
        "../1/080182b5/images/001.jpg",
        "1/../images/001.jpg",
    ):
        assert not FRAME.fullmatch(name)


def test_plan_keeps_only_wanted_objects_sorted():
    names = ["1/b/images/002.jpg", "1/b/images/001.jpg", "1/a/images/001.jpg", "1/b/x.ini"]
    assert plan(names, {"1/b"}) == {"1/b": ["1/b/images/001.jpg", "1/b/images/002.jpg"]}


def _setup(tmp_path):
    data = tmp_path / "data" / "mvi_00"
    (data / "1" / "full" / "images").mkdir(parents=True)
    (data / "1" / "full" / "images" / "001.jpg").write_bytes(b"local")
    (data / "1" / "empty" / "images").mkdir(parents=True)
    archive = tmp_path / "mvi_00.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("1/full/images/001.jpg", b"zip-full")
        z.writestr("1/empty/images/001.jpg", b"a")
        z.writestr("1/empty/images/002.jpg", b"b")
        z.writestr("1/empty/images/001_bg_removed.png", b"png")
        z.writestr("1/empty/sparse/0/cameras.bin", b"bin")
    return tmp_path / "data", archive


def test_extracts_only_frames_of_empty_objects(tmp_path):
    data, archive = _setup(tmp_path)
    assert empty_objects(data, "mvi_00") == {"1/empty"}
    out = tmp_path / "out"
    main(["--zip", f"mvi_00={archive}", "--objects-dir", str(data), "--output-dir", str(out)])

    frames = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.jpg"))
    assert frames == ["mvi_00/1/empty/images/001.jpg", "mvi_00/1/empty/images/002.jpg"]
    assert not list(out.rglob("*.png")) and not list(out.rglob("*.bin"))
    rows = [json.loads(line) for line in (out / "extraction_manifest.jsonl").open()]
    assert [r["path"] for r in rows] == frames
    assert (data / "mvi_00" / "1" / "full" / "images" / "001.jpg").read_bytes() == b"local"


def test_refuses_to_overwrite_different_content(tmp_path):
    data, archive = _setup(tmp_path)
    out = tmp_path / "out"
    target = out / "mvi_00" / "1" / "empty" / "images"
    target.mkdir(parents=True)
    (target / "001.jpg").write_bytes(b"other")
    with pytest.raises(FileExistsError):
        extract(archive, "mvi_00", {"1/empty"}, out)


def test_fails_when_an_object_is_not_in_the_archive(tmp_path):
    data, archive = _setup(tmp_path)
    with pytest.raises(ValueError, match="no frames"):
        extract(archive, "mvi_00", {"1/empty", "1/absent"}, tmp_path / "out")
