import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.composite_img import make_composite_grid, object_square_box
from src.crop_composites import cut_fraction, scale_box


@pytest.mark.parametrize(
    ("size", "object_box", "expected"),
    [
        ((9, 16), (0, 2, 9, 6), (0, 0, 9, 9)),  # object near the top: clamp to the top
        ((9, 16), (0, 10, 9, 16), (0, 7, 9, 16)),  # near the bottom: clamp to the bottom
        ((9, 16), (2, 5, 7, 9), (0, 3, 9, 12)),  # centred on the object's middle (y = 7)
        ((16, 9), (10, 0, 14, 9), (7, 0, 16, 9)),  # landscape slides horizontally
        ((9, 16), None, (0, 4, 9, 13)),  # no object: centre of the frame
        ((8, 8), (1, 1, 3, 3), (0, 0, 8, 8)),  # square frame is kept whole
    ],
)
def test_object_square_box(size, object_box, expected):
    assert object_square_box(size, object_box) == expected


def test_object_square_box_rejects_boxes_outside_the_frame():
    with pytest.raises(ValueError, match="outside"):
        object_square_box((9, 16), (0, 10, 9, 17))


def test_tall_object_is_cut_equally_at_both_ends():
    box = object_square_box((9, 16), (0, 1, 9, 14))
    assert box == (0, 3, 9, 12)
    assert cut_fraction((0, 1, 9, 14), box) == pytest.approx(4 / 13)


def test_cut_fraction_is_zero_inside_and_without_object():
    assert cut_fraction((1, 4, 8, 10), (0, 3, 9, 12)) == 0.0
    assert cut_fraction(None, (0, 3, 9, 12)) == 0.0


def test_scale_box_rounds_outwards_and_clamps():
    assert scale_box((1, 1, 3, 3), 1.5, 1.5, (4, 4)) == (1, 1, 4, 4)


BAND = (220, 180, 140)  # bright and free of zero channels, so "no black" is checkable


def _frame(path: Path, size, rows: tuple[int, int], color=BAND) -> Path:
    """Dark frame with a bright full-width band over ``rows`` (the "object")."""
    image = Image.new("RGB", size, (10, 10, 10))
    image.paste(Image.new("RGB", (size[0], rows[1] - rows[0]), color), (0, rows[0]))
    image.save(path)
    return path


def test_crop_boxes_fill_the_tile_without_bars(tmp_path):
    paths = [_frame(tmp_path / f"{i}.png", (9, 16), (2, 6)) for i in range(4)]
    boxes = [object_square_box((9, 16), (0, 2, 9, 6))] * 4
    grid = make_composite_grid(paths, target_h=9, target_w=9, crop_boxes=boxes)
    assert grid.size == (18, 18)
    tile = np.asarray(grid)[:9, :9]
    assert tile[3, 4].tolist() == list(BAND)  # the band survives at the same height
    assert tile.min() == 10  # no black padding anywhere


def test_crop_boxes_must_match_the_tile_ratio(tmp_path):
    paths = [_frame(tmp_path / f"{i}.png", (9, 16), (2, 6)) for i in range(4)]
    with pytest.raises(ValueError, match="tile ratio"):
        make_composite_grid(paths, target_h=9, target_w=9, crop_boxes=[(0, 0, 9, 16)] * 4)


def test_letterbox_default_is_unchanged(tmp_path):
    paths = [_frame(tmp_path / f"{i}.png", (8, 16), (0, 16)) for i in range(4)]
    tile = np.asarray(make_composite_grid(paths, target_h=4, target_w=4))[:4, :4]
    assert tile[:, 0].tolist() == [[0, 0, 0]] * 4  # black bar on the left
    assert tile[:, 1].tolist() == [list(BAND)] * 4


# ---------------------------------------------------------------- end to end (scipy)


def _bright_masker(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L")) > 50


def _source_split(tmp_path: Path, rows_by_stem: dict[str, tuple[int, int]]):
    objects = tmp_path / "data"
    split = tmp_path / "split"
    (split / "train").mkdir(parents=True)
    manifest, split_rows = [], []
    for stem, rows in rows_by_stem.items():
        frame_dir = objects / "mvi_00" / "1" / stem / "images"
        frame_dir.mkdir(parents=True)
        views = []
        for index, position in enumerate(("TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT")):
            path = _frame(frame_dir / f"{index:03d}.jpg", (18, 32), rows)
            views.append(
                {
                    "position": position,
                    "path": path.relative_to(objects).as_posix(),
                    "sha256": _sha(path),
                }
            )
        frames = [objects / v["path"] for v in views]
        make_composite_grid(frames, target_h=8, target_w=8).save(split / "train" / f"{stem}.png")
        (split / "train" / f"{stem}.txt").write_text(f"caption for {stem}")
        manifest.append({"output_image": f"{stem}.png", "views": views})
        split_rows.append(
            {
                "caption": f"train/{stem}.txt",
                "holdout_fraction": 0.1,
                "image": f"train/{stem}.png",
                "image_sha256": _sha(split / "train" / f"{stem}.png"),
                "seed": 17,
                "source_image_sha256": [v["sha256"] for v in views],
                "source_instance": f"mvi_00/1/{stem}",
                "split": "train",
                "stem": stem,
            }
        )
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (split / "split_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in split_rows))
    return objects, split


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(tmp_path, objects, split, out):
    from src.crop_composites import build

    return build(
        source_split=split,
        source_manifest=tmp_path / "manifest.jsonl",
        objects_dir=objects,
        output_dir=out,
        masker=_bright_masker,
        masker_info={"model": "test"},
        tile_size=8,
    )


def test_build_crops_around_the_object_and_keeps_captions(tmp_path):
    pytest.importorskip("scipy")
    objects, split = _source_split(tmp_path, {"high": (2, 8), "low": (24, 30)})
    out = tmp_path / "out"
    info = _build(tmp_path, objects, split, out)

    crops = {r["stem"]: r for r in map(json.loads, (out / "crop_manifest.jsonl").open())}
    assert crops["high"]["views"][0]["crop_box"] == [0, 0, 18, 18]
    assert crops["low"]["views"][0]["crop_box"] == [0, 14, 18, 32]
    assert all(v["object_inside"] for r in crops.values() for v in r["views"])
    assert (out / "train" / "high.txt").read_text() == "caption for high"

    rows = [json.loads(line) for line in (out / "split_manifest.jsonl").open()]
    source = [json.loads(line) for line in (split / "split_manifest.jsonl").open()]
    assert [r["stem"] for r in rows] == sorted(r["stem"] for r in source)
    assert all(r["image_sha256"] == _sha(out / r["image"]) for r in rows)
    assert info["summary"]["composites"] == 2
    assert info["summary"]["mask_fallback_views"] == 0


def test_build_refuses_non_empty_output(tmp_path):
    pytest.importorskip("scipy")
    objects, split = _source_split(tmp_path, {"a": (2, 8)})
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("")
    with pytest.raises(FileExistsError):
        _build(tmp_path, objects, split, out)


def test_build_rejects_frames_that_do_not_rebuild_the_trained_composite(tmp_path):
    pytest.importorskip("scipy")
    objects, split = _source_split(tmp_path, {"a": (2, 8)})
    Image.new("RGB", (16, 16), (1, 2, 3)).save(split / "train" / "a.png")
    with pytest.raises(ValueError, match="letterboxed rebuild differs"):
        _build(tmp_path, objects, split, tmp_path / "out")


def test_largest_component_ignores_small_specks():
    pytest.importorskip("scipy")
    from src.crop_composites import largest_component_box

    mask = np.zeros((20, 10), dtype=bool)
    mask[2:8, 1:9] = True  # the object
    mask[15, 5] = True  # a speck
    box, share = largest_component_box(mask)
    assert box == (1, 2, 9, 8)
    assert share == pytest.approx(48 / 200)
    assert largest_component_box(np.zeros((20, 10), dtype=bool)) == (None, 0.0)
