import json
from pathlib import Path

import pytest
from PIL import Image

from src import dataset_builder
from src.composite_img import make_composite_grid
from src.dataset_builder import (
    CAPTION_SCHEMA_VERSION,
    InvalidInstanceError,
    choose_four_views,
    hash_selected_views,
)


def _image(path: Path, size: tuple[int, int], color: tuple[int, int, int], mode="RGB"):
    Image.new(mode, size, color if mode == "RGB" else color[0]).save(path)
    return path


@pytest.mark.parametrize(
    ("count", "expected_indices"),
    [
        (4, [0, 1, 2, 3]),
        (5, [0, 1, 2, 3]),
        (8, [0, 2, 4, 6]),
        (20, [0, 5, 10, 15]),
    ],
)
def test_choose_four_views_is_deterministic(count, expected_indices):
    paths = [Path(f"{index:02}.jpg") for index in range(count)]
    assert choose_four_views(reversed(paths)) == [paths[index] for index in expected_indices]


@pytest.mark.parametrize("count", range(4))
def test_choose_four_views_rejects_short_sequences(count):
    paths = [Path(f"{index:02}.jpg") for index in range(count)]
    with pytest.raises(InvalidInstanceError, match="At least four"):
        choose_four_views(paths)


def test_choose_four_views_rejects_resolved_path_aliases(tmp_path):
    original = _image(tmp_path / "00.jpg", (4, 4), (255, 0, 0))
    alias = tmp_path / "01.jpg"
    alias.symlink_to(original)
    paths = [original, alias]
    paths.extend(
        [
            _image(tmp_path / "02.jpg", (4, 4), (0, 255, 0)),
            _image(tmp_path / "03.jpg", (4, 4), (0, 0, 255)),
        ]
    )

    with pytest.raises(InvalidInstanceError, match="distinct resolved paths"):
        choose_four_views(paths)


def test_hash_selected_views_rejects_exact_duplicates(tmp_path):
    paths = [
        _image(tmp_path / f"{index:02}.png", (4, 4), (index * 30, 0, 0))
        for index in range(4)
    ]
    paths[-1].write_bytes(paths[0].read_bytes())

    with pytest.raises(InvalidInstanceError, match="duplicate file contents"):
        hash_selected_views(paths)


def test_duplicate_unselected_file_does_not_invalidate_selection(tmp_path):
    paths = [
        _image(tmp_path / f"{index:02}.png", (4, 4), (index * 20, 0, 0))
        for index in range(8)
    ]
    paths[-1].write_bytes(paths[0].read_bytes())

    selected = choose_four_views(paths)

    assert selected == [paths[index] for index in (0, 2, 4, 6)]
    assert len(set(hash_selected_views(selected))) == 4


def test_composite_grid_preserves_order_and_converts_rgb(tmp_path):
    paths = [
        _image(tmp_path / "red.png", (4, 4), (255, 0, 0)),
        _image(tmp_path / "green.png", (4, 4), (0, 255, 0)),
        _image(tmp_path / "blue.png", (4, 4), (0, 0, 255)),
        _image(tmp_path / "gray.png", (4, 4), (128, 0, 0), mode="L"),
    ]
    grid = make_composite_grid(paths, target_w=4, target_h=4)

    assert grid.mode == "RGB"
    assert grid.size == (8, 8)
    assert grid.getpixel((1, 1)) == (255, 0, 0)
    assert grid.getpixel((5, 1)) == (0, 255, 0)
    assert grid.getpixel((1, 5)) == (0, 0, 255)
    assert grid.getpixel((5, 5)) == (128, 128, 128)


def test_composite_grid_resizes_large_images_and_centers_small_ones(tmp_path):
    paths = [
        _image(tmp_path / "large.png", (8, 8), (255, 0, 0)),
        _image(tmp_path / "small.png", (2, 2), (0, 255, 0)),
        _image(tmp_path / "third.png", (4, 4), (0, 0, 255)),
        _image(tmp_path / "fourth.png", (4, 4), (255, 255, 255)),
    ]
    grid = make_composite_grid(paths, target_w=4, target_h=4)

    assert grid.getpixel((1, 1)) == (255, 0, 0)
    assert grid.getpixel((4, 0)) == (0, 0, 0)
    assert grid.getpixel((5, 1)) == (0, 255, 0)


def test_composite_grid_preserves_aspect_ratio_when_resizing(tmp_path):
    paths = [
        _image(tmp_path / f"{index}.png", (8, 4), (255, 0, 0))
        for index in range(4)
    ]

    grid = make_composite_grid(paths, target_w=4, target_h=4)

    assert grid.getpixel((1, 0)) == (0, 0, 0)
    assert grid.getpixel((1, 1)) == (255, 0, 0)
    assert grid.getpixel((1, 2)) == (255, 0, 0)
    assert grid.getpixel((1, 3)) == (0, 0, 0)


def test_composite_grid_requires_four_images(tmp_path):
    path = _image(tmp_path / "one.png", (4, 4), (255, 0, 0))
    with pytest.raises(ValueError, match="exactly four"):
        make_composite_grid([path])


def _instance(root: Path, obj_id: str, instance_id: str, count: int) -> Path:
    images_dir = root / obj_id / instance_id / "images"
    images_dir.mkdir(parents=True)
    for index in range(count):
        _image(images_dir / f"{index:02}.jpg", (4, 4), (index * 40, 10, 20))
    return images_dir


def test_builder_writes_manifest_and_usage_summary(tmp_path, monkeypatch, capsys):
    objects_dir = tmp_path / "objects"
    _instance(objects_dir, "0", "valid", 5)
    _instance(objects_dir, "1", "short", 3)
    category_file = tmp_path / "categories.txt"
    category_file.write_text("0,bag\n1,bottle\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        dataset_builder,
        "generate_caption_composite_grid",
        lambda image, category: f"[FOUR-VIEWS] Four views of the same {category}.",
    )

    result = dataset_builder.main(
        [
            "--objects-dir",
            str(objects_dir),
            "--category-file",
            str(category_file),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
            "--tile-width",
            "4",
            "--tile-height",
            "4",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Datapoints used: 1" in output
    assert "Images used: 4" in output
    assert "Invalid instances skipped: 1" in output
    assert "Unexpected processing failures: 0" in output

    manifest_lines = (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 1
    record = json.loads(manifest_lines[0])
    assert record["instance"] == "0/valid"
    assert [view["position"] for view in record["views"]] == [
        "TOP-LEFT",
        "TOP-RIGHT",
        "BOTTOM-LEFT",
        "BOTTOM-RIGHT",
    ]
    assert [view["filename"] for view in record["views"]] == [
        "00.jpg",
        "01.jpg",
        "02.jpg",
        "03.jpg",
    ]
    assert all(len(view["sha256"]) == 64 for view in record["views"])
    assert all(CAPTION_SCHEMA_VERSION in path.name for path in cache_dir.iterdir())


def test_manifest_is_sorted_and_rewritten(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    records = [{"instance": "b/item"}, {"instance": "a/item"}]

    dataset_builder.write_manifest(output_dir, records)

    manifest = output_dir / "manifest.jsonl"
    assert [json.loads(line)["instance"] for line in manifest.read_text().splitlines()] == [
        "a/item",
        "b/item",
    ]

    dataset_builder.write_manifest(output_dir, [{"instance": "c/item"}])

    assert [json.loads(line)["instance"] for line in manifest.read_text().splitlines()] == [
        "c/item"
    ]


def test_builder_returns_nonzero_for_unexpected_failures(tmp_path, monkeypatch, capsys):
    objects_dir = tmp_path / "objects"
    image_dir = _instance(objects_dir, "0", "broken", 4)
    category_file = tmp_path / "categories.txt"
    category_file.write_text("0,bag\n", encoding="utf-8")
    monkeypatch.setattr(dataset_builder, "find_image_dirs", lambda root: [image_dir])

    def fail_processing(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(dataset_builder, "process_one", fail_processing)

    result = dataset_builder.main(
        [
            "--objects-dir",
            str(objects_dir),
            "--category-file",
            str(category_file),
            "--output-dir",
            str(tmp_path / "output"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    assert result == 1
    assert "Unexpected processing failures: 1" in capsys.readouterr().out
