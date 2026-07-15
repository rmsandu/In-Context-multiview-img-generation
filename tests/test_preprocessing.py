from pathlib import Path

import pytest
from PIL import Image

from src.composite_img import make_composite_grid
from src.dataset_builder import choose_four_views


def _image(path: Path, size: tuple[int, int], color: tuple[int, int, int], mode="RGB"):
    Image.new(mode, size, color if mode == "RGB" else color[0]).save(path)
    return path


def test_choose_four_views_is_deterministic():
    paths = [Path(f"{index:02}.jpg") for index in range(8)]
    assert choose_four_views(reversed(paths)) == [paths[0], paths[2], paths[4], paths[6]]


def test_choose_four_views_duplicates_short_sweeps():
    paths = [Path("a.jpg"), Path("b.jpg")]
    assert choose_four_views(paths) == [paths[0], paths[0], paths[1], paths[0]]


def test_choose_four_views_rejects_empty_input():
    with pytest.raises(ValueError, match="empty image set"):
        choose_four_views([])


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


def test_composite_grid_requires_four_images(tmp_path):
    path = _image(tmp_path / "one.png", (4, 4), (255, 0, 0))
    with pytest.raises(ValueError, match="exactly four"):
        make_composite_grid([path])
