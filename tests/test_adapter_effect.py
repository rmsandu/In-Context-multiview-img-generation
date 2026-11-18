import numpy as np
from PIL import Image

from analysis.adapter_effect import cosine_rows, soft_bar_fraction, tiles


def _grid(tmp_path, bar_px: int):
    """2x2 grid of 64 px grey tiles; each tile gets black side bars of ``bar_px``."""
    tile = np.full((64, 64), 128, dtype=np.uint8)
    if bar_px:
        tile[:, :bar_px] = 0
        tile[:, -bar_px:] = 0
    grid = np.block([[tile, tile], [tile, tile]])
    path = tmp_path / f"grid_{bar_px}.png"
    Image.fromarray(grid).save(path)
    return path


def test_tiles_split_grid_in_reading_order(tmp_path):
    grid = np.zeros((8, 8), dtype=np.uint8)
    grid[:4, 4:] = 1
    grid[4:, :4] = 2
    grid[4:, 4:] = 3
    path = tmp_path / "g.png"
    Image.fromarray(grid).save(path)
    means = [np.asarray(t.convert("L")).mean() for t in tiles(path)]
    assert means == [0, 1, 2, 3]


def test_soft_bar_fraction_measures_side_bars(tmp_path):
    assert soft_bar_fraction(_grid(tmp_path, 0)) == 0.0
    assert abs(soft_bar_fraction(_grid(tmp_path, 14)) - 28 / 64) < 0.02


def test_cosine_rows():
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [-1.0, -1.0]])
    np.testing.assert_allclose(cosine_rows(a, b), [1.0, -1.0], atol=1e-9)
