"""Grid geometry and viewpoint bookkeeping.

Everything here is deterministic and model-free: splitting a grid into tiles,
reading the per-tile view labels out of a four-view caption, and turning
pairwise relative azimuths into "how many distinct views does this grid show".
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from PIL import Image

TILE_POSITIONS = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT")
TILE_PAIRS: tuple[tuple[int, int], ...] = tuple(itertools.combinations(range(4), 2))

# Camera position around the object, in degrees, counter-clockwise seen from above,
# with the object's front at 0. Only the *differences* between tiles are used, and
# the sign convention of the pose model is calibrated on real data (see protocol).
HORIZONTAL_AZIMUTH = {
    "front": 0.0,
    "front-right three-quarter": 45.0,
    "right side": 90.0,
    "rear-right three-quarter": 135.0,
    "back": 180.0,
    "rear": 180.0,
    "rear-left three-quarter": 225.0,
    "left side": 270.0,
    "front-left three-quarter": 315.0,
}
# Labels that describe a view but carry no usable azimuth.
NON_AZIMUTH_LABELS = {"top", "bottom", "indeterminate"}

_TILE_RE = re.compile(r"\[(TOP-LEFT|TOP-RIGHT|BOTTOM-LEFT|BOTTOM-RIGHT)\]\s*([^;\[]*)")


@dataclass(frozen=True)
class TileView:
    position: str
    label: str | None
    azimuth: float | None
    elevation: str | None


def split_grid(image: Image.Image) -> list[Image.Image]:
    """Return the four quadrants in TOP-LEFT, TOP-RIGHT, BOTTOM-LEFT, BOTTOM-RIGHT order."""
    width, height = image.size
    if width % 2 or height % 2:
        raise ValueError(f"Grid dimensions must be even, got {width}x{height}")
    half_w, half_h = width // 2, height // 2
    boxes = [
        (0, 0, half_w, half_h),
        (half_w, 0, width, half_h),
        (0, half_h, half_w, height),
        (half_w, half_h, width, height),
    ]
    return [image.crop(box) for box in boxes]


def join_tiles(tiles: list[Image.Image]) -> Image.Image:
    """Inverse of :func:`split_grid` for equally sized tiles."""
    if len(tiles) != 4:
        raise ValueError("Exactly four tiles are required")
    w, h = tiles[0].size
    if any(tile.size != (w, h) for tile in tiles):
        raise ValueError("All tiles must share one size")
    grid = Image.new("RGB", (2 * w, 2 * h))
    for tile, (x, y) in zip(tiles, [(0, 0), (w, 0), (0, h), (w, h)], strict=True):
        grid.paste(tile.convert("RGB"), (x, y))
    return grid


def parse_tile_views(caption: str) -> dict[str, TileView]:
    """Read ``[POSITION] <horizontal view>, <elevation>; ...`` labels from a caption.

    Positions that are missing, or whose label is not a known horizontal view,
    get ``azimuth=None`` so they are excluded from pose scoring rather than guessed.
    """
    found: dict[str, TileView] = {}
    for position, body in _TILE_RE.findall(caption):
        parts = [part.strip().rstrip(".") for part in body.split(",")]
        label = parts[0].lower() if parts and parts[0] else None
        elevation = parts[1].lower() if len(parts) > 1 and parts[1] else None
        azimuth = HORIZONTAL_AZIMUTH.get(label) if label else None
        found[position] = TileView(position, label, azimuth, elevation)
    return {
        position: found.get(position, TileView(position, None, None, None))
        for position in TILE_POSITIONS
    }


def wrap_degrees(angle: float) -> float:
    """Map any angle to the half-open interval (-180, 180]."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def angular_distance(a: float, b: float) -> float:
    return abs(wrap_degrees(a - b))


def expected_relative_azimuth(views: dict[str, TileView], i: int, j: int) -> float | None:
    """Camera azimuth change from tile ``i`` to tile ``j`` implied by the caption."""
    a = views[TILE_POSITIONS[i]].azimuth
    b = views[TILE_POSITIONS[j]].azimuth
    if a is None or b is None:
        return None
    return wrap_degrees(b - a)


def count_distinct_views(same_view: dict[tuple[int, int], bool]) -> int:
    """Number of connected components when 'same view' tile pairs are merged.

    Four tiles that all differ give 4; a grid that repeats one view gives 1.
    Merging is transitive on purpose: A~B and B~C means the grid only shows
    one viewpoint region, even if A and C are slightly further apart.
    """
    parent = list(range(4))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), same in same_view.items():
        if same:
            parent[find(i)] = find(j)
    return len({find(k) for k in range(4)})


def letterbox_fraction(tile: Image.Image, *, threshold: int = 16) -> float:
    """Share of the tile covered by solid near-black bars along its edges.

    The real MVImgNet composites pad portrait and landscape frames to square
    tiles with black bars, so a LoRA trained on them may learn to draw bars.
    Counts whole edge rows/columns whose brightest pixel is below ``threshold``.
    """
    import numpy as np

    grey = np.asarray(tile.convert("L"))
    dark_rows = grey.max(axis=1) < threshold
    dark_cols = grey.max(axis=0) < threshold

    def edge_run(flags) -> int:
        run = 0
        for flag in flags:
            if not flag:
                break
            run += 1
        return run

    h, w = grey.shape
    rows = min(h, edge_run(dark_rows) + edge_run(dark_rows[::-1]))
    cols = min(w, edge_run(dark_cols) + edge_run(dark_cols[::-1]))
    covered = rows * w + cols * h - rows * cols
    return covered / (h * w)
