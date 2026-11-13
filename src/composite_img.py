import math
from collections.abc import Sequence

from PIL import Image

Box = tuple[int, int, int, int]  # left, top, right, bottom in pixels, right/bottom exclusive


def object_square_box(size: tuple[int, int], object_box: Box | None) -> Box:
    """Largest square window, slid along the long side to centre the object.

    The side of the square is the frame's short side, so nothing is cropped across
    the short side and no padding is ever needed. Along the long side the window is
    centred on the middle of ``object_box`` and clamped to the frame. Without an
    object box (no foreground found) the window is centred on the frame.

    An object taller (or wider) than the short side cannot fit; it is then cut
    equally at both ends.
    """
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("Frame dimensions must be positive")
    side = min(width, height)
    if object_box is None:
        centre_x, centre_y = width / 2, height / 2
    else:
        left, top, right, bottom = object_box
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"Object box {object_box} is outside the {width}x{height} frame")
        centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
    # Round half up (not Python's round-half-to-even) so the rule is easy to state.
    x0 = min(max(math.floor(centre_x - side / 2 + 0.5), 0), width - side)
    y0 = min(max(math.floor(centre_y - side / 2 + 0.5), 0), height - side)
    return (x0, y0, x0 + side, y0 + side)


def make_composite_grid(
    img_paths,
    target_h: int = 512,
    target_w: int = 512,
    crop_boxes: Sequence[Box] | None = None,
):
    """
    Create a 2x2 grid composite image from a list of image paths.

    Without ``crop_boxes`` (the Study 1 behaviour) each image is shrunk to fit the
    tile and padded with black bars. With ``crop_boxes`` each image is cropped to
    its box, which must have the tile's aspect ratio, and resized to fill the tile.
    """
    img_paths = list(img_paths)
    if len(img_paths) != 4:
        raise ValueError(f"Expected exactly four images, received {len(img_paths)}")
    if target_h < 1 or target_w < 1:
        raise ValueError("Target dimensions must be positive")
    if crop_boxes is not None and len(crop_boxes) != 4:
        raise ValueError(f"Expected exactly four crop boxes, received {len(crop_boxes)}")

    imgs = []
    for index, p in enumerate(img_paths):
        im = Image.open(p).convert("RGB")
        if crop_boxes is not None:
            im = _crop_to_tile(im, crop_boxes[index], target_w, target_h)
        else:
            im = _pad_to_tile(im, target_w, target_h)
        imgs.append(im)

    # Calculate dimensions for the 2x2 grid
    grid_width = target_w * 2
    grid_height = target_h * 2

    grid = Image.new("RGB", (grid_width, grid_height))

    # Arrange images in the grid
    x_positions = [0, target_w]
    y_positions = [0, target_h]

    for i, im in enumerate(imgs):
        x = x_positions[i % 2]
        y = y_positions[i // 2]
        grid.paste(im, (x, y))

    return grid


def _pad_to_tile(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    if im.width > target_w or im.height > target_h:
        im.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    x_offset = (target_w - im.width) // 2
    y_offset = (target_h - im.height) // 2
    padded.paste(im, (x_offset, y_offset))
    return padded


def _crop_to_tile(im: Image.Image, box: Box, target_w: int, target_h: int) -> Image.Image:
    left, top, right, bottom = box
    if not (0 <= left < right <= im.width and 0 <= top < bottom <= im.height):
        raise ValueError(f"Crop box {box} is outside the {im.width}x{im.height} image")
    # Integer boxes can be off by one pixel from the exact tile ratio.
    if abs((right - left) * target_h - (bottom - top) * target_w) > max(target_w, target_h):
        raise ValueError(f"Crop box {box} does not have the {target_w}:{target_h} tile ratio")
    return im.crop(box).resize((target_w, target_h), Image.Resampling.LANCZOS)
