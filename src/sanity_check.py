"""Create a random contact sheet of captioned four-view composites."""

from __future__ import annotations

import argparse
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(frozen=True)
class SheetEntry:
    image_path: Path
    title_lines: tuple[str, ...]
    caption_lines: tuple[str, ...]


def find_captioned_composites(input_dir: Path) -> list[Path]:
    """Return sorted PNG files that have a matching caption text file."""
    return sorted(
        image_path
        for image_path in input_dir.glob("*.png")
        if image_path.with_suffix(".txt").is_file()
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def create_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    *,
    columns: int = 3,
    tile_size: int = 384,
) -> None:
    """Write a labeled contact sheet for the supplied composite images."""
    if not image_paths:
        raise ValueError("At least one image is required")
    if columns < 1 or tile_size < 1:
        raise ValueError("columns and tile_size must be positive")

    rows = (len(image_paths) + columns - 1) // columns
    title_font = _font(17)
    caption_font = _font(14)
    title_line_height = title_font.getbbox("Ag")[3] + 3
    caption_line_height = caption_font.getbbox("Ag")[3] + 3
    title_wrap_width = max(20, tile_size // 10)
    caption_wrap_width = max(25, tile_size // 8)

    entries: list[SheetEntry] = []
    for index, image_path in enumerate(image_paths):
        caption = image_path.with_suffix(".txt").read_text(encoding="utf-8").strip()
        entries.append(
            SheetEntry(
                image_path=image_path,
                title_lines=tuple(
                    textwrap.wrap(
                        f"{index + 1:02d}. {image_path.name}", width=title_wrap_width
                    )
                ),
                caption_lines=tuple(
                    textwrap.wrap(
                        caption,
                        width=caption_wrap_width,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                ),
            )
        )

    row_label_heights: list[int] = []
    for row in range(rows):
        row_entries = entries[row * columns : (row + 1) * columns]
        row_label_heights.append(
            max(
                16
                + len(entry.title_lines) * title_line_height
                + len(entry.caption_lines) * caption_line_height
                for entry in row_entries
            )
        )

    row_offsets: list[int] = []
    current_y = 0
    for label_height in row_label_heights:
        row_offsets.append(current_y)
        current_y += tile_size + label_height

    sheet = Image.new("RGB", (columns * tile_size, current_y), "white")
    draw = ImageDraw.Draw(sheet)

    for index, entry in enumerate(entries):
        row, column = divmod(index, columns)
        x = column * tile_size
        y = row_offsets[row]

        with Image.open(entry.image_path) as source:
            image = ImageOps.contain(source.convert("RGB"), (tile_size, tile_size))
        image_x = x + (tile_size - image.width) // 2
        image_y = y + (tile_size - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), outline="#777777")

        text_y = y + tile_size + 6
        draw.multiline_text(
            (x + 8, text_y),
            "\n".join(entry.title_lines),
            fill="black",
            font=title_font,
            spacing=3,
        )
        text_y += len(entry.title_lines) * title_line_height + 2
        draw.multiline_text(
            (x + 8, text_y),
            "\n".join(entry.caption_lines),
            fill="#333333",
            font=caption_font,
            spacing=3,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".pdf"}:
        raise ValueError("Output must use a .png, .jpg, .jpeg, or .pdf extension")
    sheet.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Randomly sample captioned composites and create a labeled contact sheet."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("sanity_check.png"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Omit it for a new random sample each run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    candidates = find_captioned_composites(args.input_dir)
    if len(candidates) < args.count:
        raise ValueError(
            f"Requested {args.count} samples, but only {len(candidates)} paired PNG/TXT files exist"
        )

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    selected = random.Random(seed).sample(candidates, args.count)
    create_contact_sheet(
        selected,
        args.output,
        columns=args.columns,
        tile_size=args.tile_size,
    )

    print(f"Contact sheet: {args.output}")
    print(f"Random seed: {seed}")
    for path in selected:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
