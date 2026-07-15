import argparse
from pathlib import Path


def rename_files_in_directory(directory: Path, start: int = 1) -> None:
    """Rename paired image and caption files to sequential numeric names."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    files = sorted(directory.iterdir())
    image_files = [path for path in files if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    text_files = [path for path in files if path.suffix.lower() == ".txt"]
    if len(image_files) != len(text_files):
        raise ValueError("The number of image files and caption files must match")

    for index, (image, caption) in enumerate(zip(image_files, text_files), start=start):
        image.rename(directory / f"{index:04d}{image.suffix.lower()}")
        caption.rename(directory / f"{index:04d}.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--start", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    rename_files_in_directory(args.directory, args.start)


if __name__ == "__main__":
    main()
