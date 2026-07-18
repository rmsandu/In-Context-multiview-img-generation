from pathlib import Path

from PIL import Image

from src.sanity_check import create_contact_sheet, find_captioned_composites, main


def _write_pair(directory: Path, stem: str, color: str) -> None:
    Image.new("RGB", (32, 32), color).save(directory / f"{stem}.png")
    (directory / f"{stem}.txt").write_text(f"Caption for {stem}", encoding="utf-8")


def test_find_captioned_composites_requires_matching_caption(tmp_path: Path) -> None:
    _write_pair(tmp_path, "paired", "red")
    Image.new("RGB", (32, 32), "blue").save(tmp_path / "unpaired.png")

    assert find_captioned_composites(tmp_path) == [tmp_path / "paired.png"]


def test_create_contact_sheet_and_seeded_cli(tmp_path: Path) -> None:
    for index in range(4):
        _write_pair(tmp_path, f"sample_{index}", "red")

    output = tmp_path / "sheet.png"
    assert main(
        [
            "--input-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--count",
            "4",
            "--columns",
            "2",
            "--tile-size",
            "64",
            "--seed",
            "7",
        ]
    ) == 0

    with Image.open(output) as sheet:
        assert sheet.width == 128
        assert sheet.height > 128


def test_create_contact_sheet_supports_pdf(tmp_path: Path) -> None:
    _write_pair(tmp_path, "sample", "green")

    output = tmp_path / "sheet.pdf"
    create_contact_sheet([tmp_path / "sample.png"], output, tile_size=64)

    assert output.is_file()
