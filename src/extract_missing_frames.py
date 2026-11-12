"""Extract the frames of objects whose local ``images/`` folder is empty.

The local copy of MVImgNet shards ``mvi_32`` and ``mvi_40`` has 1,367 objects with
COLMAP files but no frames. The original shard zips have all of them. This copies
only the JPEG frames of those objects into a separate tree, leaving ``data/``
untouched:

    python -m src.extract_missing_frames \\
        --zip mvi_32=/mnt/c/Users/Raluca/develop/mvi_32.zip \\
        --zip mvi_40=/mnt/c/Users/Raluca/develop/mvi_40.zip \\
        --objects-dir data --output-dir data_runb

Only members named ``<category>/<object>/images/<digits>.jpg`` are read; anything
else in the archive (COLMAP files, ``*_bg_removed.png`` cut-outs) is ignored.
``extraction_manifest.jsonl`` lists every written frame with its SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

FRAME = re.compile(r"(?P<category>\d+)/(?P<object>[0-9A-Za-z]+)/images/(?P<frame>\d+\.jpg)")


def empty_objects(objects_dir: Path, shard: str) -> set[str]:
    """``category/object`` keys whose local images folder has no JPEG."""
    return {
        f"{d.parent.parent.name}/{d.parent.name}"
        for d in (objects_dir / shard).glob("*/*/images")
        if d.is_dir() and not any(d.glob("*.jpg"))
    }


def plan(names: list[str], wanted: set[str]) -> dict[str, list[str]]:
    """Archive members to extract, grouped by object, for the wanted objects only."""
    out: dict[str, list[str]] = {}
    for name in names:
        match = FRAME.fullmatch(name)
        if match and (key := f"{match['category']}/{match['object']}") in wanted:
            out.setdefault(key, []).append(name)
    return {key: sorted(members) for key, members in sorted(out.items())}


def extract(archive: Path, shard: str, wanted: set[str], output_dir: Path) -> list[dict]:
    records = []
    with zipfile.ZipFile(archive) as z:
        members = plan(z.namelist(), wanted)
        missing = wanted - set(members)
        if missing:
            raise ValueError(f"{len(missing)} objects have no frames in {archive}")
        for key, names in members.items():
            target_dir = output_dir / shard / key / "images"
            target_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                data = z.read(name)
                target = target_dir / Path(name).name
                if target.exists() and target.read_bytes() != data:
                    raise FileExistsError(f"{target} exists with different content")
                target.write_bytes(data)
                records.append(
                    {
                        "path": f"{shard}/{name}",
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "archive": archive.name,
                    }
                )
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zip", action="append", required=True, help="SHARD=PATH")
    parser.add_argument("--objects-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_runb"))
    args = parser.parse_args(argv)

    records = []
    for spec in args.zip:
        shard, _, path = spec.partition("=")
        wanted = empty_objects(args.objects_dir, shard)
        part = extract(Path(path), shard, wanted, args.output_dir)
        print(f"{shard}: {len(wanted)} objects, {len(part)} frames", flush=True)
        records += part
    manifest = args.output_dir / "extraction_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    print(f"wrote {len(records)} frames; manifest {manifest}")


if __name__ == "__main__":
    main()
