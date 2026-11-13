import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from analysis.study1b_runb_effect import measure


def _run(root: Path, arms, prompts=("p0", "p1"), seeds=(1001,)) -> Path:
    rows = []
    for p in prompts:
        for s in seeds:
            for arm in ("base", *arms):
                rel = Path("images") / p / str(s) / f"{arm}.png"
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shade = 40 + 20 * len(rows)
                Image.new("RGB", (64, 64), (shade, 100, 150)).save(root / rel)
                digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()
                rows.append(
                    {
                        "prompt_id": p,
                        "seed": s,
                        "condition": arm,
                        "output_path": rel.as_posix(),
                        "output_sha256": digest,
                    }
                )
    (root / "generation_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def _embed(tiles) -> np.ndarray:
    """Unit vectors from each tile's mean colour: a stand-in for DreamSim."""
    vectors = np.array([np.asarray(t, dtype=float).mean(axis=(0, 1)) for t in tiles])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_measures_each_runs_adapters_against_its_own_base(tmp_path):
    runs = {
        "runb": (_run(tmp_path / "main", ("b1", "b2")), ("b1", "b2")),
        "runb_sweep": (_run(tmp_path / "sweep", ("b1_s2000", "b1")), ("b1_s2000", "b1")),
    }
    distances, bars = measure(runs, _embed)
    assert len(distances) == (2 * 2 + 2 * 2) * 4  # (prompts x adapter arms) x tiles, per run
    assert {(d["family"], d["arm"]) for d in distances} == {
        ("runb", "b1"),
        ("runb", "b2"),
        ("runb_sweep", "b1_s2000"),
        ("runb_sweep", "b1"),
    }
    assert all(d["dreamsim_to_base"] >= 0 for d in distances)
    assert len(bars) == 2 * 3 + 2 * 3  # every grid, base included


def test_refuses_a_changed_image_or_a_missing_arm(tmp_path):
    root = _run(tmp_path / "main", ("b1",))
    with pytest.raises(ValueError, match="lacks arms"):
        measure({"runb": (root, ("b1", "b2"))}, _embed)
    Image.new("RGB", (64, 64), (0, 0, 0)).save(root / "images/p0/1001/b1.png")
    with pytest.raises(ValueError, match="manifest hash"):
        measure({"runb": (root, ("b1",))}, _embed)
