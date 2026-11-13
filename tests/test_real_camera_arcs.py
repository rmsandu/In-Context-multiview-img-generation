import struct

import numpy as np
import pytest

from analysis.real_camera_arcs import annotation_azimuths, orbit_geometry, read_images_bin


def rotmat_to_qvec(r: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(1e-12, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    return np.array(
        [
            w,
            (r[2, 1] - r[1, 2]) / (4 * w),
            (r[0, 2] - r[2, 0]) / (4 * w),
            (r[1, 0] - r[0, 1]) / (4 * w),
        ]
    )


def look_at(centre: np.ndarray, target: np.ndarray, up=np.array([0.0, 0.0, 1.0])):
    """COLMAP world-to-camera (R, t): +z forward, +y down in the image."""
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    r = np.stack([right, down, forward])
    return r, -r @ centre


def write_images_bin(path, poses):
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(poses)))
        for k, (name, (r, t)) in enumerate(poses.items()):
            handle.write(struct.pack("<i", k + 1))
            handle.write(struct.pack("<4d", *rotmat_to_qvec(r)))
            handle.write(struct.pack("<3d", *t))
            handle.write(struct.pack("<i", 1))
            handle.write(name.encode() + b"\0")
            handle.write(struct.pack("<Q", 1))
            handle.write(struct.pack("<ddq", 1.0, 2.0, -1))


def test_orbit_recovers_azimuth_and_elevation(tmp_path):
    target = np.array([0.3, -0.2, 0.1])
    azimuths = [10.0, 40.0, 75.0, 130.0, 150.0]  # counter-clockwise seen from above
    elevation = 35.0
    poses = {}
    for k, az in enumerate(azimuths):
        a, e = np.radians(az), np.radians(elevation)
        centre = target + 2.5 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        poses[f"{k + 1:03d}.jpg"] = look_at(centre, target)
    write_images_bin(tmp_path / "images.bin", poses)

    read = read_images_bin(tmp_path / "images.bin")
    for name, (r, t) in poses.items():
        np.testing.assert_allclose(read[name][0], r, atol=1e-9)
        np.testing.assert_allclose(read[name][1], t, atol=1e-9)

    geometry = orbit_geometry(read)
    relative = [geometry["frames"][n]["azimuth"] for n in sorted(read)]
    np.testing.assert_allclose(relative, np.array(azimuths) - azimuths[0], atol=1e-6)
    for frame in geometry["frames"].values():
        assert frame["elevation"] == pytest.approx(elevation, abs=1e-6)
    assert geometry["video_arc"] == pytest.approx(140.0)
    assert geometry["planarity"] < 1e-6


def test_annotation_azimuths_match_caption_labels():
    views = [
        {"horizontal_view": "front", "side": "neither"},
        {"horizontal_view": "front_three_quarter", "side": "right"},
        {"horizontal_view": "side", "side": "left"},
        {"horizontal_view": "indeterminate", "side": "indeterminate"},
    ]
    assert annotation_azimuths({"views": views}) == [0.0, 45.0, 270.0, None]


def test_calibrated_distinct_views():
    pd = pytest.importorskip("pandas")
    from analysis.study1_report import calibrated_distinct_views

    rows = []
    for grid, angles in {
        "wide": [40, 80, 120, 40, 80, 40],
        "pair": [15, 60, 90, 50, 80, 25],
    }.items():
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        rows += [
            {"grid_id": grid, "i": i, "j": j, "rel_azimuth": a} for (i, j), a in zip(pairs, angles)
        ]
    rows.append({"grid_id": "nan", "i": 0, "j": 1, "rel_azimuth": float("nan")})
    counts = calibrated_distinct_views(pd.DataFrame(rows))
    # 25 deg is above the calibrated 21 deg threshold, 15 deg is below it.
    assert counts.to_dict() == {"nan": 4, "pair": 3, "wide": 4}
