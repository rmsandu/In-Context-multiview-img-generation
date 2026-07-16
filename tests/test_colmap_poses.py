import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src import colmap_poses
from src.colmap_poses import PoseRecord, load_colmap_poses


class FakeRotation:
    def __init__(self, quat):
        self.quat = np.asarray(quat, dtype=np.float64)


class FakeImage:
    def __init__(self, name, *, quat=(0.0, 0.0, 0.0, 1.0), translation=(0, 0, 0), points=0):
        self.name = name
        self._pose = SimpleNamespace(
            rotation=FakeRotation(quat),
            translation=np.asarray(translation, dtype=np.float64),
        )
        self.num_points3D = points

    def cam_from_world(self):
        return self._pose


class FakeReconstruction:
    def __init__(self, images, registered_ids):
        self._images = images
        self._registered_ids = registered_ids
        self.requested_image_ids = []

    def reg_image_ids(self):
        return list(self._registered_ids)

    def image(self, image_id):
        self.requested_image_ids.append(image_id)
        return self._images[image_id]


def _instance(tmp_path: Path) -> Path:
    instance_dir = tmp_path / "instance"
    (instance_dir / "images").mkdir(parents=True)
    (instance_dir / "sparse" / "0").mkdir(parents=True)
    return instance_dir


def _install_pycolmap(monkeypatch, reconstruction):
    calls = []

    def constructor(path):
        calls.append(Path(path))
        return reconstruction

    monkeypatch.setitem(sys.modules, "pycolmap", SimpleNamespace(Reconstruction=constructor))
    return calls


def test_loads_registered_nested_image_and_converts_xyzw_pose(tmp_path, monkeypatch):
    instance_dir = _instance(tmp_path)
    nested_image = instance_dir / "images" / "nested" / "frame.jpg"
    nested_image.parent.mkdir()
    nested_image.touch()
    root_image = instance_dir / "images" / "ignored.jpg"
    root_image.touch()

    half_sqrt_two = np.sqrt(0.5)
    reconstruction = FakeReconstruction(
        {
            7: FakeImage(
                "nested/frame.jpg",
                quat=(0.0, 0.0, half_sqrt_two, half_sqrt_two),
                translation=(1.0, 2.0, 3.0),
                points=42,
            ),
            99: FakeImage("ignored.jpg", translation=(np.nan, 0.0, 0.0)),
        },
        registered_ids=[7],
    )
    constructor_calls = _install_pycolmap(monkeypatch, reconstruction)

    records = load_colmap_poses(instance_dir)

    assert constructor_calls == [instance_dir / "sparse" / "0"]
    assert reconstruction.requested_image_ids == [7]
    assert len(records) == 1
    pose = records[0]
    assert pose.image_id == 7
    assert pose.image_name == "nested/frame.jpg"
    assert pose.image_path == nested_image
    np.testing.assert_allclose(
        pose.rotation_matrix,
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(pose.translation, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose.camera_center, [-2.0, 1.0, -3.0])
    np.testing.assert_allclose(pose.forward, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(pose.up, [-1.0, 0.0, 0.0], atol=1e-12)
    assert pose.num_points3D == 42


def test_sorts_registered_image_ids(tmp_path, monkeypatch):
    instance_dir = _instance(tmp_path)
    for name in ("one.jpg", "two.jpg"):
        (instance_dir / "images" / name).touch()
    reconstruction = FakeReconstruction(
        {1: FakeImage("one.jpg"), 2: FakeImage("two.jpg")}, registered_ids=[2, 1]
    )
    _install_pycolmap(monkeypatch, reconstruction)

    records = load_colmap_poses(instance_dir)

    assert [record.image_id for record in records] == [1, 2]


def test_requires_exact_relative_image_path(tmp_path, monkeypatch):
    instance_dir = _instance(tmp_path)
    (instance_dir / "images" / "frame.jpg").touch()
    reconstruction = FakeReconstruction({1: FakeImage("nested/frame.jpg")}, registered_ids=[1])
    _install_pycolmap(monkeypatch, reconstruction)

    with pytest.raises(FileNotFoundError, match="does not exist exactly"):
        load_colmap_poses(instance_dir)


def test_rejects_ambiguous_registered_names(tmp_path, monkeypatch):
    instance_dir = _instance(tmp_path)
    (instance_dir / "images" / "same.jpg").touch()
    reconstruction = FakeReconstruction(
        {1: FakeImage("same.jpg"), 2: FakeImage("same.jpg")}, registered_ids=[1, 2]
    )
    _install_pycolmap(monkeypatch, reconstruction)

    with pytest.raises(ValueError, match="ambiguous duplicate names"):
        load_colmap_poses(instance_dir)


@pytest.mark.parametrize(
    ("quat", "translation"),
    [
        ((np.nan, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0, 1.0), (np.inf, 0.0, 0.0)),
    ],
)
def test_rejects_non_finite_or_invalid_poses(tmp_path, monkeypatch, quat, translation):
    instance_dir = _instance(tmp_path)
    (instance_dir / "images" / "frame.jpg").touch()
    reconstruction = FakeReconstruction(
        {1: FakeImage("frame.jpg", quat=quat, translation=translation)},
        registered_ids=[1],
    )
    _install_pycolmap(monkeypatch, reconstruction)

    with pytest.raises(ValueError, match="quaternion|translation"):
        load_colmap_poses(instance_dir)


def test_requires_expected_instance_directories(tmp_path):
    with pytest.raises(FileNotFoundError, match="Images directory"):
        load_colmap_poses(tmp_path)

    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="reconstruction"):
        load_colmap_poses(tmp_path)


def test_cli_prints_pose_summary(tmp_path, monkeypatch, capsys):
    record = PoseRecord(
        image_id=3,
        image_name="frame.jpg",
        image_path=tmp_path / "images" / "frame.jpg",
        rotation_matrix=np.eye(3),
        translation=np.array([0.0, 0.0, 0.0]),
        camera_center=np.array([1.25, -2.5, 3.0]),
        forward=np.array([0.0, 0.0, 1.0]),
        up=np.array([0.0, -1.0, 0.0]),
        num_points3D=17,
    )
    monkeypatch.setattr(colmap_poses, "load_colmap_poses", lambda path: [record])

    assert colmap_poses.main([str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Loaded 1 registered camera pose(s)" in output
    assert "3: frame.jpg center=(1.25, -2.5, 3) points3D=17" in output
