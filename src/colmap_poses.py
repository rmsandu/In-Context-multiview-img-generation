"""Load registered camera poses from a COLMAP reconstruction."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PoseRecord:
    """A registered COLMAP image and its world-space camera pose."""

    image_id: int
    image_name: str
    image_path: Path
    rotation_matrix: FloatArray
    translation: FloatArray
    camera_center: FloatArray
    forward: FloatArray
    up: FloatArray
    num_points3D: int


def _import_pycolmap() -> ModuleType:
    try:
        return importlib.import_module("pycolmap")
    except ImportError as error:
        raise RuntimeError(
            "PyCOLMAP is required to load reconstructions. "
            'Install the preprocessing dependencies with `pip install -e ".[preprocessing]`.'
        ) from error


def _rotation_matrix_from_xyzw(quaternion: object) -> FloatArray:
    """Convert PyCOLMAP's XYZW quaternion to a world-to-camera rotation matrix."""
    xyzw = np.asarray(quaternion, dtype=np.float64)
    if xyzw.shape != (4,) or not np.isfinite(xyzw).all():
        raise ValueError("rotation quaternion must contain four finite XYZW values")

    norm = np.linalg.norm(xyzw)
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("rotation quaternion has zero or non-finite norm")

    x, y, z, w = xyzw / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _relative_image_path(images_dir: Path, image_name: str) -> Path:
    """Resolve an exact COLMAP POSIX-relative image name below ``images_dir``."""
    relative = PurePosixPath(image_name)
    if not image_name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"COLMAP image name is not a safe relative path: {image_name!r}")

    image_path = images_dir.joinpath(*relative.parts)
    if not image_path.is_file():
        raise FileNotFoundError(
            f"COLMAP image {image_name!r} does not exist exactly under {images_dir}"
        )
    if not image_path.resolve().is_relative_to(images_dir.resolve()):
        raise ValueError(f"COLMAP image path escapes the images directory: {image_name!r}")
    return image_path


def load_colmap_poses(instance_dir: Path) -> list[PoseRecord]:
    """Load all registered poses from an MVImgNet-style instance directory.

    COLMAP stores a world-to-camera transform. PyCOLMAP exposes its quaternion
    in XYZW order, unlike the WXYZ order documented for COLMAP model files.
    """
    instance_dir = Path(instance_dir)
    images_dir = instance_dir / "images"
    model_dir = instance_dir / "sparse" / "0"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"COLMAP reconstruction does not exist: {model_dir}")

    pycolmap = _import_pycolmap()
    reconstruction = pycolmap.Reconstruction(model_dir)
    image_ids = sorted(int(image_id) for image_id in reconstruction.reg_image_ids())

    registered_images = [reconstruction.image(image_id) for image_id in image_ids]
    image_names = [str(image.name) for image in registered_images]
    duplicate_names = sorted({name for name in image_names if image_names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            "Registered COLMAP images contain ambiguous duplicate names: "
            + ", ".join(repr(name) for name in duplicate_names)
        )

    records: list[PoseRecord] = []
    for image_id, image, image_name in zip(image_ids, registered_images, image_names, strict=True):
        image_path = _relative_image_path(images_dir, image_name)
        cam_from_world = image.cam_from_world()
        rotation = _rotation_matrix_from_xyzw(cam_from_world.rotation.quat)
        translation = np.asarray(cam_from_world.translation, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(
                f"Registered image {image_name!r} has a non-finite or invalid translation"
            )

        camera_center = -rotation.T @ translation
        forward = rotation.T @ np.array([0.0, 0.0, 1.0])
        # COLMAP camera coordinates use +Y down, so image up is camera -Y.
        up = rotation.T @ np.array([0.0, -1.0, 0.0])
        pose_values = (rotation, translation, camera_center, forward, up)
        if not all(np.isfinite(value).all() for value in pose_values):
            raise ValueError(f"Registered image {image_name!r} has a non-finite pose")

        records.append(
            PoseRecord(
                image_id=image_id,
                image_name=image_name,
                image_path=image_path,
                rotation_matrix=rotation,
                translation=translation.copy(),
                camera_center=camera_center,
                forward=forward,
                up=up,
                num_points3D=int(image.num_points3D),
            )
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print registered camera poses for one COLMAP instance."
    )
    parser.add_argument("instance_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    poses = load_colmap_poses(args.instance_dir)
    print(f"Loaded {len(poses)} registered camera pose(s) from {args.instance_dir}")
    for pose in poses:
        center = ", ".join(f"{coordinate:.6g}" for coordinate in pose.camera_center)
        print(f"{pose.image_id}: {pose.image_name} center=({center}) points3D={pose.num_points3D}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
