"""Tests for Polycam import functionality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.import_polycam import (
    detect_polycam_format,
    rotation_matrix_to_quaternion,
    write_colmap_cameras,
    write_colmap_images,
)


def test_rotation_matrix_to_quaternion():
    """Test rotation matrix to quaternion conversion."""
    # Identity rotation
    R = np.eye(3)
    quat = rotation_matrix_to_quaternion(R)
    assert quat[0] == pytest.approx(1.0)  # w
    assert quat[1] == pytest.approx(0.0)  # x
    assert quat[2] == pytest.approx(0.0)  # y
    assert quat[3] == pytest.approx(0.0)  # z
    
    # 90 degree rotation around Z axis
    R = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    quat = rotation_matrix_to_quaternion(R)
    # Should be [sqrt(2)/2, 0, 0, sqrt(2)/2]
    assert quat[0] == pytest.approx(0.7071, abs=0.01)
    assert quat[3] == pytest.approx(0.7071, abs=0.01)


def test_detect_colmap_format(tmp_path: Path):
    """Test detection of COLMAP export format."""
    # Create COLMAP structure
    sparse_dir = tmp_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.txt").write_text("# test")
    
    assert detect_polycam_format(tmp_path) == "colmap"


def test_detect_polycam_lidar_format(tmp_path: Path):
    """Test detection of Polycam LiDAR format."""
    (tmp_path / "cameras.json").write_text('{"frames": []}')
    (tmp_path / "mesh.ply").write_text("ply header")
    
    assert detect_polycam_format(tmp_path) == "polycam_lidar"


def test_detect_unknown_format(tmp_path: Path):
    """Test error on unknown format."""
    with pytest.raises(ValueError, match="Could not detect Polycam format"):
        detect_polycam_format(tmp_path)


def test_write_colmap_cameras(tmp_path: Path):
    """Test writing COLMAP cameras.txt."""
    output = tmp_path / "cameras.txt"
    intrinsics = {
        "width": 1920,
        "height": 1080,
        "fx": 1500.0,
        "fy": 1500.0,
        "cx": 960.0,
        "cy": 540.0,
    }
    
    write_colmap_cameras(intrinsics, output)
    
    assert output.exists()
    content = output.read_text()
    assert "PINHOLE" in content
    assert "1920 1080" in content
    assert "1500.0" in content


def test_write_colmap_images(tmp_path: Path):
    """Test writing COLMAP images.txt."""
    output = tmp_path / "images.txt"
    frames = [
        {
            "file_path": "frame_0000.jpg",
            "transform_matrix": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        },
        {
            "file_path": "frame_0001.jpg",
            "transform_matrix": [
                [1, 0, 0, 0.1],
                [0, 1, 0, 0.0],
                [0, 0, 1, 0.0],
                [0, 0, 0, 1],
            ],
        },
    ]
    
    write_colmap_images(frames, output)
    
    assert output.exists()
    content = output.read_text()
    assert "frame_0000.jpg" in content
    assert "frame_0001.jpg" in content
    # Count non-comment lines (should have image lines, empty point lines are allowed to be blank)
    lines = [l for l in content.splitlines() if not l.startswith("#")]
    # Should have at least 2 image lines (empty lines may or may not be present)
    assert len([l for l in lines if l]) >= 2


def test_invalid_directory():
    """Test error on non-existent directory."""
    with pytest.raises(ValueError, match="does not exist"):
        detect_polycam_format(Path("/nonexistent/path"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
