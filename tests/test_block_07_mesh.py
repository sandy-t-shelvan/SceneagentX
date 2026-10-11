"""Block 7 / E: mesh score on a synthetic cube. A missing mesh is not achieved."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.block_gates import SCENE_CHAMFER_CM_MAX, SCENE_F1_1CM_MIN, chamfer_cm, cube_corners, f1_at_1cm, write_mesh_score


def test_known_box_chamfer_matches_hand_value():
    cube = cube_corners(0.02)
    assert chamfer_cm(cube, cube) == 0.0
    shifted = cube + np.array([0.01, 0.0, 0.0])
    assert abs(chamfer_cm(cube, shifted) - 1.0) < 1e-6


def test_f1_at_1cm_on_fixture(tmp_path: Path):
    cube = cube_corners(0.02)
    score = f1_at_1cm(cube, cube)
    assert score == 1.0
    assert score > SCENE_F1_1CM_MIN
    mesh = tmp_path / "cube.ply"
    mesh.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    present = write_mesh_score(tmp_path, mesh, chamfer_cm=0.0, f1=score)
    assert present["achieved"] is True
    assert present["f1_at_1cm"] > 0.20


def test_missing_mesh_not_achieved(tmp_path: Path):
    missing = write_mesh_score(tmp_path, tmp_path / "no-mesh.ply", chamfer_cm=0.1, f1=0.9)
    assert missing["state"] == "unstarted"
    assert missing["achieved"] is False
    assert missing["chamfer_cm"] is None
    saved = json_score(tmp_path)
    assert saved["achieved"] is False
    cube = tmp_path / "cube.ply"
    cube.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    present = write_mesh_score(tmp_path, cube, chamfer_cm=0.2342, f1=0.5)
    assert present["achieved"] is True
    assert present["chamfer_cm"] < SCENE_CHAMFER_CM_MAX
    assert present["f1_at_1cm"] > SCENE_F1_1CM_MIN
    wide = write_mesh_score(tmp_path, cube, chamfer_cm=5.0, f1=0.1)
    assert wide["achieved"] is False


def json_score(folder: Path) -> dict:
    import json

    return json.loads((folder / "mesh_score.json").read_text(encoding="utf-8"))
