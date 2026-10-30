"""Design-block scores stay honest and land in the change log."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.sceneagent_blocks import run_design


def test_blocks_record_partial_work_and_skip_the_policy(tmp_path: Path):
    folder = tmp_path / "data" / "bottle"
    _write_bottle(folder)
    state = run_design("bottle", tmp_path)
    by_id = {step["id"]: step for step in state["steps"]}
    assert by_id["A"]["state"] == "partial"
    assert "M1 is not met" in by_id["A"]["result"]
    assert by_id["C"]["state"] == "partial"
    assert "infill" in by_id["C"]["result"]
    assert by_id["F"]["state"] == "not_started"
    assert by_id["J"]["state"] == "not_started"
    assert by_id["K"]["state"] == "not_started"
    assert all(step["learned_prior"] is False for step in state["steps"])
    text = (folder / "changelog.md").read_text(encoding="utf-8")
    assert "## Design blocks" in text
    assert "docs/sceneagent_system_design.md" in text
    assert "not_started" in text
    index = np.load(folder / "semantics" / "gaussian_feature_idx.npy")
    assert index.dtype == np.int8
    assert int((index == 0).sum()) == 2
    assert int((index == -1).sum()) == 1
    scale = json.loads((folder / "ingest" / "scale.json").read_text(encoding="utf-8"))
    assert scale["metres_per_unit"] is None


def test_crop_drops_faint_gaussians_and_recenters(tmp_path: Path):
    from tools.design_workspace import crop_splat

    splat = tmp_path / "splat.ply"
    rows = np.array(
        [
            [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\nproperty float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\nend_header\n"
    )
    splat.write_bytes(header.encode("ascii") + rows.tobytes())
    stats = crop_splat(splat, np.array([True, True, True]), tmp_path / "object.ply")
    assert stats["n_in"] == 2
    assert stats["infill"] == "not_run"
    kept = np.frombuffer((tmp_path / "object.ply").read_bytes().split(b"end_header\n", 1)[1], dtype="<f4")
    kept = kept.reshape(2, 7)
    assert np.allclose(kept[:, 0:3].mean(axis=0), 0.0, atol=1e-5)


def test_learned_prior_true_is_refused(tmp_path: Path):
    folder = tmp_path / "data" / "scan"
    folder.mkdir(parents=True)
    (folder / "physics.json").write_text(
        json.dumps({"material": "steel", "mass_kg": 1, "learned_prior": True}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="learned_prior"):
        run_design("scan", tmp_path)


def _write_bottle(folder: Path) -> None:
    colmap = folder / "colmap" / "orbit1"
    colmap.mkdir(parents=True)
    (colmap / "analyzer.txt").write_text(
        "num_images: 224\nnum_reg_images: 224\nmean_reprojection_error: 0.54\n",
        encoding="utf-8",
    )
    (colmap / "models.json").write_text('["0"]\n', encoding="utf-8")
    (colmap / "splat.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1000\nend_header\n",
        encoding="utf-8",
    )
    masks = folder / "masks"
    masks.mkdir()
    (masks / "object.json").write_text(
        json.dumps(
            {
                "grounding_dino": True,
                "sam": True,
                "projected": True,
                "infill": "skipped",
                "n_views_with_detection": 224,
                "n_inside": 10,
            }
        ),
        encoding="utf-8",
    )
    (folder / "codebook.json").write_text(
        json.dumps({"dinov2": True, "openclip": True, "vector": [0.1, 0.2]}),
        encoding="utf-8",
    )
    np.save(masks / "gaussian_in_mask.npy", np.array([True, False, True]))
    (folder / "physics.json").write_text(
        json.dumps({"material": "steel", "mass_kg": 0.35, "per_gaussian": {"learned_prior": False}}),
        encoding="utf-8",
    )
    (folder / "asset.urdf").write_text("<robot><link name='body'/></robot>\n", encoding="utf-8")
