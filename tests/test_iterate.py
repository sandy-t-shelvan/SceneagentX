"""The loop keeps a finished COLMAP model and does not retrain a splat."""

from __future__ import annotations

import json
from pathlib import Path

from tools.common.status import load_status, save_status, set_path
from tools.iterate import iterate


ANALYZER = """
num_images: 224
num_reg_images: 224
mean_reprojection_error: 0.543369
"""


def test_video_keeps_colmap_and_splat_without_rerun(tmp_path: Path, monkeypatch):
    def fail_colmap(*args, **kwargs):
        raise AssertionError("COLMAP was restarted")

    monkeypatch.setattr("tools.step01_colmap.run_orbit", fail_colmap)
    folder = tmp_path / "data" / "bottle"
    raw = folder / "raw"
    raw.mkdir(parents=True)
    (raw / "orbit1.mp4").write_bytes(b"video")
    colmap = folder / "colmap" / "orbit1"
    colmap.mkdir(parents=True)
    (colmap / "analyzer.txt").write_text(ANALYZER, encoding="utf-8")
    (colmap / "models.json").write_text('["0"]\n', encoding="utf-8")
    (colmap / "splat.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 400000\nend_header\n",
        encoding="utf-8",
    )
    masks = folder / "masks"
    masks.mkdir()
    (masks / "object.json").write_text(
        json.dumps({"grounding_dino": True, "sam": True, "n_views_with_detection": 224}),
        encoding="utf-8",
    )
    (folder / "codebook.json").write_text(
        json.dumps({"dinov2": True, "openclip": True, "label": "bottle"}),
        encoding="utf-8",
    )
    code = iterate("bottle", tmp_path)
    assert code == 0
    text = (folder / "iterate.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    actions = {row["block"]: row["action"] for row in payload["steps"]}
    assert actions == {
        "B2": "kept",
        "B3": "kept",
        "B4": "skip",
        "B5": "kept",
        "B6": "skip",
        "B7": "kept",
    }
    doc = load_status("bottle", tmp_path)
    assert doc["step01"]["orbit1"]["passed"] is True
    assert doc["step02"]["orbit1"]["trained_again"] is False
    assert doc["step02"]["orbit1"]["photometric"] == "not measured"
    assert next(step for step in doc["steps"] if step["id"] == "B3")["state"] == "pass"


def test_ply_does_not_call_colmap(tmp_path: Path, monkeypatch):
    def fail_colmap(*args, **kwargs):
        raise AssertionError("COLMAP was called")

    monkeypatch.setattr("tools.step01_colmap.run_orbit", fail_colmap)
    doc = load_status("scan", tmp_path)
    set_path(doc, "input", {"kind": "polycam_mesh", "cameras": False, "capture_images": False})
    save_status("scan", doc, tmp_path)
    code = iterate("scan", tmp_path)
    assert code == 0
    payload = json.loads((tmp_path / "data" / "scan" / "iterate.json").read_text(encoding="utf-8"))
    assert payload["path"] == "ply"
    assert payload["colmap"] == "not called"
