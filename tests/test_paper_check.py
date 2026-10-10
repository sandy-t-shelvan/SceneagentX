"""Paper-check scores laptop artifacts without calling a vision API."""

from __future__ import annotations

import json
from pathlib import Path

from tools.common.status import find_step, load_status, save_status, set_path
from tools.paper_check import build_report, write_paper_check


def _stage(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "scan"
    (folder / "masks").mkdir(parents=True)
    (folder / "sisters").mkdir()
    (folder / "colmap" / "orbit1").mkdir(parents=True)
    splat = folder / "colmap" / "orbit1" / "splat.ply"
    splat.write_bytes(b"ply\n")
    (folder / "scene.usdz").write_bytes(b"PK")
    (folder / "masks" / "object.json").write_text(
        json.dumps({"segmenter": "silhouette", "grounding_dino": False, "sam": False}),
        encoding="utf-8",
    )
    (folder / "physics.json").write_text(
        json.dumps(
            {
                "source": "prior_table",
                "per_gaussian": {"copied_from": "object", "n": 4, "learned_prior": False},
            }
        ),
        encoding="utf-8",
    )
    (folder / "codebook.json").write_text(
        json.dumps({"encoder": "geometric", "dinov2": False, "openclip": False}),
        encoding="utf-8",
    )
    (folder / "parts.json").write_text(
        json.dumps({"partfield": False, "parts": [{"id": "body"}]}),
        encoding="utf-8",
    )
    (folder / "asset.urdf").write_text("<robot><link name=\"body\"/></robot>\n", encoding="utf-8")
    (folder / "sisters" / "objects.json").write_text(
        json.dumps({"trellis": False, "objects": []}),
        encoding="utf-8",
    )
    (folder / "sisters" / "scenes.json").write_text(
        json.dumps({"scenes": [{"file": "layout_01.usdz"}]}),
        encoding="utf-8",
    )
    doc = load_status("scan", tmp_path)
    set_path(
        doc,
        "input",
        {"kind": "polycam_mesh", "cameras": False, "capture_images": False},
    )
    set_path(
        doc,
        "step02.orbit1",
        {
            "source": "mesh",
            "splat": str(splat),
            "n_gaussians": 4,
            "photometric": False,
        },
    )
    set_path(doc, "align.report", {"photometric": "skipped", "scale_m": 0.25})
    set_path(doc, "orientation", {"skipped": True, "reason": "no separate background"})
    set_path(doc, "inspect", {"isaac": False})
    find_step(doc, "B8")["actual"] = {"chamfer_cm": 0.23}
    save_status("scan", doc, tmp_path)


def test_paper_check_marks_gpu_blocks_unmatched(tmp_path: Path):
    _stage(tmp_path)
    by_id = {row["id"]: row["match"] for row in build_report("scan", tmp_path)["blocks"]}
    assert by_id == {
        "input": True,
        "3dgs": True,
        "semantics": False,
        "physics": False,
        "decomposition": "partial",
        "articulation": "partial",
        "alignment": "partial",
        "orientation": False,
        "package": "partial",
        "inspector": False,
        "object_sisters": False,
        "scene_sisters": "partial",
        "policy": False,
    }
    physics = tmp_path / "data" / "scan" / "physics.json"
    physics.write_text(
        json.dumps(
            {
                "source": "vlm",
                "model": "gpt-6-astra",
                "per_gaussian": {"copied_from": "object", "n": 4, "learned_prior": False, "variation": "not_run"},
            }
        ),
        encoding="utf-8",
    )
    vlm_rows = {row["id"]: row for row in build_report("scan", tmp_path)["blocks"]}
    assert vlm_rows["physics"]["match"] == "partial"
    assert "gpt-6-astra" in vlm_rows["physics"]["note"]
    assert "GroundingDINO" in vlm_rows["semantics"]["note"]
    path = write_paper_check("scan", tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "sk-" not in path.read_text(encoding="utf-8")
    assert [row["id"] for row in saved["blocks"]] == list(by_id)
    doc = load_status("scan", tmp_path)
    assert find_step(doc, "B16")["state"] == "fail"
    assert find_step(doc, "B5")["actual"]["paper_match"] is False
