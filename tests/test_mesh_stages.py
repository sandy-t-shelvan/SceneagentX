"""Mesh-only stages after ingest: masks, physics, URDF, USDZ, sisters."""

from __future__ import annotations

import json
import urllib.error
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from tools.common.status import load_status, save_status, set_path
from tools.mesh_stages import _VISION_MODEL, _physics
from tools.pipeline import run


def test_mesh_pipeline_writes_the_scene_package(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mesh = tmp_path / "scan.obj"
    box = trimesh.creation.box(extents=(0.12, 0.05, 0.04))
    box.export(mesh)
    code = run(
        "speaker",
        through="B14",
        root=tmp_path,
        source=mesh,
        n_gaussians=48,
        n_views=4,
    )
    assert code == 0
    root = tmp_path / "data" / "speaker"
    doc = load_status("speaker", tmp_path)
    assert doc["input"]["kind"] == "polycam_mesh"
    assert next(step for step in doc["steps"] if step["id"] == "B2")["state"] == "not_started"
    assert (root / "masks" / "object.json").is_file()
    assert doc["steps"] and next(step for step in doc["steps"] if step["id"] == "B5")["actual"]["background"] == "none"
    physics = (root / "physics.json").read_text(encoding="utf-8")
    assert "friction" in physics
    assert "rigidity" in physics
    assert "prior_table" in physics
    urdf = (root / "asset.urdf").read_text(encoding="utf-8")
    assert "<joint" not in urdf
    assert "body" in urdf
    align = doc["align"]["report"]
    assert align["photometric"] == "skipped"
    assert doc["orientation"]["skipped"] is True
    usdz = root / "scene.usdz"
    with zipfile.ZipFile(usdz) as archive:
        usda = archive.read("scene.usda").decode("utf-8")
    assert "physics:friction" in usda
    assert "def Mesh" in usda
    chamfer = next(step for step in doc["steps"] if step["id"] == "B8")["actual"]["chamfer_cm"]
    assert chamfer < 1.0
    assert len(list((root / "sisters" / "objects").glob("*.usdz"))) == 3
    assert len(list((root / "sisters" / "scenes").glob("*.usdz"))) == 3
    assert doc["inspect"]["isaac"] is False
    assert (root / "inspect" / "preview.png").is_file()
    masks = json.loads((root / "masks" / "object.json").read_text(encoding="utf-8"))
    assert masks["projected"] is True
    assert masks["infill"] == "skipped"
    assert masks["n_inside"] > 0
    physics = json.loads((root / "physics.json").read_text(encoding="utf-8"))
    assert physics["per_gaussian"]["learned_prior"] is False
    assert physics["gaps"]["heterogeneous_openusd_rigidity"]
    assert "physics:learned_prior = false" in usda
    assert "physics:heterogeneous_rigidity = false" in usda
    changelog = (root / "changelog.md").read_text(encoding="utf-8")
    assert "**alignment** — Achieved" not in changelog
    assert "**orientation** — Achieved" not in changelog
    assert "SCENEAGENT-RUN-STEP" in changelog
    review = json.loads((root / "review.json").read_text(encoding="utf-8"))
    assert review["sceneagent_run_step"] is False
    agents = json.loads((root / "agents.json").read_text(encoding="utf-8"))
    assert agents["ran"] is True
    assert agents["visual_review"] is False
    assert len(agents["skills"]) == 9


def test_trellis_sisters_replace_the_procedural_copies(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mesh = tmp_path / "scan.obj"
    box = trimesh.creation.box(extents=(0.08, 0.04, 0.03))
    box.export(mesh)
    assert run("speaker", through="B14", root=tmp_path, source=mesh, n_gaussians=16, n_views=2) == 0
    glbs = []
    for index, scale in enumerate((0.9, 1.0, 1.1), start=1):
        path = tmp_path / f"sister_{index:02d}.glb"
        sister = box.copy()
        sister.apply_scale(scale)
        sister.export(path)
        glbs.append(path)
    from tools.mesh_stages import adopt_trellis_sisters

    adopt_trellis_sisters("speaker", glbs, tmp_path, n_views=8)
    manifest = json.loads((tmp_path / "data" / "speaker" / "sisters" / "objects.json").read_text(encoding="utf-8"))
    assert manifest["trellis"] is True
    assert manifest["n_views"] == 8
    scenes = json.loads((tmp_path / "data" / "speaker" / "sisters" / "scenes.json").read_text(encoding="utf-8"))
    assert scenes["trellis"] is True
    assert len(scenes["scenes"]) == 3


def test_policy_block_stops_after_the_package(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mesh = tmp_path / "scan.ply"
    trimesh.creation.box(extents=(0.1, 0.04, 0.03)).export(mesh)
    code = run("speaker", through="B16", root=tmp_path, source=mesh, n_gaussians=16, n_views=2)
    assert code == 3
    assert (tmp_path / "data" / "speaker" / "scene.usdz").is_file()


def _stage_physics_inputs(tmp_path: Path, obj: str = "cup") -> None:
    mesh_path = tmp_path / "mesh.obj"
    trimesh.creation.box(extents=(0.2, 0.1, 0.05)).export(mesh_path)
    splat = tmp_path / "data" / obj / "colmap" / "orbit1" / "splat.ply"
    splat.parent.mkdir(parents=True, exist_ok=True)
    count = 3
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\nend_header\n"
    splat.write_bytes(header.encode("ascii") + np.zeros(count * 17, dtype="<f4").tobytes())
    view = tmp_path / "data" / obj / "renders" / "synthetic" / "view_000.png"
    view.parent.mkdir(parents=True, exist_ok=True)
    view.write_bytes(b"\x89PNG\r\n\x1a\n")
    doc = load_status(obj, tmp_path)
    set_path(doc, "step02.orbit1.mesh", str(mesh_path))
    set_path(doc, "step02.orbit1.splat", str(splat))
    save_status(obj, doc, tmp_path)


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_physics_writes_vlm_fields_from_a_mocked_vision_call(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stage_physics_inputs(tmp_path)
    seen = {}

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        seen["model"] = body["model"]
        seen["has_image"] = "data:image/png;base64," in json.dumps(body)
        seen["authorization"] = request.get_header("Authorization")
        content = {
            "material": "plastic",
            "friction": 0.35,
            "rigidity": 0.7,
            "mass_kg": 0.42,
            "density_kg_m3": 900.0,
        }
        payload = {"choices": [{"message": {"content": json.dumps(content)}}]}
        return _Response(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("tools.mesh_stages.urllib.request.urlopen", fake_urlopen)
    _physics("cup", tmp_path)
    assert seen["model"] == "gpt-6-astra"
    assert seen["model"] == _VISION_MODEL
    assert seen["has_image"] is True
    assert seen["authorization"] == "Bearer test-key"
    record = json.loads((tmp_path / "data" / "cup" / "physics.json").read_text(encoding="utf-8"))
    assert record["source"] == "vlm"
    assert record["model"] == "gpt-6-astra"
    assert record["vlm"] is True
    assert record["per_gaussian"]["copied_from"] == "object"
    assert record["per_gaussian"]["learned_prior"] is False
    assert record["per_gaussian"]["variation"] == "geometry"
    assert record["gaps"]["learned_prior"].startswith("not released")
    assert record["gaps"]["vlm_review_of_later_steps"] == "not run"
    assert (tmp_path / "data" / "cup" / "physics_gaussians.npz").is_file()
    assert record["material"] == "plastic"
    assert record["friction"] == 0.35
    assert record["rigidity"] == 0.7
    assert record["mass_kg"] == 0.42
    assert record["density_kg_m3"] == 900.0
    doc = load_status("cup", tmp_path)
    assert next(step for step in doc["steps"] if step["id"] == "B10")["actual"]["source"] == "vlm"


def test_physics_keeps_the_prior_table_when_the_vision_call_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stage_physics_inputs(tmp_path)

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            404,
            "not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("tools.mesh_stages.urllib.request.urlopen", fake_urlopen)
    _physics("cup", tmp_path)
    record = json.loads((tmp_path / "data" / "cup" / "physics.json").read_text(encoding="utf-8"))
    assert record["source"] == "prior_table"
    assert record["vlm"] is False
