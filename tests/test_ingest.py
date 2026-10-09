"""Polycam mesh ingest: no COLMAP, a surface splat, and synthetic views."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from tools.common.status import load_status
from tools.ingest import InputError, classify_input
from tools.pipeline import run
from tools.step02_splatfacto import count_gaussians


def _colored_box(extents: tuple[float, float, float] = (0.12, 0.05, 0.04)) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.visual.vertex_colors = np.tile(np.array([220, 40, 40, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def _write_formats(directory: Path, extents: tuple[float, float, float] = (0.12, 0.05, 0.04)) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    mesh = _colored_box(extents)
    written = {}
    for suffix in (".obj", ".ply", ".glb"):
        path = directory / f"scan{suffix}"
        mesh.export(path)
        written[suffix] = path
    gltf = directory / "scan.gltf"
    mesh.export(gltf)
    written[".gltf"] = gltf
    return written


def test_classify_video_and_mesh(tmp_path: Path):
    video = tmp_path / "orbit1.mp4"
    video.write_bytes(b"not a real video")
    assert classify_input(video)["kind"] == "video"

    mesh = tmp_path / "alone" / "scan.obj"
    mesh.parent.mkdir()
    _colored_box().export(mesh)
    record = classify_input(mesh)
    assert record["kind"] == "polycam_mesh"
    assert record["cameras"] is False
    assert record["capture_images"] is False


def test_mesh_beside_cameras_is_refused(tmp_path: Path):
    folder = tmp_path / "scan"
    folder.mkdir()
    mesh = folder / "scan.ply"
    _colored_box().export(mesh)
    (folder / "cameras.json").write_text("{}", encoding="utf-8")
    try:
        classify_input(mesh)
    except InputError as exc:
        message = str(exc)
    else:
        raise AssertionError("cameras.json beside a mesh should be refused")
    assert "cameras.json" in message


def test_obj_ply_and_gltf_write_a_splat(tmp_path: Path):
    folder = tmp_path / "exports"
    written = _write_formats(folder)
    for suffix, mesh in written.items():
        code = run(
            "speaker",
            through="B1",
            root=tmp_path / suffix.strip("."),
            source=mesh,
            n_gaussians=64,
            n_views=4,
        )
        assert code == 0, suffix
        root = tmp_path / suffix.strip(".")
        splat = root / "data" / "speaker" / "colmap" / "orbit1" / "splat.ply"
        views = list((root / "data" / "speaker" / "renders" / "synthetic").glob("view_*.png"))
        doc = load_status("speaker", root)
        assert count_gaussians(splat) == 64
        assert len(views) == 4
        assert doc["input"]["kind"] == "polycam_mesh"
        assert doc["input"]["cameras"] is False
        assert doc["input"]["capture_images"] is False
        orbit = doc["step02"]["orbit1"]
        assert orbit["source"] == "mesh"
        assert orbit["photometric"] is False
        assert orbit["synthetic"] is True
        assert orbit["psnr"] is None
        assert orbit["ssim"] is None


def test_implausible_scale_requires_a_ruler(tmp_path: Path):
    mesh = tmp_path / "millimetres.obj"
    _colored_box((140.0, 40.0, 30.0)).export(mesh)
    code = run("speaker", root=tmp_path, source=mesh, n_gaussians=16, n_views=2)
    assert code == 2
    code = run(
        "speaker",
        root=tmp_path,
        source=mesh,
        length_m=0.14,
        n_gaussians=16,
        n_views=2,
    )
    assert code == 0
    orbit = load_status("speaker", tmp_path)["step02"]["orbit1"]
    assert orbit["length_m"] == 0.14


def test_pipeline_refuses_colmap_for_a_mesh(tmp_path: Path):
    mesh = tmp_path / "scan.obj"
    _colored_box().export(mesh)
    code = run("speaker", through="B2", root=tmp_path, source=mesh, n_gaussians=16, n_views=2)
    assert code == 2
    doc = load_status("speaker", tmp_path)
    assert doc["input"]["kind"] == "polycam_mesh"
    assert (tmp_path / "data" / "speaker" / "colmap" / "orbit1" / "splat.ply").is_file()
    assert next(step for step in doc["steps"] if step["id"] == "B2")["state"] == "not_started"
