"""Block B score writer. A named instance and a physics vector are not mIoU."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

from tools.common.paths import repo_root


def _report():
    path = repo_root() / "vast-cluster" / "scripts" / "object_block_b_report.py"
    spec = importlib.util.spec_from_file_location("object_block_b_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_langsplat_npz_needs_the_three_arrays(tmp_path: Path):
    report = _report()
    missing = tmp_path / "missing.npz"
    assert report.langsplat_npz_measured(missing) is False
    physics = tmp_path / "physics.npz"
    with zipfile.ZipFile(physics, "w") as archive:
        archive.writestr("object_vector.npy", b"not-a-codebook")
    assert report.langsplat_npz_measured(physics) is False
    real = tmp_path / "features.codebook.npz"
    with zipfile.ZipFile(real, "w") as archive:
        archive.writestr("codebook.npy", b"a")
        archive.writestr("clip_features.npy", b"b")
        archive.writestr("levels.npy", b"c")
    assert report.langsplat_npz_measured(real) is True


def test_score_leaves_m1_and_m2_unmet(tmp_path: Path):
    report = _report()
    folder = tmp_path / "data" / "vaseline"
    orbit = folder / "masks" / "orbit1"
    orbit.mkdir(parents=True)
    (orbit / "object.json").write_text(
        json.dumps(
            {
                "prompt": "vaseline bottle.",
                "n_views": 10,
                "n_views_with_detection": 4,
                "projected": True,
                "covers_room": True,
                "span_inside": 5.0,
                "span_scene": 8.0,
            }
        ),
        encoding="utf-8",
    )
    for name in ("orbit2", "orbit3"):
        dest = folder / "masks" / name
        dest.mkdir(parents=True)
        (dest / "object.json").write_text("{}\n", encoding="utf-8")
    code = report.main(["--obj", "vaseline", "--root", str(tmp_path)])
    assert code == 0
    score = json.loads((folder / "ingest" / "block_b.json").read_text(encoding="utf-8"))
    assert score["m1_met"] is False
    assert score["m2_met"] is False
    assert score["miou"] is None
    assert score["hand_labels"] == 0
    assert score["langsplat_measured"] is False
    row = json.loads((folder / "ingest" / "block_b_orbit1.json").read_text(encoding="utf-8"))
    parts = {part["id"]: part for part in row["parts"]}
    assert parts["B1"]["detection_rate"] == 0.4
    assert parts["B1"]["prompt"] == "vaseline bottle."
    assert parts["B3"]["feature_field"] == "unstarted"
    assert parts["B4"]["miou"] is None
    assert parts["B4"]["named_instance_is_not_miou"] is True
