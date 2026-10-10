"""The change log lists block names and Harvard status words."""

from __future__ import annotations

import json
from pathlib import Path

from tools.changelog import write_changelog
from tools.common.status import load_status, save_status, set_path


def test_changelog_lists_block_names_and_status(tmp_path: Path):
    folder = tmp_path / "data" / "scan"
    folder.mkdir(parents=True)
    (folder / "physics.json").write_text(
        json.dumps(
            {
                "source": "vlm",
                "model": "gpt-6-astra",
                "material": "aluminum alloy and steel",
                "friction": 0.5,
                "rigidity": 0.98,
                "mass_kg": 2.8,
                "density_kg_m3": 3500.0,
                "per_gaussian": {"copied_from": "object", "n": 12, "learned_prior": False},
            }
        ),
        encoding="utf-8",
    )
    doc = load_status("scan", tmp_path)
    set_path(
        doc,
        "input",
        {"kind": "polycam_mesh", "cameras": False, "capture_images": False},
    )
    set_path(doc, "inspect", {"isaac": False})
    save_status("scan", doc, tmp_path)
    paper = {
        "object": "scan",
        "paper": "Hollis, Fan, Yang 2026",
        "blocks": [
            {
                "id": "input",
                "paper_requirement": "Mesh-only input.",
                "artifact": "status.json input",
                "match": True,
                "note": "polycam mesh without photos.",
            },
            {
                "id": "physics",
                "paper_requirement": "Physics copied onto Gaussians.",
                "artifact": "physics.json",
                "match": "partial",
                "note": "Object-level vision model.",
            },
            {
                "id": "policy",
                "paper_requirement": "Robot policy.",
                "artifact": "status.json B15 B16 B17",
                "match": False,
                "note": "Not run.",
            },
        ],
    }
    (folder / "paper_check.json").write_text(json.dumps(paper), encoding="utf-8")
    path = write_changelog("scan", tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path == folder / "changelog.md"
    for name in ("input", "physics", "policy"):
        assert name in text
    for word in ("Achieved", "Partial", "Not yet"):
        assert word in text
    assert "friction=0.5" in text
    assert "copied_onto=12" in text
    assert "sk-" not in text
    assert text.index("**policy**") < text.index("**physics**") < text.index("**input**")
    assert "SCENEAGENT-RUN-STEP" in text
    assert "Git commit:" in text


def test_skipped_alignment_is_not_achieved(tmp_path: Path):
    folder = tmp_path / "data" / "scan"
    folder.mkdir(parents=True)
    (folder / "align.json").write_text(
        json.dumps({"photometric": "skipped", "scale_m": 0.2}),
        encoding="utf-8",
    )
    (folder / "orientation.json").write_text(
        json.dumps({"skipped": True}),
        encoding="utf-8",
    )
    doc = load_status("scan", tmp_path)
    set_path(doc, "input", {"kind": "polycam_mesh", "cameras": False, "capture_images": False})
    save_status("scan", doc, tmp_path)
    paper = {
        "blocks": [
            {"id": "alignment", "match": True, "artifact": "align.json", "note": "skipped photometric"},
            {"id": "orientation", "match": True, "artifact": "orientation.json", "note": "skipped"},
        ]
    }
    (folder / "paper_check.json").write_text(json.dumps(paper), encoding="utf-8")
    text = write_changelog("scan", tmp_path).read_text(encoding="utf-8")
    assert "**alignment** — Achieved" not in text
    assert "**alignment** — Partial" in text
    assert "**orientation** — Achieved" not in text
    assert "**orientation** — Not yet" in text
