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


def test_asked_false_is_not_achieved(tmp_path: Path):
    folder = tmp_path / "data" / "scan"
    folder.mkdir(parents=True)
    reviews = folder / "reviews"
    reviews.mkdir()
    (reviews / "inspect_1.json").write_text(
        json.dumps({"step": "inspect", "asked": False}),
        encoding="utf-8",
    )
    doc = load_status("scan", tmp_path)
    set_path(doc, "input", {"kind": "polycam_mesh", "cameras": False, "capture_images": False})
    save_status("scan", doc, tmp_path)
    paper = {
        "blocks": [
            {"id": "inspector", "match": True, "artifact": "inspect/frame_000.png", "note": "asked false"},
            {"id": "input", "match": True, "artifact": "status.json input", "note": "mesh"},
        ]
    }
    (folder / "paper_check.json").write_text(json.dumps(paper), encoding="utf-8")
    text = write_changelog("scan", tmp_path).read_text(encoding="utf-8")
    assert "**inspector** — Achieved" not in text
    assert "**inspector** — Not yet" in text


def test_closed_changelog_section_is_not_rewritten(tmp_path: Path):
    from tools.repo_changelog import ChangelogError, append_unreleased, close_block, validate_changelog

    path = tmp_path / "CHANGELOG.md"
    path.write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [block-01] 2026-10-10 — Ingest\n"
        "### Added\n- tools/iterate.py\n### Milestone\n- bottle: registered\n### Tests\n- pytest: 1 passed\n",
        encoding="utf-8",
    )
    closed = path.read_text(encoding="utf-8").split("## [block-01]", 1)[1]
    append_unreleased(path, "Fixed", "tools/common/status.py: append runs.jsonl")
    assert path.read_text(encoding="utf-8").split("## [block-01]", 1)[1] == closed
    assert "tools/common/status.py" in path.read_text(encoding="utf-8").split("## [block-01]", 1)[0]

    fresh = tmp_path / "close.md"
    fresh.write_text("# Changelog\n\n## [Unreleased]\n\n### Added\n- tools/common/vlm.py\n", encoding="utf-8")
    close_block(
        fresh,
        "block-09",
        "2026-10-14",
        "Package and Isaac inspection",
        ["camera: frame saved"],
        ["pytest: 5 passed"],
        artifact_present=True,
    )
    closed_text = fresh.read_text(encoding="utf-8")
    assert closed_text.split("## [block-09]", 1)[0].split("## [Unreleased]", 1)[1].strip() == ""
    validate_changelog(closed_text, artifact_present=True)
    try:
        validate_changelog(
            "## [block-09] 2026-10-14 — Inspect\n### Milestone\n- achieved\n### Tests\n- pytest: 1 passed\n",
            artifact_present=False,
        )
    except ChangelogError as exc:
        assert "artifact" in str(exc).lower()
    else:
        raise AssertionError("expected ChangelogError")


def test_changelog_entry_appended_per_run(tmp_path: Path):
    from tools.common.status import append_run, run_timestamp

    entry = {
        "ts": run_timestamp(),
        "object": "bottle",
        "skill": "SCENEAGENT-RUN-STEP",
        "try": 1,
        "host": "laptop",
        "outputs": ["masks/object.json"],
        "measure_gate": {"name": "projection", "pass": True},
        "vision_gate": {"asked": False},
        "correction": None,
        "kept": False,
    }
    path = append_run("bottle", entry, tmp_path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["vision_gate"]["asked"] is False
    assert saved["kept"] is False
    assert not (tmp_path / "data" / "bottle" / "status.json").exists()

