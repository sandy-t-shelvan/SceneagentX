"""Block C crop, hole masks, and M3. A skip is not a pass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.block_gates import infill_state
from tools.block_c import (
    apply_seam_policy,
    crop_decision,
    hole_mask,
    m3_met,
    ready_orbits,
    run_c1,
    run_c2,
    run_c3,
    score_disk_stages,
    write_hole_mask,
    write_hole_masks,
)


def test_missing_splat_writes_nothing(tmp_path: Path):
    mask_dir = tmp_path / "data" / "cup" / "masks" / "orbit1"
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / "gaussian_in_mask.npy", np.array([True, False]))
    row = run_c1("cup", "orbit1", tmp_path)
    folder = tmp_path / "data" / "cup"
    assert row["state"] == "skipped"
    assert row["infill"] == "skipped"
    assert row["m3_met"] is False
    assert row["object_ply"] is None
    assert row["world_T_object"] is None
    assert not any(folder.rglob("*.ply"))
    assert not any(folder.rglob("world_T_object.json"))
    assert not any(path.name.endswith("_hole.npy") for path in folder.rglob("*"))
    assert not any(folder.rglob("*.png"))
    holes = write_hole_masks(folder / "background" / "orbit1" / "holes", [])
    assert holes["status"] == "not_run"
    assert holes["n_masks"] == 0
    assert not (folder / "background").exists()
    scored = score_disk_stages("cup", "orbit1", tmp_path)
    assert scored["state"] == "skipped"
    assert not (folder / "background" / "orbit1" / "background_infilled.ply").exists()
    saved = json.loads((folder / "ingest" / "block_c.json").read_text(encoding="utf-8"))
    assert saved["m3_met"] is False
    assert saved["orbits"]["orbit1"]["infill"] == "skipped"


def test_hole_mask_follows_alpha_drop(tmp_path: Path):
    full = np.array([[0.90, 0.90, 0.20], [0.40, 0.20, 0.80]], dtype=np.float64)
    back = np.array([[0.90, 0.20, 0.00], [0.50, 0.20, 0.79]], dtype=np.float64)
    expected = np.array([[False, True, True], [False, False, True]])
    mask = hole_mask(full, back)
    assert mask is not None
    assert np.array_equal(mask, expected)
    assert hole_mask(full, back[:, :1]) is None
    bad = full.copy()
    bad[0, 0] = np.nan
    assert hole_mask(bad, back) is None
    assert write_hole_mask(tmp_path / "bad.npy", bad, back) is None
    assert not (tmp_path / "bad.npy").exists()
    refused = write_hole_masks(tmp_path / "holes", [("frame", full, back[:, :1])])
    assert refused["status"] == "not_run"
    assert not (tmp_path / "holes").exists()
    written = write_hole_masks(tmp_path / "holes", [("frame", full, back)])
    assert written["status"] == "ran"
    assert written["n_masks"] == 1
    saved = np.load(tmp_path / "holes" / "frame_hole.npy")
    assert np.array_equal(saved, expected)
    empty = run_c2("vase", "orbit1", tmp_path)
    assert empty["holes"]["status"] == "not_run"
    assert empty["holes"]["n_masks"] == 0
    assert empty["m3_met"] is False
    assert not any((tmp_path / "data" / "vase").rglob("*_hole.npy"))


def test_c3_without_lama_is_not_run(tmp_path: Path):
    background = np.zeros((4, 4, 3), dtype=np.uint8)
    masks = [np.array([[True, False], [False, True], [False, False], [True, True]])]
    row = run_c3("cup", "orbit2", tmp_path, background=background, masks=masks, loader=lambda: None)
    assert row["infill"] == "not_run"
    assert row["fine_tune"] == "not_run"
    assert row["n_inpainted"] == 0
    assert row["m3_met"] is False
    assert row["infill"] != "pass"
    assert infill_state(row) != "pass"
    folder = tmp_path / "data" / "cup"
    assert not any(folder.rglob("*.png"))
    assert not any(folder.rglob("background_infilled.ply"))


def test_infill_and_finetune_stay_not_run_without_real_images(tmp_path: Path):
    _write_pair(tmp_path, "widget", "orbit1")
    cropped = run_c1("widget", 1, tmp_path)
    assert cropped["state"] == "partial"
    assert cropped["infill"] == "not_run"
    assert cropped["fine_tune"] == "not_run"
    scored = score_disk_stages("widget", "orbit1", tmp_path)
    assert scored["infill"] == "not_run"
    assert scored["fine_tune"] == "not_run"
    assert scored["m3_met"] is False
    background = tmp_path / "data" / "widget" / "background" / "orbit1"
    assert not (background / "background_infilled.ply").exists()
    fake = background / "inpaint" / "frame.png"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"\x89PNG\r\n\x1a\n")
    again = score_disk_stages("widget", "orbit1", tmp_path)
    assert again["infill"] == "not_run"
    assert again["fine_tune"] == "not_run"
    assert again["n_inpainted"] == 0
    assert not (background / "background_infilled.ply").exists()


def test_m3_met_requires_infill_finetune_floaters_psnr_and_seam():
    good = {
        "infill": "ran",
        "fine_tune": "ran",
        "floater_count": 49,
        "psnr_drop_db": 1.49,
        "seam_second_fail": False,
        "vlm": {"asked": True, "verdict": "pass", "seams": False},
    }
    assert m3_met(good) is True
    cases = {
        "infill": "not_run",
        "fine_tune": "not_run",
        "floater_count": 50,
        "psnr_drop_db": 1.5,
        "seam_second_fail": True,
    }
    for key, value in cases.items():
        row = dict(good)
        row[key] = value
        assert m3_met(row) is False
    unanswered = dict(good)
    unanswered["vlm"] = {"asked": False, "verdict": None, "seams": None}
    assert m3_met(unanswered) is False
    seams = dict(good)
    seams["vlm"] = {"asked": True, "verdict": "fail", "seams": True}
    assert m3_met(seams) is False
    skipped = dict(good)
    skipped["infill"] = "skipped"
    assert m3_met(skipped) is False


def test_second_seam_fail_is_recorded():
    calls = {"ask": 0, "rerun": 0}

    def ask(retry: bool) -> dict:
        del retry
        calls["ask"] += 1
        return {"asked": True, "verdict": "fail", "seams": True, "problem": "seam"}

    def rerun(wider: bool) -> dict:
        assert wider is True
        calls["rerun"] += 1
        return {
            "infill": "ran",
            "fine_tune": "ran",
            "psnr_before": 30.0,
            "psnr_after": 29.0,
            "psnr_drop_db": 1.0,
        }

    row = apply_seam_policy(
        {
            "infill": "ran",
            "fine_tune": "ran",
            "floater_count": 10,
            "psnr_drop_db": 0.4,
            "psnr_before": 30.0,
            "psnr_after": 29.6,
        },
        ask,
        rerun,
    )
    assert calls == {"ask": 2, "rerun": 1}
    assert row["seam_attempts"] == 2
    assert row["seam_second_fail"] is True
    assert row["m3_met"] is False
    assert row["vlm"]["seams"] is True


def test_flat_mask_is_only_orbit1(tmp_path: Path):
    folder = tmp_path / "data" / "bottle"
    for orbit in ("orbit1", "orbit2"):
        orbit_dir = folder / "colmap" / orbit
        orbit_dir.mkdir(parents=True)
        (orbit_dir / "splat.ply").write_bytes(b"ply\n")
    mask_dir = folder / "masks"
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / "gaussian_in_mask.npy", np.array([True, False]))
    assert ready_orbits("bottle", tmp_path) == ["orbit1"]


def test_wide_anchor_loses_to_a_compact_crop():
    from tools.block_c import choose_crop_policy

    chosen = choose_crop_policy(
        [
            {"name": "or1_mad8", "iou": 0.4009, "recall": 0.9852, "n_inside": 44598, "splat_frac": 0.690, "covers_room": True},
            {"name": "or1_mad3", "iou": 0.6189, "recall": 0.9851, "n_inside": 28783, "splat_frac": 0.445, "covers_room": False},
            {"name": "or2_mad3", "iou": 0.6189, "recall": 0.9851, "n_inside": 28783, "splat_frac": 0.445, "covers_room": False},
            {"name": "anchor_only", "iou": 0.8192, "recall": 0.9928, "n_inside": 32366, "splat_frac": 0.282, "covers_room": True},
        ]
    )
    assert chosen == "or1_mad3"


def test_compact_anchor_stays_when_it_does_not_cover_the_room():
    from tools.block_c import choose_crop_policy

    chosen = choose_crop_policy(
        [
            {"name": "or1_mad3", "iou": 0.887, "recall": 0.927, "n_inside": 26334, "splat_frac": 0.331, "covers_room": False},
            {"name": "anchor_only", "iou": 0.629, "recall": 1.0, "n_inside": 32786, "splat_frac": 0.343, "covers_room": False},
        ]
    )
    assert chosen == "anchor_only"


def test_every_eligible_room_crop_returns_no_crop_and_skips_the_mask(tmp_path: Path):
    from tools.block_c import apply_crop_policy, choose_crop_policy

    rows = [
        {"name": "wide", "iou": 0.91, "recall": 0.991, "n_inside": 40000, "splat_frac": 0.70, "covers_room": True, "object_span": 12.0},
        {"name": "mid", "iou": 0.72, "recall": 0.985, "n_inside": 22000, "splat_frac": 0.40, "covers_room": True, "object_span": 10.0},
        {"name": "low_recall", "iou": 0.95, "recall": 0.50, "n_inside": 8000, "splat_frac": 0.20, "covers_room": False, "object_span": 1.0},
    ]
    assert choose_crop_policy(rows) is None
    folder = tmp_path / "data" / "mug"
    mask_dir = folder / "masks" / "orbit1"
    mask_dir.mkdir(parents=True)
    original = np.array([True, False, True, False])
    mask = mask_dir / "gaussian_in_mask.npy"
    np.save(mask, original)
    before = mask.read_bytes()
    stamp = mask.stat().st_mtime_ns
    (mask_dir / "object.json").write_text(json.dumps({"prompt": "mug on a table"}) + "\n", encoding="utf-8")
    (folder / "request.json").write_text(
        json.dumps({"job": "room", "prompt": "mug on a table", "target": "mug"}),
        encoding="utf-8",
    )
    assert apply_crop_policy(folder, "orbit1", rows, span_scene=20.0) is None
    assert mask.read_bytes() == before
    assert mask.stat().st_mtime_ns == stamp
    assert not np.array_equal(np.load(mask), np.ones(4, dtype=bool))
    record = json.loads((mask_dir / "object.json").read_text(encoding="utf-8"))
    assert record["prompt"] == "mug on a table"
    assert record["covers_room"] is True
    assert record["achieved"] is False
    assert record["infill"] == "not_run"
    assert record["policy"] is None
    assert record["span_inside"] == 12.0
    assert record["span_scene"] == 20.0
    assert not (folder / "semantics").exists()
    assert not list(folder.rglob("*.ply"))
    assert not list(folder.rglob("*feature*"))


def test_orbit_job_still_names_a_crop_when_every_row_covers_the_room(tmp_path: Path):
    from tools.block_c import apply_crop_policy, choose_crop_policy

    rows = [
        {"name": "wide", "iou": 0.91, "recall": 0.991, "n_inside": 40000, "splat_frac": 0.70, "covers_room": True, "object_span": 12.0},
        {"name": "mid", "iou": 0.72, "recall": 0.985, "n_inside": 22000, "splat_frac": 0.40, "covers_room": True, "object_span": 10.0},
    ]
    assert choose_crop_policy(rows) is None
    folder = tmp_path / "data" / "mug"
    folder.mkdir(parents=True)
    (folder / "request.json").write_text(
        json.dumps({"job": "orbit", "prompt": "orbit a mug", "target": "mug"}),
        encoding="utf-8",
    )
    assert apply_crop_policy(folder, "orbit1", rows, span_scene=20.0) == "wide"
    assert not (folder / "masks" / "orbit1" / "gaussian_in_mask.npy").exists()
    assert not (folder / "masks" / "orbit1" / "object.json").exists()


def test_room_covering_mask_is_not_a_crop(tmp_path: Path):
    _write_pair(tmp_path, "can", "orbit2")
    folder = tmp_path / "data" / "can"
    _write_lift(
        folder,
        "orbit2",
        {"covers_room": True, "span_inside": 9.781726, "span_scene": 20.435517, "n_inside": 3, "n_outside": 1},
    )
    _write_instances(folder, "orbit2")
    semantics = folder / "semantics" / "orbit2"
    semantics.mkdir(parents=True, exist_ok=True)
    (semantics / "feature_field_refused.json").write_text(
        json.dumps({"feature_field": "refused", "reason": "mask lift covers the room"}),
        encoding="utf-8",
    )
    measured = semantics / "feature_field.json"
    measured.write_text(json.dumps({"feature_field": "measured", "achieved": False}), encoding="utf-8")
    planted = folder / "objects" / "can" / "orbit2"
    planted.mkdir(parents=True)
    (planted / "object.ply").write_bytes(b"ply\nfake")
    (planted / "world_T_object.json").write_text("{}\n", encoding="utf-8")
    background = folder / "background" / "orbit2"
    background.mkdir(parents=True)
    (background / "background.ply").write_bytes(b"ply\nfake")
    row = run_c1("can", "orbit2", tmp_path)
    assert row["crop"] == "refused"
    assert row["state"] == "measured"
    assert row["achieved"] is False
    assert row["m3_met"] is False
    assert row["infill"] == "not_run"
    assert row["infill"] != "pass"
    assert row["n_inside"] == 3
    assert row["n_outside"] == 1
    assert row["feature_field"] == "refused"
    assert row["instance_name"] == "can"
    assert row["block_b"]["mask"].endswith("gaussian_in_mask.npy")
    assert row["block_b"]["instances"].endswith("instances.json")
    assert not (planted / "object.ply").exists()
    assert not (planted / "world_T_object.json").exists()
    assert not (background / "background.ply").exists()
    assert not (background / "background_infilled.ply").exists()
    assert "measured" in measured.read_text(encoding="utf-8")
    kept = score_disk_stages("can", "orbit2", tmp_path)
    assert kept["crop"] == "refused"
    assert kept["achieved"] is False
    painted = run_c3("can", "orbit2", tmp_path, background=np.zeros((2, 2, 3), dtype=np.uint8), masks=[np.ones((2, 2), dtype=bool)])
    assert painted["crop"] == "refused"
    assert painted["infill"] == "not_run"
    assert not any(folder.rglob("*.png"))
    half = {"covers_room": False, "span_inside": 5.0, "span_scene": 10.0}
    assert crop_decision(folder, half, 4) == "refused"


def test_object_only_infill_stays_skipped(tmp_path: Path):
    _write_pair(tmp_path, "bottle", "orbit1")
    folder = tmp_path / "data" / "bottle"
    _write_lift(
        folder,
        "orbit1",
        {
            "covers_room": False,
            "span_inside": 1.6,
            "span_scene": 22.5,
            "infill_reason": "no separate background fill; the room was not infilled",
        },
    )
    row = run_c1("bottle", "orbit1", tmp_path)
    assert row["crop"] == "object_only"
    assert row["state"] == "measured"
    assert row["infill"] == "skipped"
    assert row["achieved"] is False
    assert row["m3_met"] is False
    assert row["n_inside"] == 3
    assert row["n_outside"] == 1
    assert (folder / "objects" / "bottle" / "orbit1" / "object.ply").is_file()
    kept = score_disk_stages("bottle", "orbit1", tmp_path)
    assert kept["infill"] == "skipped"
    assert kept["infill"] != "pass"
    assert kept["m3_met"] is False


def test_room_object_mask_requires_infill(tmp_path: Path):
    _write_pair(tmp_path, "can", "orbit1")
    folder = tmp_path / "data" / "can"
    _write_lift(folder, "orbit1", {"covers_room": False, "span_inside": 0.87, "span_scene": 9.29, "n_inside": 3, "n_outside": 1})
    _write_instances(folder, "orbit1")
    row = run_c1("can", "orbit1", tmp_path)
    assert row["crop"] == "room"
    assert row["state"] == "partial"
    assert row["infill"] == "not_run"
    assert row["infill_required"] is True
    assert row["achieved"] is False
    assert row["m3_met"] is False
    assert row["n_object"] is not None
    assert row["n_background"] is not None
    assert row["n_inside"] == 3
    assert row["n_outside"] == 1
    assert (folder / "objects" / "can" / "orbit1" / "object.ply").is_file()
    assert (folder / "background" / "orbit1" / "background.ply").is_file()
    assert "not a pass" in row["note"]


def test_host_script_does_not_render_a_room_mask(tmp_path: Path):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "vast-cluster" / "scripts" / "object_block_c.py"
    spec = importlib.util.spec_from_file_location("object_block_c_room", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _write_pair(tmp_path, "can", "orbit2")
    folder = tmp_path / "data" / "can"
    _write_lift(folder, "orbit2", {"covers_room": True, "span_inside": 9.78, "span_scene": 20.44})
    code = module.run_orbit("can", "orbit2", tmp_path, "all", False, 1)
    assert code == 0
    stage = json.loads((folder / "background" / "orbit2" / "stage.json").read_text(encoding="utf-8"))
    assert stage["crop"] == "refused"
    assert stage["infill"] == "not_run"
    assert stage["achieved"] is False
    assert not (folder / "background" / "orbit2" / "background_infilled.ply").exists()


def _write_lift(folder: Path, orbit: str, payload: dict) -> None:
    path = folder / "masks" / orbit / "object.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_instances(folder: Path, orbit: str) -> None:
    path = folder / "semantics" / orbit / "instances.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "instances": [{"name": "can", "foreground": True, "n_gaussians": 3}],
                "block_c_ready": False,
                "review": {"verdict": "fail", "problem": "extra_part"},
            }
        ),
        encoding="utf-8",
    )


def test_orbit_job_skips_the_half_scene_test(tmp_path: Path):
    _write_pair(tmp_path, "marker", "orbit1")
    folder = tmp_path / "data" / "marker"
    _write_request(folder, "orbit", "This is an orbit video of G.")
    lift = {"covers_room": True, "span_inside": 5.0, "span_scene": 10.0, "n_inside": 3, "n_outside": 1}
    _write_lift(folder, "orbit1", lift)
    assert crop_decision(folder, lift, 4) == "object_only"
    assert crop_decision(folder, {"covers_room": False, "span_inside": 0.2, "span_scene": 4.0, "view_fractions": [0.93]}, 4) == "object_only"
    row = run_c1("marker", "orbit1", tmp_path)
    assert row["crop"] == "object_only"
    assert row["infill"] == "skipped"
    assert row["infill_required"] is False
    assert row["achieved"] is False
    assert (folder / "objects" / "marker" / "orbit1" / "object.ply").is_file()
    assert not (folder / "background" / "orbit1" / "background_infilled.ply").exists()


def test_first_room_sized_mask_is_not_cropped_and_a_smaller_retry_is(tmp_path: Path):
    _write_pair(tmp_path, "kitchen", "orbit1")
    folder = tmp_path / "data" / "kitchen"
    _write_request(folder, "room", "a room; I want sisters of Y")
    room = {"covers_room": True, "span_inside": 9.78, "span_scene": 20.44, "n_inside": 3, "n_outside": 1}
    _write_lift(folder, "orbit1", room)
    (folder / "mask_retry.json").write_text(
        json.dumps({"detection": "pending", "orbits": ["orbit1"], "params": {"box_threshold": 0.3}, "retrain": False}),
        encoding="utf-8",
    )
    planted = folder / "objects" / "kitchen" / "orbit1"
    planted.mkdir(parents=True)
    (planted / "object.ply").write_bytes(b"ply\nfake")
    held = run_c1("kitchen", "orbit1", tmp_path)
    assert held["crop"] == "refused"
    assert held["infill"] == "not_run"
    assert held["feature_field"] == "unstarted"
    assert "pending" in held["note"]
    assert not (planted / "object.ply").exists()
    assert not list(folder.rglob("feature_field.json"))
    small = {"covers_room": False, "span_inside": 0.4, "span_scene": 10.0, "n_inside": 3, "n_outside": 1}
    _write_lift(folder, "orbit1", small)
    (folder / "mask_retry.json").write_text(
        json.dumps({"detection": "rerun", "orbits": ["orbit1"], "params": {"box_threshold": 0.3}, "retrain": False}),
        encoding="utf-8",
    )
    assert crop_decision(folder, small, 1) == "room"
    cropped = run_c1("kitchen", "orbit1", tmp_path)
    assert cropped["crop"] == "room"
    assert cropped["infill"] == "skipped"
    assert cropped["infill_required"] is False
    assert (folder / "objects" / "kitchen" / "orbit1" / "object.ply").is_file()
    assert (folder / "background" / "orbit1" / "background.ply").is_file()

    def paint():
        raise AssertionError("infill ran")

    painted = run_c3(
        "kitchen",
        "orbit1",
        tmp_path,
        background=np.zeros((2, 2, 3), dtype=np.uint8),
        masks=[np.ones((2, 2), dtype=bool)],
        loader=paint,
    )
    assert painted["infill"] == "skipped"
    assert not any(folder.rglob("*.png"))


def test_still_room_sized_lift_stays_refused_after_the_retry(tmp_path: Path):
    _write_pair(tmp_path, "kitchen", "orbit1")
    folder = tmp_path / "data" / "kitchen"
    _write_request(folder, "room", "deliver the room with the object removed", deliverable="object_removed")
    room = {"covers_room": True, "span_inside": 9.78, "span_scene": 20.44, "n_inside": 3, "n_outside": 1}
    _write_lift(folder, "orbit1", room)
    (folder / "mask_retry.json").write_text(
        json.dumps({"detection": "rerun", "orbits": ["orbit1"], "params": {"box_threshold": 0.3}, "retrain": False}),
        encoding="utf-8",
    )
    row = run_c1("kitchen", "orbit1", tmp_path)
    assert row["crop"] == "refused"
    assert row["infill"] == "not_run"
    assert "9.78" in row["note"]
    assert "20.44" in row["note"]
    assert "not repeated" in row["note"]
    assert not (folder / "objects" / "kitchen" / "orbit1" / "object.ply").exists()
    assert not (folder / "background" / "orbit1" / "background.ply").exists()


def test_pending_mask_retry_refuses_crop_before_mesh(tmp_path: Path):
    _write_pair(tmp_path, "kitchen", "orbit1")
    folder = tmp_path / "data" / "kitchen"
    _write_request(folder, "room", "Here is the room. I want the can.")
    _write_lift(folder, "orbit1", {"covers_room": False, "span_inside": 0.4, "span_scene": 10.0, "n_inside": 3, "n_outside": 1})
    (folder / "mask_retry.json").write_text(
        json.dumps({"detection": "pending", "orbits": ["orbit1"], "params": {"box_threshold": 0.3}, "retrain": False}),
        encoding="utf-8",
    )
    assert crop_decision(folder, {"covers_room": False, "span_inside": 0.4, "span_scene": 10.0}, 1) == "refused"
    row = run_c1("kitchen", "orbit1", tmp_path)
    assert row["crop"] == "refused"
    assert "pending" in row["note"]
    assert not (folder / "objects" / "kitchen" / "orbit1" / "object.ply").exists()


def test_object_removed_spec_still_requires_infill(tmp_path: Path):
    _write_pair(tmp_path, "kitchen", "orbit1")
    folder = tmp_path / "data" / "kitchen"
    _write_request(folder, "room", "deliver the room with the object removed", deliverable="object_removed")
    _write_lift(folder, "orbit1", {"covers_room": False, "span_inside": 0.4, "span_scene": 10.0, "n_inside": 3, "n_outside": 1})
    row = run_c1("kitchen", "orbit1", tmp_path)
    assert row["crop"] == "room"
    assert row["infill_required"] is True
    assert row["infill"] == "not_run"
    assert (folder / "background" / "orbit1" / "background.ply").is_file()


def test_frame_covering_view_is_not_cropped(tmp_path: Path):
    from PIL import Image

    _write_pair(tmp_path, "kitchen", "orbit1")
    folder = tmp_path / "data" / "kitchen"
    _write_request(folder, "room", "a room; I want sisters of Y")
    _write_lift(folder, "orbit1", {"covers_room": False, "span_inside": 0.2, "span_scene": 4.0, "n_inside": 3, "n_outside": 1})
    cover = np.zeros((10, 10), dtype=np.uint8)
    cover.ravel()[:90] = 255
    Image.fromarray(cover, mode="L").save(folder / "masks" / "orbit1" / "frame_mask.png")
    assert crop_decision(folder, {"covers_room": False, "span_inside": 0.2, "span_scene": 4.0, "view_fractions": [0.89]}, 4) == "room"
    row = run_c1("kitchen", "orbit1", tmp_path)
    assert row["crop"] == "refused"
    assert row["infill"] == "not_run"
    assert not (folder / "objects" / "kitchen" / "orbit1" / "object.ply").exists()


def _write_request(folder: Path, job: str, prompt: str, deliverable: str = "sisters_of_target") -> None:
    (folder / "request.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "job": job,
                "target": "Y",
                "seen": ["X", "Y", "Z"],
                "deliverable": deliverable,
            }
        ),
        encoding="utf-8",
    )


def _write_pair(root: Path, obj: str, orbit: str) -> None:
    rows = np.array(
        [
            [0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0, 8.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, -8.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\nproperty float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\nend_header\n"
    )
    splat = root / "data" / obj / "colmap" / orbit / "splat.ply"
    splat.parent.mkdir(parents=True, exist_ok=True)
    splat.write_bytes(header.encode("ascii") + rows.tobytes())
    mask = root / "data" / obj / "masks" / orbit / "gaussian_in_mask.npy"
    mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask, np.array([True, True, True, False]))
