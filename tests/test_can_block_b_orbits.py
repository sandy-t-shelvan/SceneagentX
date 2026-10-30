"""Can orbit 2 and 3 Block B scores. A skipped check is never a pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.can_block_b_check import check_orbit, feature_field_refusal, find_lift_span
from tools.common.paths import repo_root

LIVE_ORBIT2 = repo_root() / "data" / "can" / "ingest" / "block_b_orbit2.json"


def test_room_sized_lift_refuses_the_feature_field():
    lift = {"covers_room": True, "span_inside": 9.78, "span_scene": 20.44, "n_inside": 32366}
    assert "covers the room" in feature_field_refusal(lift)
    assert "covers the room" in feature_field_refusal(lift, job="room")
    assert feature_field_refusal(lift, job="orbit") is None
    assert feature_field_refusal({"covers_room": False, "span_inside": 0.67, "span_scene": 10.2}) is None
    assert "covers the room" in feature_field_refusal({"covers_room": False, "span_inside": 5.0, "span_scene": 10.0})
    assert feature_field_refusal({"covers_room": False, "span_inside": 5.0, "span_scene": 10.0}, job="orbit") is None
    found = find_lift_span({"crop": lift})
    assert found["n_inside"] == 32366


def test_checker_records_each_sub_step(tmp_path: Path):
    folder = _plant(tmp_path, "orbit2")
    _plant(tmp_path, "orbit3")
    record = check_orbit(folder, "orbit2")
    other = check_orbit(folder, "orbit3")
    first = check_orbit(_plant(tmp_path / "o1", "orbit1"), "orbit1")
    for step_id in ("A", "B1", "B2", "B3", "B4"):
        assert step_id in record["steps"]
        assert step_id in other["steps"]
        assert step_id in first["steps"]
        assert record["steps"][step_id]["skipped"] == []
        assert other["steps"][step_id]["skipped"] == []
        assert first["steps"][step_id]["skipped"] == []
    assert (folder / "debug" / "block_b_orbit2_check.json").is_file()
    assert (folder / "debug" / "block_b_orbit3_check.json").is_file()
    assert first["steps"]["B1"]["numbers"]["n_views_with_detection"] == 96


def test_photometry_and_full_register_without_scale_leave_m1_false(tmp_path: Path):
    folder = _plant(tmp_path, "orbit2")
    record = check_orbit(folder, "orbit2")
    assert record["steps"]["A"]["numbers"]["photometric_pass"] is True
    assert record["steps"]["A"]["numbers"]["register_pass"] is True
    assert record["steps"]["A"]["numbers"]["metres_per_unit"] is None
    assert record["m1_met"] is False
    assert record["steps"]["A"]["achieved_claimed"] is False


def test_null_miou_and_zero_hand_labels_leave_m2_false(tmp_path: Path):
    folder = _plant(tmp_path, "orbit3")
    record = check_orbit(folder, "orbit3")
    assert record["steps"]["B4"]["numbers"]["miou"] is None
    assert record["steps"]["B4"]["numbers"]["hand_labels"] == 0
    assert record["m2_met"] is False
    assert record["steps"]["B4"]["achieved_claimed"] is False


def test_missing_hand_label_set_does_not_fail_block_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    folder = _plant(tmp_path, "orbit2")
    assert not (folder / "hand_labels").exists()

    def boom(*_args, **_kwargs):
        raise AssertionError("hand labels were scored")

    monkeypatch.setattr("tools.can_hand_labels.score_object", boom)
    record = check_orbit(folder, "orbit2")
    assert record["ok"] is True
    assert record["m2_met"] is False
    assert record["achieved"] is False
    assert record["steps"]["B4"]["ok"] is True
    assert record["steps"]["B4"]["skipped"] == []


def test_measured_hand_labels_do_not_mark_m2_achieved(tmp_path: Path):
    folder = _plant(tmp_path, "orbit3")
    instances = json.loads((folder / "semantics" / "orbit3" / "instances.json").read_text(encoding="utf-8"))
    instances["miou"] = 0.91
    instances["hand_labels"] = 20
    (folder / "semantics" / "orbit3" / "instances.json").write_text(json.dumps(instances), encoding="utf-8")
    score = json.loads((folder / "ingest" / "block_b_orbit3.json").read_text(encoding="utf-8"))
    score["m2_met"] = True
    part = _part(score, "B4")
    part["empirical"]["miou"] = 0.91
    part["empirical"]["hand_labels"] = 20
    (folder / "ingest" / "block_b_orbit3.json").write_text(json.dumps(score), encoding="utf-8")
    record = check_orbit(folder, "orbit3")
    assert record["m2_met"] is False
    assert record["achieved"] is False
    assert record["steps"]["B4"]["ok"] is True


def test_claimed_m1_or_m2_without_gate_artifacts_is_mismatch(tmp_path: Path):
    folder = _plant(tmp_path, "orbit2")
    score = json.loads((folder / "ingest" / "block_b_orbit2.json").read_text(encoding="utf-8"))
    score["m1_achieved"] = True
    score["m1_met"] = True
    (folder / "ingest" / "block_b_orbit2.json").write_text(json.dumps(score), encoding="utf-8")
    m1 = check_orbit(folder, "orbit2")
    assert m1["m1_met"] is False
    assert m1["steps"]["A"]["ok"] is False
    assert any("m1" in item for item in m1["steps"]["A"]["mismatch"])

    folder3 = _plant(tmp_path / "other", "orbit3")
    score3 = json.loads((folder3 / "ingest" / "block_b_orbit3.json").read_text(encoding="utf-8"))
    score3["m2_achieved"] = True
    score3["m2_met"] = True
    (folder3 / "ingest" / "block_b_orbit3.json").write_text(json.dumps(score3), encoding="utf-8")
    m2 = check_orbit(folder3, "orbit3")
    assert m2["m2_met"] is False
    assert m2["steps"]["B4"]["ok"] is False
    assert any("m2" in item for item in m2["steps"]["B4"]["mismatch"])


def test_silhouette_iou_is_not_written_as_miou(tmp_path: Path):
    folder = _plant(tmp_path, "orbit2")
    (folder / "mesh_score.json").write_text(
        json.dumps({"orbits": [{"orbit": "orbit2", "iou": 0.8191895331100135}]}),
        encoding="utf-8",
    )
    record = check_orbit(folder, "orbit2")
    assert record["steps"]["B4"]["numbers"]["miou"] is None
    assert record["steps"]["B4"]["numbers"]["silhouette_iou"] == pytest.approx(0.8191895331100135)
    assert record["steps"]["B4"]["numbers"]["silhouette_is_not_miou"] is True

    score = json.loads((folder / "ingest" / "block_b_orbit2.json").read_text(encoding="utf-8"))
    _part(score, "B4")["empirical"]["miou"] = 0.8191895331100135
    (folder / "ingest" / "block_b_orbit2.json").write_text(json.dumps(score), encoding="utf-8")
    copied = check_orbit(folder, "orbit2")
    assert copied["steps"]["B4"]["ok"] is False
    assert any("silhouette" in item.lower() or "invented miou" in item for item in copied["steps"]["B4"]["mismatch"])
    assert copied["steps"]["B4"]["numbers"]["miou"] is None


def test_learned_prior_true_is_refused(tmp_path: Path):
    folder = _plant(tmp_path, "orbit2")
    codebook = json.loads((folder / "codebook_orbit2.json").read_text(encoding="utf-8"))
    codebook["learned_prior"] = True
    (folder / "codebook_orbit2.json").write_text(json.dumps(codebook), encoding="utf-8")
    record = check_orbit(folder, "orbit2")
    assert record["steps"]["B2"]["ok"] is False
    assert any("learned_prior" in item for item in record["steps"]["B2"]["mismatch"])
    assert record["steps"]["B2"]["achieved_claimed"] is False


def test_b1_empty_views_are_not_a_pass(tmp_path: Path):
    folder = _plant(tmp_path, "orbit2")
    record = check_orbit(folder, "orbit2")
    assert record["steps"]["B1"]["numbers"]["empty_views"] is True
    assert record["steps"]["B1"]["numbers"]["n_views_with_detection"] < record["steps"]["B1"]["numbers"]["n_views"]
    assert record["steps"]["B1"]["achieved_claimed"] is False
    assert record["steps"]["B1"]["ok"] is True

    score = json.loads((folder / "ingest" / "block_b_orbit2.json").read_text(encoding="utf-8"))
    _part(score, "B1")["achieved"] = True
    (folder / "ingest" / "block_b_orbit2.json").write_text(json.dumps(score), encoding="utf-8")
    claimed = check_orbit(folder, "orbit2")
    assert claimed["steps"]["B1"]["ok"] is False
    assert any("empty views" in item for item in claimed["steps"]["B1"]["mismatch"])


def test_b3_covers_room_kept_is_not_an_achieve(tmp_path: Path):
    folder = _plant(tmp_path, "orbit2")
    record = check_orbit(folder, "orbit2")
    numbers = record["steps"]["B3"]["numbers"]
    assert numbers["covers_room"] is True
    assert numbers["judge_projection"] == "kept"
    assert numbers["span_ratio"] == pytest.approx(0.478, abs=0.001)
    assert record["steps"]["B3"]["achieved_claimed"] is False
    assert record["achieved"] is False


def test_missing_review_png_sets_asked_false(tmp_path: Path):
    folder = _plant(tmp_path, "orbit3", reviews=False)
    record = check_orbit(folder, "orbit3")
    assert record["steps"]["B3"]["numbers"]["asked"] is False
    assert record["steps"]["B3"]["numbers"]["missing_reviews"]


def test_feature_field_measured_without_npy_is_a_fail(tmp_path: Path):
    folder = _plant(tmp_path, "orbit1")
    score = json.loads((folder / "ingest" / "block_b_orbit1.json").read_text(encoding="utf-8"))
    _part(score, "B3")["empirical"]["feature_field"] = "measured"
    (folder / "ingest" / "block_b_orbit1.json").write_text(json.dumps(score), encoding="utf-8")
    record = check_orbit(folder, "orbit1")
    assert record["steps"]["B3"]["ok"] is False
    assert any("gaussian_features.npy" in item for item in record["steps"]["B3"]["mismatch"])
    assert record["steps"]["B3"]["achieved_claimed"] is False


def test_b2_label_matches_langsplat_codebook(tmp_path: Path):
    import numpy as np

    folder = _plant(tmp_path, "orbit1")
    bare = check_orbit(folder, "orbit1")
    assert bare["steps"]["B2"]["numbers"]["langsplat"] is False
    assert bare["steps"]["B2"]["label"] == "LangSplat unstarted"
    np.savez(
        folder / "semantics" / "features.codebook.npz",
        codebook=np.zeros((2, 3), dtype=np.float32),
        clip_features=np.zeros((2, 4), dtype=np.float32),
        levels=np.zeros((2,), dtype=np.int16),
    )
    measured = check_orbit(folder, "orbit1")
    assert measured["steps"]["B2"]["numbers"]["langsplat"] is True
    assert measured["steps"]["B2"]["label"] == "LangSplat measured"
    assert measured["achieved"] is False


def test_missing_source_is_a_fail_not_a_pass(tmp_path: Path):
    folder = _plant(tmp_path, "orbit2")
    (folder / "masks" / "orbit2" / "object.json").unlink()
    record = check_orbit(folder, "orbit2")
    assert record["steps"]["B1"]["ok"] is False
    assert record["steps"]["B1"]["skipped"]
    assert record["steps"]["B1"]["label"] == "skipped"
    assert record["ok"] is False


@pytest.mark.skipif(not LIVE_ORBIT2.is_file(), reason="data/can/ingest/block_b_orbit2.json is absent")
def test_live_orbit_scores_keep_m1_and_m2_false():
    root = repo_root() / "data" / "can"
    for orbit in ("orbit1", "orbit2", "orbit3"):
        score = json.loads((root / "ingest" / f"block_b_{orbit}.json").read_text(encoding="utf-8"))
        assert score["m1_met"] is False
        assert score["m2_met"] is False
        assert score["m1_achieved"] is False
        assert score["m2_achieved"] is False
        assert score["achieved"] is False


def _plant(tmp_path: Path, orbit: str, *, reviews: bool = True) -> Path:
    folder = tmp_path / "data" / "can" if (tmp_path / "data").name != "can" else tmp_path
    if folder.name != "can":
        folder = tmp_path / "data" / "can"
    ingest = folder / "ingest"
    ingest.mkdir(parents=True, exist_ok=True)
    (folder / "masks" / orbit).mkdir(parents=True, exist_ok=True)
    (folder / "semantics" / orbit).mkdir(parents=True, exist_ok=True)
    (folder / "debug").mkdir(parents=True, exist_ok=True)
    (folder / "reviews").mkdir(parents=True, exist_ok=True)
    if orbit == "orbit1":
        n_views, n_det, inside, outside, gaussians = 367, 96, 42108, 61237, 103345
        span_in, span_sc = 0.556, 9.293
        covers = False
        stem = "frame_00167"
        psnr, lpips = 36.77, 0.052
        pixels = 2034119
        reproj = 0.635
    elif orbit == "orbit2":
        n_views, n_det, inside, outside, gaussians = 315, 309, 32366, 117497, 149863
        span_in, span_sc = 9.777, 20.436
        covers = True
        stem = "frame_00001"
        psnr, lpips = 32.23, 0.059
        pixels = 2342775
        reproj = 0.714
    else:
        n_views, n_det, inside, outside, gaussians = 305, 302, 32786, 66353, 99139
        span_in, span_sc = 0.669, 10.226
        covers = False
        stem = "frame_00125"
        psnr, lpips = 36.71, 0.035
        pixels = 2830105
        reproj = 0.986
    block_a_path = ingest / "block_a.json"
    block_a = json.loads(block_a_path.read_text(encoding="utf-8")) if block_a_path.is_file() else {"orbits": []}
    block_a.setdefault("orbits", [])
    block_a["orbits"] = [row for row in block_a["orbits"] if row.get("orbit") != orbit]
    block_a["orbits"].append(
        {
            "orbit": orbit,
            "registered_pct": 100.0,
            "mean_reproj_px": reproj,
            "psnr": psnr,
            "lpips": lpips,
            "metres_per_unit": None,
        }
    )
    block_a["metres_per_unit"] = None
    block_a["scale"] = "unmeasured"
    block_a_path.write_text(json.dumps(block_a), encoding="utf-8")
    (ingest / f"ns_eval_{orbit}.json").write_text(
        json.dumps({"results": {"psnr": psnr, "lpips": lpips}}),
        encoding="utf-8",
    )
    (folder / "masks" / orbit / "object.json").write_text(
        json.dumps(
            {
                "detections": ["can"],
                "n_views": n_views,
                "n_views_with_detection": n_det,
                "largest_foreground_pixels": pixels,
                "n_inside": inside,
                "n_outside": outside,
                "n_gaussians": gaussians,
                "n_views_projected": n_det,
                "n_views_skipped_empty": n_views - n_det,
                "covers_room": covers,
                "span_inside": span_in,
                "span_scene": span_sc,
                "infill": "skipped",
                "projected": True,
            }
        ),
        encoding="utf-8",
    )
    codebook_name = "codebook.json" if orbit == "orbit1" else f"codebook_{orbit}.json"
    (folder / codebook_name).write_text(
        json.dumps(
            {
                "dinov2": True,
                "openclip": True,
                "learned_prior": False,
                "langsplat_codebook": False,
                "vector": [0.0] * 32,
                "n_copied": inside,
                "n_gaussians": gaussians,
            }
        ),
        encoding="utf-8",
    )
    (folder / "semantics" / orbit / "instances.json").write_text(
        json.dumps(
            {
                "instance_count": 1,
                "miou": None,
                "hand_labels": 0,
                "n_object": inside,
                "n_unassigned": outside,
                "learned_prior": False,
                "langsplat_codebook": False,
            }
        ),
        encoding="utf-8",
    )
    score = {
        "block": "B",
        "object": "can",
        "orbit": orbit,
        "state": "partial",
        "achieved": False,
        "m1_met": False,
        "m1_achieved": False,
        "m2_met": False,
        "m2_achieved": False,
        "m1": {
            "metres_per_unit": None,
            "scale": "unmeasured",
            "orbit": {"orbit": orbit, "psnr": psnr, "lpips": lpips, "registered_pct": 100.0},
        },
        "parts": [
            {
                "id": "B1",
                "state": "measured",
                "achieved": False,
                "empirical": {
                    "n_views": n_views,
                    "n_views_with_detection": n_det,
                    "detections": ["can"],
                    "largest_foreground_pixels": pixels,
                    "miou": None,
                },
            },
            {
                "id": "B2",
                "state": "unstarted",
                "achieved": False,
                "empirical": {
                    "langsplat": "unstarted",
                    "learned_prior": False,
                    "object_vector": {"state": "measured", "achieved": False, "dim": 32, "n_copied": inside, "learned_prior": False},
                },
            },
            {
                "id": "B3",
                "state": "partial",
                "achieved": False,
                "empirical": {
                    "reviews": {"asked": False},
                    "mask_lift": {
                        "n_inside": inside,
                        "n_outside": outside,
                        "n_gaussians": gaussians,
                        "n_views_projected": n_det,
                        "n_views_skipped_empty": n_views - n_det,
                        "covers_room": covers,
                        "span_inside": span_in,
                        "span_scene": span_sc,
                        "judge_projection": "kept",
                        "infill": "skipped",
                    },
                },
            },
            {
                "id": "B4",
                "state": "partial",
                "achieved": False,
                "empirical": {
                    "instance_count": 1,
                    "n_object": inside,
                    "n_unassigned": outside,
                    "miou": None,
                    "hand_labels": 0,
                    "learned_prior": False,
                },
            },
        ],
    }
    (ingest / f"block_b_{orbit}.json").write_text(json.dumps(score), encoding="utf-8")
    if reviews:
        for name in (f"{stem}.png", f"{stem}_mask.png", f"{stem}_splat.png", f"{stem}_splat_crop.png"):
            (folder / "reviews" / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    return folder


def _part(score: dict, part_id: str) -> dict:
    for part in score["parts"]:
        if part["id"] == part_id:
            return part
    raise KeyError(part_id)
