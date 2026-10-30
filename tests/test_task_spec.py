"""The user's prompt is the pipeline entry. The task spec is written before segmentation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tools.block_b_job import feature_field_may_start, mask_covers_frame, union_target_mask
from tools.common.paths import repo_root
from tools.pipeline import apply_mask_retry, mask_retry_commands, run
from tools.task_spec import (
    accept_pipeline_prompt,
    gate_detected_target,
    raised_box_threshold,
    record_room_mask_retry,
    request_path,
    run_target_mask_check,
)


def test_stored_prompt_matches_the_user_and_a_missing_key_invents_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    user = "Here is the video of X, Y, Z "
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    called = {"b0": False}

    def b0(*_args, **_kwargs):
        called["b0"] = True
        return 0

    monkeypatch.setattr("tools.pipeline.run_b0", b0)
    code = run("kitchen", through="B0", root=tmp_path, prompt=user)
    assert code == 2
    assert called["b0"] is False
    spec = json.loads(request_path("kitchen", tmp_path).read_text(encoding="utf-8"))
    assert spec["prompt"] == user
    assert "job" not in spec
    assert "target" not in spec
    assert "seen" not in spec
    assert "deliverable" not in spec


def test_spec_is_written_before_segmentation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    user = "a room; I want sisters of Y"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    order: list[str] = []

    def post(_url, payload, api_key, timeout=120):
        order.append("parse")
        assert api_key == "sk-test-secret"
        assert "sk-test-secret" not in json.dumps(payload)
        stored = json.loads(request_path("kitchen", tmp_path).read_text(encoding="utf-8"))
        assert stored == {"prompt": user}
        text = payload["messages"][0]["content"][0]["text"]
        assert user in text
        images = [item for item in payload["messages"][0]["content"] if item["type"] == "image_url"]
        assert len(images) == 4
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "job": "room",
                                "target": "Y",
                                "seen": ["X", "Y", "Z"],
                                "deliverable": "sisters_of_target",
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("tools.common.vlm._http_post", post)
    frames = []
    for index in range(6):
        path = tmp_path / f"frame_{index}.png"
        path.write_bytes(b"\x89PNG\r\n")
        frames.append(path)
    code = accept_pipeline_prompt("kitchen", user, tmp_path, frames=frames[:4])
    assert code == 0
    assert order == ["parse"]
    spec = json.loads(request_path("kitchen", tmp_path).read_text(encoding="utf-8"))
    assert spec["prompt"] == user
    assert spec["job"] == "room"
    assert spec["target"] == "Y"
    assert spec["seen"] == ["X", "Y", "Z"]
    assert "sk-test-secret" not in json.dumps(spec)

    def segment(target, params):
        order.append("segment")
        on_disk = json.loads(request_path("kitchen", tmp_path).read_text(encoding="utf-8"))
        assert on_disk["prompt"] == user
        assert on_disk["target"] == "Y"
        assert target == "Y"
        assert "retrain" not in params
        return {"overlay": str(tmp_path / "overlay.png"), "view_fractions": [0.12], "names": [target]}

    result = run_target_mask_check(
        tmp_path / "data" / "kitchen",
        segment,
        review_fn=lambda _path, _attempt: {"verdict": "pass", "correction": {"parameter": None, "value": None, "reason": ""}},
    )
    assert order == ["parse", "segment"]
    assert result["action"] == "keep"
    assert result["retrain"] is False
    assert result["sisters"] is False
    assert result["feature_field"] == "not_this_step"


def test_bad_parse_keeps_the_prompt_and_does_not_invent_a_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    user = "This is an orbit video of G."
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")

    def post(_url, _payload, _api_key, timeout=120):
        return {"choices": [{"message": {"content": json.dumps({"job": "scene", "target": "", "seen": [], "deliverable": ""})}}]}

    monkeypatch.setattr("tools.common.vlm._http_post", post)
    code = accept_pipeline_prompt("scene", user, tmp_path, frames=[])
    assert code == 2
    spec = json.loads(request_path("scene", tmp_path).read_text(encoding="utf-8"))
    assert spec == {"prompt": user}


def test_room_prompt_does_not_put_other_names_in_the_target_mask():
    masks = np.zeros((3, 2, 2), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 0, 1] = True
    masks[2, 1, 0] = True
    merged, kept = union_target_mask(masks, ["X", "Y", "Z"], "Y")
    assert kept == ["Y"]
    assert bool(merged[0, 0]) is False
    assert bool(merged[0, 1]) is True
    assert bool(merged[1, 0]) is False
    merged_case, kept_case = union_target_mask(masks, ["x", "y", "z"], "Y")
    assert kept_case == ["y"]
    assert bool(merged_case[1, 0]) is False


def test_room_sized_mask_does_not_start_the_feature_field(tmp_path: Path):
    folder = tmp_path / "data" / "kitchen"
    folder.mkdir(parents=True)
    user = "a room; I want sisters of Y"
    (folder / "request.json").write_text(
        json.dumps(
            {
                "prompt": user,
                "job": "room",
                "target": "Y",
                "seen": ["X", "Y", "Z"],
                "deliverable": "sisters_of_target",
            }
        ),
        encoding="utf-8",
    )
    assert mask_covers_frame(90, 100) is True
    assert mask_covers_frame(89, 100) is False
    assert feature_field_may_start("room", view_fractions=[0.93], covers_room=True, span_ratio=0.8) is False
    assert feature_field_may_start("room", view_fractions=[0.2], covers_room=False, span_ratio=0.62) is False
    assert feature_field_may_start("orbit", view_fractions=[0.2], covers_room=True, span_ratio=0.8) is True
    assert feature_field_may_start("orbit", view_fractions=[0.95]) is False

    def segment(_target, _params):
        return {
            "overlay": str(tmp_path / "overlay.png"),
            "view_fractions": [0.93],
            "covers_room": True,
            "span_ratio": 0.8,
            "mask_pixels": 930,
        }

    result = run_target_mask_check(
        folder,
        segment,
        review_fn=lambda _path, _attempt: {"verdict": "pass", "correction": {}},
    )
    assert result["feature_field"] == "not_started"
    assert result["dropped_views"] == [0]
    assert result["retrain"] is False
    assert list(folder.rglob("*sister*")) == []
    saved = json.loads((folder / "reviews" / "target_mask_check.json").read_text(encoding="utf-8"))
    assert saved["mask"]["mask_pixels"] == 930
    assert saved["prompt"] == user


def test_second_mask_miss_stops_with_the_measured_mask(tmp_path: Path):
    folder = tmp_path / "data" / "kitchen"
    folder.mkdir(parents=True)
    (folder / "request.json").write_text(
        json.dumps(
            {
                "prompt": "a room; I want sisters of Y",
                "job": "room",
                "target": "Y",
                "seen": ["X", "Y", "Z"],
                "deliverable": "sisters_of_target",
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, dict]] = []

    def segment(target, params):
        calls.append((target, dict(params)))
        return {"overlay": str(tmp_path / "overlay.png"), "view_fractions": [0.2], "mask_pixels": 40 + len(calls)}

    verdicts = [
        {"verdict": "fail", "correction": {"parameter": "box_threshold", "value": 0.4, "reason": "wide"}},
        {"verdict": "fail", "correction": {"parameter": "text_prompt", "value": "Y.", "reason": "still wide"}},
    ]

    def review(_path, attempt):
        return verdicts[attempt - 1]

    result = run_target_mask_check(folder, segment, review_fn=review)
    assert calls == [("Y", {}), ("Y", {"box_threshold": 0.4})]
    assert result["attempts"] == 2
    assert result["action"] == "stop"
    assert result["retrain"] is False
    assert result["sisters"] is False
    assert result["mask"]["mask_pixels"] == 42
    assert list(folder.rglob("*sister*")) == []

    calls.clear()

    def merge_review(_path, _attempt):
        return {"verdict": "fail", "correction": {"parameter": "merge", "value": None, "reason": "other names"}}

    stopped = run_target_mask_check(folder, segment, review_fn=merge_review)
    assert calls == [("Y", {})]
    assert stopped["action"] == "stop"
    assert stopped["retrain"] is False


def test_an_existing_overlay_is_looked_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    folder = tmp_path / "data" / "kitchen"
    reviews = folder / "reviews"
    reviews.mkdir(parents=True)
    (reviews / "frame_00001_mask.png").write_bytes(b"\x89PNG\r\n")
    user = "a room; I want sisters of Y"
    (folder / "request.json").write_text(
        json.dumps(
            {
                "prompt": user,
                "job": "room",
                "target": "Y",
                "seen": ["X", "Y", "Z"],
                "deliverable": "sisters_of_target",
            }
        ),
        encoding="utf-8",
    )
    looks: list[str] = []

    def fake_review(_path, target, record_path=None):
        looks.append(target)
        return {
            "verdict": "fail",
            "confidence": 0.9,
            "problem": "extra_part",
            "correction": {"parameter": "text_prompt", "value": "Y.", "reason": "too wide"},
            "evidence": "includes the table",
        }

    monkeypatch.setattr("tools.task_spec.review_target_overlay", fake_review)
    code = gate_detected_target("kitchen", tmp_path)
    assert code == 0
    assert looks == ["Y"]
    retry = json.loads((folder / "mask_retry.json").read_text(encoding="utf-8"))
    assert retry["params"] == {"text_prompt": "Y."}
    assert retry["retrain"] is False
    assert retry["sisters"] is False
    assert list(folder.rglob("*sister*")) == []


def test_segmentation_refuses_to_start_without_a_spec(tmp_path: Path):
    called: list[str] = []

    def segment(target, _params):
        called.append(target)
        return {}

    with pytest.raises(FileNotFoundError):
        run_target_mask_check(tmp_path, segment)
    assert called == []


def test_missing_overlay_is_asked_false_not_a_pass(tmp_path: Path):
    folder = tmp_path / "data" / "kitchen"
    folder.mkdir(parents=True)
    (folder / "request.json").write_text(
        json.dumps(
            {
                "prompt": "Here is the room. I want the can.",
                "job": "room",
                "target": "can",
                "seen": ["can"],
                "deliverable": "sisters_of_target",
            }
        ),
        encoding="utf-8",
    )

    def segment(_target, _params):
        return {"view_fractions": [0.2], "covers_room": False, "mask_pixels": 10}

    result = run_target_mask_check(folder, segment)
    assert result["action"] == "stop"
    assert result["asked"] is False
    assert result["reason"] == "no target overlay"
    assert result["retrain"] is False
    assert result["sisters"] is False
    saved = json.loads((folder / "reviews" / "target_mask_check.json").read_text(encoding="utf-8"))
    assert saved["asked"] is False
    assert saved["action"] == "stop"
    assert saved["reason"] == "no target overlay"


def test_gpu_scripts_do_not_parse_the_sentence():
    root = repo_root()
    semantics = (root / "vast-cluster" / "scripts" / "object_semantics.py").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in semantics
    assert "chat/completions" not in semantics
    assert "union_target_mask" in semantics
    assert '"bottle"' not in semantics.split("reject_object", 1)[1].split("def ", 1)[0]
    b2 = (root / "vast-cluster" / "scripts" / "object_langsplat_b2.py").read_text(encoding="utf-8")
    assert b2.index("mask_covers_frame") < b2.index("def _encode_clip")
    b3 = (root / "vast-cluster" / "scripts" / "object_langsplat_b3.py").read_text(encoding="utf-8")
    assert 'if job == "orbit"' in b3
    assert 'job != "orbit"' in b3


def test_b2_drops_a_view_that_covers_the_frame(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from PIL import Image

    path = repo_root() / "vast-cluster" / "scripts" / "object_langsplat_b2.py"
    spec = importlib.util.spec_from_file_location("object_langsplat_b2_drop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    view = tmp_path / "frame.png"
    mask_path = tmp_path / "frame_mask.png"
    Image.new("RGB", (10, 10), (20, 20, 20)).save(view)
    cover = np.zeros((10, 10), dtype=np.uint8)
    cover.ravel()[:90] = 255
    Image.fromarray(cover, mode="L").save(mask_path)
    assert module._load_rgb_mask(view, mask_path) is None
    tight = np.zeros((10, 10), dtype=np.uint8)
    tight.ravel()[:70] = 255
    Image.fromarray(tight, mode="L").save(mask_path)
    assert module._load_rgb_mask(view, mask_path) is not None


def test_room_sized_lift_records_one_retry_and_does_not_start_the_feature_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    folder = tmp_path / "data" / "kitchen"
    _room_spec(folder)
    _lift(folder, "orbit2", {"covers_room": False, "span_inside": 0.2, "span_scene": 2.0})
    _lift(folder, "clip", {"covers_room": True, "span_inside": 9.78, "span_scene": 20.44})
    _lift(folder, "half", {"covers_room": False, "span_inside": 5.0, "span_scene": 10.0})
    (folder / "reviews").mkdir(parents=True)
    (folder / "reviews" / "frame_00001_mask.png").write_bytes(b"\x89PNG\r\n")

    def boom(*_args, **_kwargs):
        raise AssertionError("vision was called")

    monkeypatch.setattr("tools.task_spec.review_target_overlay", boom)
    code = gate_detected_target("kitchen", tmp_path)
    assert code == 0
    retry = json.loads((folder / "mask_retry.json").read_text(encoding="utf-8"))
    assert retry["orbits"] == ["clip", "half"]
    assert "orbit2" not in retry["orbits"]
    assert "can" not in json.dumps(retry)
    assert retry["params"]["box_threshold"] == raised_box_threshold()
    assert retry["params"]["box_threshold"] > 0.25
    assert retry["retrain"] is False
    assert retry["splat_retrained"] is False
    assert retry["sisters"] is False
    assert retry["detection"] == "pending"
    assert retry["feature_field"] == "not_started"
    saved = json.loads((folder / "reviews" / "target_mask_check.json").read_text(encoding="utf-8"))
    assert saved["feature_field"] == "not_started"
    assert saved["asked"] is False
    assert saved["retrain"] is False
    assert list(folder.rglob("*sister*")) == []
    assert list(folder.rglob("feature_field.json")) == []
    text = (folder / "mask_retry.json").read_text(encoding="utf-8")
    again = record_room_mask_retry(folder)
    assert again["wrote"] is False
    assert (folder / "mask_retry.json").read_text(encoding="utf-8") == text
    assert list(folder.glob("mask_retry*.json")) == [folder / "mask_retry.json"]


def test_second_room_sized_lift_stays_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    folder = tmp_path / "data" / "kitchen"
    _room_spec(folder)
    _lift(folder, "clip", {"covers_room": True, "span_inside": 9.78, "span_scene": 20.44})
    _configs(folder, ["clip"])
    record_room_mask_retry(folder)
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    code = apply_mask_retry("kitchen", tmp_path, runner=runner)
    err = capsys.readouterr().err
    assert code == 2
    assert "9.78" in err
    assert "20.44" in err
    assert "not repeated" in err
    retry = json.loads((folder / "mask_retry.json").read_text(encoding="utf-8"))
    assert retry["detection"] == "rerun"
    assert retry["params"]["box_threshold"] == raised_box_threshold()
    assert retry["retrain"] is False
    assert retry["splat_retrained"] is False
    check = json.loads((folder / "reviews" / "target_mask_check.json").read_text(encoding="utf-8"))
    assert check["action"] == "stop"
    assert check["attempts"] == 2
    assert check["feature_field"] == "not_started"
    assert check["measured"][0]["span_inside"] == 9.78
    assert check["measured"][0]["span_scene"] == 20.44
    names = [Path(argv[0]).name for argv in calls]
    assert names == ["object_semantics.py", "object_crop_iou.py"]
    assert calls[0][calls[0].index("--box-threshold") + 1] == f"{raised_box_threshold():.2f}"
    assert "--box-threshold" not in calls[1]
    assert "object_langsplat_b3.py" not in names
    assert all("ns-train" not in " ".join(argv) for argv in calls)
    assert all(argv[argv.index("--orbit") + 1] == "clip" for argv in calls)
    second = apply_mask_retry("kitchen", tmp_path, runner=runner)
    assert second == 2
    assert len(calls) == 2
    assert list(folder.glob("mask_retry*.json")) == [folder / "mask_retry.json"]


def test_orbit_job_skips_the_half_scene_test(tmp_path: Path):
    folder = tmp_path / "data" / "kitchen"
    _room_spec(folder, job="orbit", prompt="This is an orbit video of G.")
    _lift(folder, "clip", {"covers_room": True, "span_inside": 5.0, "span_scene": 10.0})
    recorded = record_room_mask_retry(folder)
    assert recorded["action"] == "skip"
    assert recorded["reason"] == "orbit job"
    assert not (folder / "mask_retry.json").exists()
    assert feature_field_may_start("orbit", view_fractions=[0.2], covers_room=True, span_ratio=0.5) is True
    code = gate_detected_target("kitchen", tmp_path)
    assert code == 0
    assert not (folder / "mask_retry.json").exists()


def test_missing_job_does_not_arm_a_retry(tmp_path: Path):
    folder = tmp_path / "data" / "kitchen"
    folder.mkdir(parents=True)
    _lift(folder, "clip", {"covers_room": True, "span_inside": 9.78, "span_scene": 20.44})
    recorded = record_room_mask_retry(folder)
    assert recorded["action"] == "skip"
    assert recorded["reason"] == "missing job"
    assert not (folder / "mask_retry.json").exists()
    assert gate_detected_target("kitchen", tmp_path) == 0
    assert not (folder / "mask_retry.json").exists()
    assert list(folder.rglob("feature_field.json")) == []


def test_frame_covering_view_records_one_retry(tmp_path: Path):
    pytest.importorskip("PIL")
    from PIL import Image

    folder = tmp_path / "data" / "kitchen"
    _room_spec(folder)
    _lift(folder, "edge", {"covers_room": False, "span_inside": 0.2, "span_scene": 4.0})
    cover = np.zeros((10, 10), dtype=np.uint8)
    cover.ravel()[:90] = 255
    Image.fromarray(cover, mode="L").save(folder / "masks" / "edge" / "frame_mask.png")
    recorded = record_room_mask_retry(folder)
    assert recorded["orbits"] == ["edge"]
    assert recorded["feature_field"] == "not_started"
    tight = tmp_path / "data" / "other"
    _room_spec(tight)
    _lift(tight, "edge", {"covers_room": False, "span_inside": 0.2, "span_scene": 4.0, "view_fractions": [0.89]})
    assert record_room_mask_retry(tight)["action"] == "skip"
    assert not (tight / "mask_retry.json").exists()


def test_no_host_keeps_the_recorded_retry_and_the_mask(tmp_path: Path):
    folder = tmp_path / "data" / "kitchen"
    _room_spec(folder)
    _lift(folder, "clip", {"covers_room": True, "span_inside": 9.78, "span_scene": 20.44})
    _configs(folder, ["clip"])
    record_room_mask_retry(folder)
    before = (folder / "masks" / "clip" / "object.json").read_text(encoding="utf-8")
    seen = {"called": False}

    def lookup():
        seen["called"] = True
        return None

    code = apply_mask_retry("kitchen", tmp_path, host_lookup=lookup)
    assert code == 0
    assert seen["called"] is True
    assert (folder / "masks" / "clip" / "object.json").read_text(encoding="utf-8") == before
    retry = json.loads((folder / "mask_retry.json").read_text(encoding="utf-8"))
    assert retry["detection"] == "pending"
    assert not (folder / "masks" / "clip" / "gaussian_in_mask.npy").exists()


def test_retry_commands_name_only_the_measured_orbits(tmp_path: Path):
    folder = tmp_path / "data" / "kitchen"
    _room_spec(folder)
    _lift(folder, "orbit2", {"covers_room": False, "span_inside": 0.2, "span_scene": 2.0})
    _lift(folder, "clip", {"covers_room": True, "span_inside": 1.0, "span_scene": 4.0})
    _configs(folder, ["orbit2", "clip"])
    record_room_mask_retry(folder)
    commands = mask_retry_commands("kitchen", tmp_path)
    orbits = [argv[argv.index("--orbit") + 1] for argv in commands]
    assert orbits == ["clip", "clip"]
    assert "orbit2" not in orbits
    detect = commands[0]
    assert Path(detect[0]).name == "object_semantics.py"
    assert detect[detect.index("--box-threshold") + 1] == f"{raised_box_threshold():.2f}"
    assert detect[detect.index("--target") + 1] == "Y"
    assert Path(detect[0]).name != "object_langsplat_b3.py"


def _room_spec(
    folder: Path,
    job: str = "room",
    deliverable: str = "sisters_of_target",
    prompt: str = "a room; I want sisters of Y",
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
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


def _lift(folder: Path, orbit: str, payload: dict) -> None:
    path = folder / "masks" / orbit / "object.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _configs(folder: Path, orbits: list[str]) -> None:
    path = folder / "ingest" / "block_a.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "orbits": [
                    {"orbit": orbit, "config": f"/root/SceneagentX/outputs/kitchen_{orbit}/splatfacto/run/config.yml"}
                    for orbit in orbits
                ]
            }
        ),
        encoding="utf-8",
    )
