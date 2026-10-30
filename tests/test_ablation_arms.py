"""Ablation arms: off never calls OpenAI, propose never writes a verdict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.common.vlm import MissingReviewImage, review
from tools.iterate import iterate_skill


def _see_fixture(tmp_path: Path) -> None:
    masks = tmp_path / "data" / "bottle" / "masks"
    masks.mkdir(parents=True)
    (masks / "object.json").write_text(
        json.dumps({"projected": True, "infill": "skipped", "n_inside": 408236, "n_outside": 546578}),
        encoding="utf-8",
    )
    (masks / "view_000_mask.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    cropped = tmp_path / "data" / "bottle" / "objects" / "bottle" / "object.ply"
    cropped.parent.mkdir(parents=True)
    cropped.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")


def _verdict(verdict: str, confidence: float, parameter: str | None, problem: str = "none") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "verdict": verdict,
                            "confidence": confidence,
                            "problem": problem,
                            "correction": {
                                "parameter": parameter if parameter is not None else "box_threshold",
                                "value": 0.3,
                                "reason": "overlay",
                            },
                            "evidence": "the overlay",
                        }
                    )
                }
            }
        ]
    }


def test_mode_off_never_calls_openai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("OpenAI was called")

    monkeypatch.setattr("tools.common.vlm._http_post", boom)
    _see_fixture(tmp_path)
    assert iterate_skill("bottle", "see", tmp_path, vlm_mode="off") == 0
    assert not list((tmp_path / "data" / "bottle" / "reviews").glob("*.json"))


def test_mode_propose_never_writes_verdict_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("propose mode must not call OpenAI for a verdict")

    monkeypatch.setattr("tools.common.vlm._http_post", boom)
    _see_fixture(tmp_path)
    assert iterate_skill("bottle", "see", tmp_path, vlm_mode="propose") == 0
    reviews = tmp_path / "data" / "bottle" / "reviews"
    written = list(reviews.glob("*.json")) if reviews.is_dir() else []
    assert written == []
    assert calls["n"] == 0


def test_mode_full_writes_mocked_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "laptop-test-key")
    monkeypatch.delenv("SCENEAGENT_VLM_MODEL", raising=False)

    def fake_post(_url, payload, api_key, timeout=120):
        assert payload["temperature"] == 0
        assert payload["model"] == "gpt-6-astra"
        assert api_key == "laptop-test-key"
        assert timeout == 120
        return _verdict("pass", 0.91, "box_threshold")

    monkeypatch.setattr("tools.common.vlm._http_post", fake_post)
    _see_fixture(tmp_path)
    assert iterate_skill("bottle", "see", tmp_path, vlm_mode="full") == 0
    record = json.loads((tmp_path / "data" / "bottle" / "reviews" / "see_1.json").read_text(encoding="utf-8"))
    assert record["request"]["temperature"] == 0
    assert record["verdict"]["verdict"] == "pass"
    assert "laptop-test-key" not in json.dumps(record)
    assert (tmp_path / "data" / "bottle" / "reviews" / "see.png").is_file()


def test_low_confidence_pass_becomes_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "laptop-test-key")
    monkeypatch.setattr("tools.common.vlm._http_post", lambda *_a, **_k: _verdict("pass", 0.4, "box_threshold"))
    image = tmp_path / "view.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    verdict = review("see", [image], "Does the overlay cover the bottle?", ["box_threshold"])
    assert verdict["verdict"] == "fail"
    assert verdict["problem"] == "other"


def test_disallowed_parameter_is_cleared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "laptop-test-key")
    monkeypatch.setattr(
        "tools.common.vlm._http_post",
        lambda *_a, **_k: _verdict("fail", 0.9, "retrain", problem="other"),
    )
    image = tmp_path / "view.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    verdict = review("see", [image], "Does the overlay cover the bottle?", ["box_threshold", "text_prompt"])
    assert verdict["correction"]["parameter"] is None
    assert verdict["correction"]["value"] is None
    assert verdict["correction"]["reason"] == "model proposed a disallowed parameter"


def test_missing_image_raises(tmp_path: Path):
    with pytest.raises(MissingReviewImage, match="missing review image"):
        review("see", [tmp_path / "missing.png"], "Look at the overlay.", ["box_threshold"])


def test_full_mode_retries_once_then_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "laptop-test-key")
    replies = [
        _verdict("fail", 0.9, "box_threshold", problem="seam"),
        _verdict("fail", 0.88, "box_threshold", problem="seam"),
    ]

    def fake_post(*_args, **_kwargs):
        return replies.pop(0)

    monkeypatch.setattr("tools.common.vlm._http_post", fake_post)
    _see_fixture(tmp_path)
    assert iterate_skill("bottle", "see", tmp_path, vlm_mode="full") == 2
    params = json.loads((tmp_path / "data" / "bottle" / "skill_params.json").read_text(encoding="utf-8"))
    assert list(params) == ["box_threshold"]
    question = json.loads((tmp_path / "data" / "bottle" / "question.json").read_text(encoding="utf-8"))
    assert question["block"] == "B5"
    runs = (tmp_path / "data" / "bottle" / "runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(runs) == 2
    assert json.loads(runs[0])["try"] == 1
    assert json.loads(runs[1])["try"] == 2
    assert json.loads(runs[1])["kept"] is False
