import json

from tools.common.blocks import BLOCKS
from tools.common.status import (
    StatusError,
    begin_step,
    default_status,
    finish_step,
    get_path,
    load_status,
    main,
    save_status,
    set_metric,
    validate_status,
)


def test_default_status_matches_schema():
    validate_status(default_status("marker"))


def test_init_writes_every_block(tmp_path):
    doc = load_status("marker", tmp_path)
    assert [step["id"] for step in doc["steps"]] == [block.id for block in BLOCKS]
    assert doc["ground_truth"]["length_m"] is None
    assert (tmp_path / "data" / "marker" / "status.json").is_file()


def test_set_metric_round_trip(tmp_path):
    set_metric("marker", "ground_truth.length_m", 0.142, tmp_path)
    set_metric("marker", "step01.orbit1.registered_pct", 95, tmp_path)
    doc = load_status("marker", tmp_path)
    assert get_path(doc, "ground_truth.length_m") == 0.142
    assert get_path(doc, "step01.orbit1.registered_pct") == 95
    assert doc["steps"][0]["id"] == "B0"


def test_attempts_increment_once_per_begin(tmp_path):
    begin_step("marker", "B1", tmp_path)
    begin_step("marker", "B1", tmp_path)
    doc = finish_step("marker", "B1", "pass", tmp_path, actual={"passed": True})
    step = next(item for item in doc["steps"] if item["id"] == "B1")
    assert step["attempts"] == 2
    assert step["state"] == "pass"
    assert step["actual"]["passed"] is True


def test_length_must_be_positive(tmp_path):
    try:
        set_metric("marker", "ground_truth.length_m", 0, tmp_path)
    except StatusError as exc:
        assert "length_m" in str(exc)
    else:
        raise AssertionError("expected StatusError")


def test_invalid_state_is_rejected(tmp_path):
    doc = default_status("marker")
    doc["steps"][0]["state"] = "Not started"
    try:
        save_status("marker", doc, tmp_path)
    except StatusError as exc:
        assert "not_started" in str(exc)
    else:
        raise AssertionError("expected StatusError")


def test_cli_set_and_show(tmp_path):
    assert main(["--root", str(tmp_path), "--obj", "marker", "set", "setup_minutes", "12.5"]) == 0
    doc = json.loads((tmp_path / "data" / "marker" / "status.json").read_text(encoding="utf-8"))
    assert doc["setup_minutes"] == 12.5


def test_merge_keeps_saved_attempt(tmp_path):
    begin_step("marker", "B0", tmp_path)
    doc = load_status("marker", tmp_path)
    assert next(step for step in doc["steps"] if step["id"] == "B0")["attempts"] == 1
    assert len(doc["steps"]) == len(BLOCKS)
