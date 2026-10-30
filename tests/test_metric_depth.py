"""Metric scale without AprilTag: depth / polycam / ruler only — never VLM metres."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.metric_depth import (
    DA3_METRIC_FOCAL_REF,
    ScaleRefuseError,
    coerce_da3_is_metric,
    da3_focal_normalized_to_metres,
    depth_backend_status,
    estimate_from_depth_map,
    is_da3_metric_checkpoint,
    is_measured_record,
    main,
    metres_per_unit_from_depth_pairs,
    metres_per_unit_from_ruler,
    recover_metric_scale,
    refuse_knowledge_as_metres,
    write_scale_json,
)


def test_depth_pairs_yield_finite_metres_per_unit():
    colmap = np.full(64, 2.0)
    metric = np.full(64, 0.5)
    scale = metres_per_unit_from_depth_pairs(colmap, metric, min_pairs=32)
    assert scale == pytest.approx(0.25)
    record = estimate_from_depth_map(metric, colmap, method="depth_anything", min_pairs=32)
    assert record["method"] == "depth_anything"
    assert record["achieved"] is True
    assert math_isfinite(record["metres_per_unit"])
    assert record["metres_per_unit"] == pytest.approx(0.25)


def test_recover_writes_measured_scale_from_mocked_depth(tmp_path: Path):
    colmap = np.linspace(1.0, 3.0, 40)
    metric = colmap * 0.1
    record = recover_metric_scale(
        "can",
        tmp_path,
        method="depth_anything",
        colmap_depths=colmap,
        metric_depths=metric,
        record_gate=False,
    )
    assert is_measured_record(record)
    assert record["method"] == "depth_anything"
    assert record["metres_per_unit"] == pytest.approx(0.1)
    saved = json.loads((tmp_path / "data" / "can" / "ingest" / "scale.json").read_text(encoding="utf-8"))
    assert saved["metres_per_unit"] == pytest.approx(0.1)
    assert saved["method"] == "depth_anything"
    assert saved["achieved"] is True


def test_recover_predictor_path_writes_finite_scale(tmp_path: Path):
    """Success path: mocked DA3-like predictor + COLMAP uv/z → finite metres_per_unit."""
    height, width = 48, 64
    true_scale = 0.2
    depth_map = np.full((height, width), 1.0, dtype=np.float64)

    def predictor(image_bgr, _intrinsics):
        assert image_bgr.shape[0] == height
        return depth_map

    rng = np.random.default_rng(0)
    uv = np.column_stack(
        (
            rng.integers(0, width, size=40),
            rng.integers(0, height, size=40),
        )
    ).astype(np.float64)
    colmap_z = np.full(40, 5.0)  # metric 1.0 / colmap 5.0 → 0.2
    image = np.zeros((height, width, 3), dtype=np.uint8)

    record = recover_metric_scale(
        "can",
        tmp_path,
        method="depth_anything",
        predictor=predictor,
        image_bgr=image,
        colmap_depths={"z": colmap_z, "uv": uv},
        record_gate=False,
    )
    assert is_measured_record(record)
    assert record["method"] == "depth_anything"
    assert record["metres_per_unit"] == pytest.approx(true_scale)
    assert record["achieved"] is True
    saved = json.loads((tmp_path / "data" / "can" / "ingest" / "scale.json").read_text(encoding="utf-8"))
    assert saved["metres_per_unit"] == pytest.approx(true_scale)


def test_missing_weights_leaves_null_and_not_achieved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SCENEAGENT_DEPTH_ANYTHING_WEIGHTS", raising=False)
    status = depth_backend_status("depth_anything")
    assert status["available"] is False
    assert status["status"] == "missing_weights"

    record = recover_metric_scale("can", tmp_path, method="depth_anything", record_gate=True)
    assert record["metres_per_unit"] is None
    assert record["achieved"] is False
    assert record["method"] == "not_measured"
    assert record["status"] == "missing_weights"
    assert is_measured_record(record) is False

    ledger = json.loads((tmp_path / "data" / "can" / "gate_ledger.json").read_text(encoding="utf-8"))
    entry = ledger["entries"][-1]
    assert entry["class"] == "score_only"
    assert entry["achieved"] is False
    assert entry["m1_met"] is False


def test_missing_frames_stamps_null_when_backend_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    weights = tmp_path / "da3_weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("SCENEAGENT_DEPTH_ANYTHING_WEIGHTS", str(weights))
    monkeypatch.setattr(
        "tools.metric_depth.depth_backend_status",
        lambda prefer="depth_anything": {
            "available": True,
            "method": "depth_anything",
            "status": "ready",
            "reason": "depth_anything weights and package are present",
            "weights": str(weights),
        },
    )
    # transforms + projected depths present so the failure is specifically missing_frames
    orbit = tmp_path / "data" / "can" / "colmap" / "orbit1"
    orbit.mkdir(parents=True)
    (orbit / "transforms.json").write_text(
        json.dumps(
            {
                "fl_x": 100.0,
                "fl_y": 100.0,
                "cx": 32.0,
                "cy": 24.0,
                "w": 64,
                "h": 48,
                "frames": [{"file_path": "images/frame_00001.png", "transform_matrix": np.eye(4).tolist()}],
            }
        ),
        encoding="utf-8",
    )
    uv = np.column_stack((np.linspace(1, 60, 40), np.linspace(1, 40, 40)))
    z = np.full(40, 2.0)

    record = recover_metric_scale(
        "can",
        tmp_path,
        method="depth_anything",
        colmap_depths={"z": z, "uv": uv, "intrinsics": {
            "fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 24.0, "width": 64, "height": 48,
        }},
        record_gate=True,
    )
    assert record["metres_per_unit"] is None
    assert record["achieved"] is False
    assert record["method"] == "not_measured"
    assert record["status"] == "missing_frames"
    assert "missing_frames" in record["reason"]
    assert is_measured_record(record) is False


def test_missing_transforms_stamps_null_when_backend_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "tools.metric_depth.depth_backend_status",
        lambda prefer="depth_anything": {
            "available": True,
            "method": "depth_anything",
            "status": "ready",
            "reason": "ready",
            "weights": "/tmp/weights",
        },
    )
    record = recover_metric_scale("can", tmp_path, method="depth_anything", record_gate=False)
    assert record["metres_per_unit"] is None
    assert record["status"] == "missing_transforms"
    assert "missing_transforms" in record["reason"]


def test_vlm_and_model_knowledge_refused(tmp_path: Path):
    with pytest.raises(ScaleRefuseError, match="model_knowledge"):
        refuse_knowledge_as_metres("model_knowledge")
    with pytest.raises(ScaleRefuseError, match="VLM"):
        refuse_knowledge_as_metres("vlm")
    with pytest.raises(ScaleRefuseError):
        write_scale_json(
            "can",
            {"metres_per_unit": 0.22, "method": "model_knowledge", "achieved": True},
            tmp_path,
        )
    with pytest.raises(ScaleRefuseError):
        recover_metric_scale("can", tmp_path, method="vlm_height", metres_per_unit=0.22)
    assert not (tmp_path / "data" / "can" / "ingest" / "scale.json").is_file()


def test_ruler_and_polycam_paths(tmp_path: Path):
    assert metres_per_unit_from_ruler(0.12, 2.4) == pytest.approx(0.05)
    ruler = recover_metric_scale(
        "can",
        tmp_path,
        method="ruler",
        length_m=0.12,
        length_units=2.4,
        record_gate=False,
    )
    assert ruler["method"] == "ruler"
    assert ruler["metres_per_unit"] == pytest.approx(0.05)

    polycam = recover_metric_scale(
        "speaker",
        tmp_path,
        method="polycam",
        metres_per_unit=1.0,
        force=True,
        record_gate=False,
    )
    assert polycam["method"] == "polycam"
    assert polycam["metres_per_unit"] == pytest.approx(1.0)


def test_cli_refuses_knowledge_and_score_only_on_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SCENEAGENT_DEPTH_ANYTHING_WEIGHTS", raising=False)
    assert main(["--obj", "can", "--root", str(tmp_path), "--method", "model_knowledge"]) == 2
    code = main(["--obj", "can", "--root", str(tmp_path), "--method", "depth_anything"])
    assert code == 0  # ScoreOnly exit so ingest continues
    saved = json.loads((tmp_path / "data" / "can" / "ingest" / "scale.json").read_text(encoding="utf-8"))
    assert saved["metres_per_unit"] is None
    assert saved["achieved"] is False


class _FakeParam:
    device = "cpu"


def test_da3_metric_focal_normalized_conversion():
    """Official DA3METRIC FAQ: metres = focal_px * net_output / 300 at depth resolution."""
    depth = np.full((50, 100), 0.5, dtype=np.float64)
    # Capture 200×100 → depth 50×100: sx=0.5, sy=0.5
    metres, focal = da3_focal_normalized_to_metres(
        depth, fx=600.0, fy=800.0, image_width=200, image_height=100
    )
    assert focal == pytest.approx(0.5 * (600.0 * 0.5 + 800.0 * 0.5))
    assert metres == pytest.approx(depth * (focal / DA3_METRIC_FOCAL_REF))
    assert is_da3_metric_checkpoint("da3metric-large")
    assert is_da3_metric_checkpoint("DA3METRIC-LARGE", "/workspace/models/depth_anything_3")
    assert not is_da3_metric_checkpoint("da3mono-large")
    assert not is_da3_metric_checkpoint("da3-large")
    assert coerce_da3_is_metric(1) == 1
    assert coerce_da3_is_metric(0) == 0
    assert coerce_da3_is_metric({}) == 0  # DA3 addict.Dict empty quirk
    assert coerce_da3_is_metric(None) == 0


def test_predict_metric_da3metric_uses_focal_path(monkeypatch: pytest.MonkeyPatch):
    """Metric checkpoint + broken is_metric={} still yields metres via focal/300."""
    from types import SimpleNamespace

    import tools.metric_depth as md

    height, width = 40, 60
    net = np.full((20, 30), 0.4, dtype=np.float64)  # processed-res stand-in

    class FakeModel:
        model_name = "da3metric-large"

        def parameters(self):
            return iter([_FakeParam()])

        def inference(self, _images, **_kwargs):
            return SimpleNamespace(depth=np.asarray([net]), is_metric={}, scale_factor=None)

    monkeypatch.setattr(
        md,
        "depth_backend_status",
        lambda prefer="depth_anything": {
            "available": True,
            "method": "depth_anything",
            "status": "ready",
            "weights": "fake-metric",
        },
    )
    monkeypatch.setattr(md, "_da3_model", lambda weights: FakeModel())
    monkeypatch.setitem(__import__("sys").modules, "depth_anything_3", SimpleNamespace())
    monkeypatch.setitem(
        __import__("sys").modules,
        "depth_anything_3.api",
        SimpleNamespace(DepthAnything3=object),
    )

    image = np.zeros((height, width, 3), dtype=np.uint8)
    intrinsics = {"fx": 300.0, "fy": 300.0, "cx": 30.0, "cy": 20.0, "width": width, "height": height}
    depth_m, meta = md.predict_metric_depth(image, intrinsics, prefer="depth_anything")
    assert depth_m.shape == (height, width)
    assert meta["metric_path"] == "focal_normalized_over_300"
    assert meta["is_metric"] == 1
    assert math_isfinite(float(np.nanmedian(depth_m)))
    assert float(np.nanmedian(depth_m)) > 0


def test_predict_non_metric_da3_refuses(monkeypatch: pytest.MonkeyPatch):
    """Relative DA3 checkpoint without is_metric/scale_factor must not invent metres."""
    from types import SimpleNamespace

    import tools.metric_depth as md

    class FakeModel:
        model_name = "da3mono-large"

        def parameters(self):
            return iter([_FakeParam()])

        def inference(self, _images, **_kwargs):
            depth = np.full((16, 16), 0.5, dtype=np.float64)
            return SimpleNamespace(depth=np.asarray([depth]), is_metric={}, scale_factor=None)

    monkeypatch.setattr(md, "_da3_model", lambda weights: FakeModel())
    monkeypatch.setattr(
        md,
        "depth_backend_status",
        lambda prefer="depth_anything": {
            "available": True,
            "method": "depth_anything",
            "status": "ready",
            "weights": "fake-mono",
        },
    )
    monkeypatch.setitem(__import__("sys").modules, "depth_anything_3", SimpleNamespace())
    monkeypatch.setitem(
        __import__("sys").modules,
        "depth_anything_3.api",
        SimpleNamespace(DepthAnything3=object),
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not metric"):
        md.predict_metric_depth(
            image,
            {"fx": 100.0, "fy": 100.0, "cx": 16.0, "cy": 16.0, "width": 32, "height": 32},
            prefer="depth_anything",
        )


def test_da3metric_without_intrinsics_refuses(monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    import tools.metric_depth as md

    class FakeModel:
        model_name = "da3metric-large"

        def parameters(self):
            return iter([_FakeParam()])

        def inference(self, *_a, **_k):
            raise AssertionError("must refuse before inference when fx/fy missing")

    monkeypatch.setattr(md, "_da3_model", lambda weights: FakeModel())
    monkeypatch.setattr(
        md,
        "depth_backend_status",
        lambda prefer="depth_anything": {
            "available": True,
            "method": "depth_anything",
            "status": "ready",
            "weights": "fake-metric",
        },
    )
    monkeypatch.setitem(__import__("sys").modules, "depth_anything_3", SimpleNamespace())
    monkeypatch.setitem(
        __import__("sys").modules,
        "depth_anything_3.api",
        SimpleNamespace(DepthAnything3=object),
    )
    with pytest.raises(RuntimeError, match="intrinsics"):
        md.predict_metric_depth(np.zeros((8, 8, 3), dtype=np.uint8), None, prefer="depth_anything")


def math_isfinite(value: float) -> bool:
    return np.isfinite(value)
