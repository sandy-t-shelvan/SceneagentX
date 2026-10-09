import numpy as np

from tools.align import (
    evaluate_align,
    length_within,
    marker_length_m,
    metres_per_unit,
    rigid_align,
    rms_mm,
    run_align,
    tag_side_units,
)
from tools.common.status import load_status, set_metric


def _square(side: float = 2.0) -> list[list[float]]:
    return [[0.0, 0.0, 0.0], [side, 0.0, 0.0], [side, side, 0.0], [0.0, side, 0.0]]


def _grid() -> list[list[float]]:
    return [[float(x), float(y), 0.0] for x in range(4) for y in range(4)]


def test_tag_scale_and_length():
    corners = np.array(_square(2.0))
    assert tag_side_units(corners) == 2.0
    scale = metres_per_unit(2.0)
    assert scale == 0.025
    measured = marker_length_m(np.array([[0.0, 0.0, 0.0], [5.6, 0.0, 0.0]]), scale)
    assert abs(measured - 0.14) < 1e-9
    assert length_within(measured, 0.14)
    assert length_within(0.14 * 1.06, 0.14) is False


def test_icp_removes_a_one_millimetre_shift():
    target = np.array(_grid(), dtype=np.float64) * 0.025
    source = target + np.array([0.001, 0.0, 0.0])
    _, rms_m = rigid_align(source, target)
    assert rms_mm(rms_m) < 3.0
    assert evaluate_align(True, rms_mm(rms_m))["rms"] is True
    assert evaluate_align(True, 3.0)["rms"] is False


def test_run_align_passes_b4(tmp_path):
    set_metric("marker", "ground_truth.length_m", 0.14, tmp_path)
    geometry = {
        "tag_corners": _square(2.0),
        "marker_endpoints": [[0.0, 0.0, 0.0], [5.6, 0.0, 0.0]],
        "orbit1_table": _grid(),
        "orbit2_table": _grid(),
        "orbit3_table": _grid(),
        "orbit2_tag_corners": _square(2.0),
        "orbit3_tag_corners": _square(2.0),
    }
    code = run_align("marker", tmp_path, geometry=geometry, rerun_orbits=False)
    assert code == 0
    doc = load_status("marker", tmp_path)
    assert abs(doc["step01"]["scale"]["orbit1"] - 0.025) < 1e-9
    assert doc["align"]["rms_mm"] < 3.0
    assert next(step for step in doc["steps"] if step["id"] == "B4")["state"] == "pass"
