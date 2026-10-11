"""Block 10 / E: a scalar rigidity is not heterogeneous. Destruction was not run."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.block_gates import GateError, destruction_record, press_required, rigidity_label


def test_scalar_rigidity_not_called_heterogeneous():
    assert rigidity_label(0.98) == "scalar"
    assert rigidity_label(0.98) != "heterogeneous"


def test_missing_press_picture_fails(tmp_path: Path):
    with pytest.raises(GateError, match="missing press picture"):
        press_required(tmp_path / "press_after.png")


def test_destruction_recorded_not_run():
    record = destruction_record()
    assert record["destruction"] == "not_run"
    assert record["physgaussian"] is False
    assert record["ran"] is False
