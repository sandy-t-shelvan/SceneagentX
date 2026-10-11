"""Block 3 / D: object vector for physics. The LangSplat codebook stays unstarted."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.block_gates import (
    GateError,
    assign_gaussian_vectors,
    lock_learned_prior,
    write_label_vectors,
)
from tools.common.blocks import get_block


def test_object_gaussians_get_object_vector():
    inside = np.array([True, False, True])
    obj = np.array([1.0, 2.0])
    scene = np.array([3.0, 4.0])
    assigned = assign_gaussian_vectors(inside, obj, scene)
    assert np.allclose(assigned[0], obj)
    assert np.allclose(assigned[2], obj)


def test_other_gaussians_get_scene_mean():
    inside = np.array([True, False, True])
    obj = np.array([1.0, 2.0])
    scene = np.array([3.0, 4.0])
    assigned = assign_gaussian_vectors(inside, obj, scene)
    assert np.allclose(assigned[1], scene)


def test_no_gaussian_without_vector():
    inside = np.array([True, False, True])
    assigned = assign_gaussian_vectors(inside, np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert assigned.shape == (3, 2)
    assert not np.isnan(assigned).any()


def test_two_label_fixture_writes_two_vectors(tmp_path: Path):
    path = write_label_vectors(
        tmp_path,
        np.array([0, 1, 0]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    saved = np.load(path)
    assert saved["vectors"].shape[0] == 2
    assert saved["per_gaussian"].shape == (3, 2)
    assert not np.isnan(saved["per_gaussian"]).any()


def test_learned_prior_stays_false():
    record = lock_learned_prior({"learned_prior": False, "label": "object vector for physics"})
    assert record["learned_prior"] is False
    assert record["per_gaussian"]["learned_prior"] is False
    with pytest.raises(GateError, match="learned_prior"):
        lock_learned_prior({"learned_prior": True})
    block = get_block("B7")
    assert block.name == "object vector for physics"
    assert "LangSplat" in get_block("B5").summary
