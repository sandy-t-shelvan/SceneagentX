"""Block 5 / D: mass stays unchecked until it is weighed. Rigidity is per region."""

from __future__ import annotations

import pytest

from tools.block_gates import BOTTLE_MASS_KG, GateError, check_mass, lock_learned_prior, rigidity_field, rigidity_label


def test_missing_weight_keeps_mass_unchecked():
    record = check_mass(BOTTLE_MASS_KG, None)
    assert record["mass_kg"] == 0.35
    assert record["checked"] is False


def test_measured_weight_accepts_or_rejects():
    accepted = check_mass(0.35, 0.35)
    assert accepted["checked"] is True
    assert accepted["accepted"] is True
    rejected = check_mass(0.35, 1.0)
    assert rejected["checked"] is False
    assert rejected["accepted"] is False


def test_rigidity_per_region_not_scalar():
    assert rigidity_label(0.98) == "scalar"
    with pytest.raises(GateError, match="per region"):
        rigidity_field(0.98)
    field = rigidity_field(
        [
            {"part": "body", "rigidity": 0.98},
            {"part": "lid", "rigidity": 0.4},
        ]
    )
    assert field["label"] == "heterogeneous"
    assert len(field["regions"]) == 2


def test_learned_prior_cannot_be_set_true():
    kept = lock_learned_prior({"learned_prior": False, "mass_kg": 0.35})
    assert kept["learned_prior"] is False
    with pytest.raises(GateError, match="learned_prior"):
        lock_learned_prior({"learned_prior": True})
    with pytest.raises(GateError, match="learned_prior"):
        lock_learned_prior({"per_gaussian": {"learned_prior": True}})
