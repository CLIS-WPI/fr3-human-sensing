"""T-3: ground-truth leakage check for data-driven localization (unit test).

Behavioral check that ``localize_body_bin`` is driven by the measured channel and
has no ground-truth input: it must follow the data, and its interface must expose
no true bin/position.
"""

from __future__ import annotations

import inspect

import numpy as np

from fr3isac.config import SPEED_OF_LIGHT_M_S
from fr3isac.sensing.rt_localize import localize_body_bin, range_m_to_bin

NSC, SCS = 816, 120e3
OCC_BW = NSC * SCS
FC = 3.5e9


def _single_path_cfr(range_m: float) -> np.ndarray:
    tau = 2.0 * range_m / SPEED_OF_LIGHT_M_S
    offsets = (np.arange(NSC) - NSC // 2) * SCS
    return np.exp(-1j * 2 * np.pi * (FC + offsets) * tau)[:, None]


def test_localization_follows_data() -> None:
    for r in (3.0, 6.0, 8.0):
        target = _single_path_cfr(r)
        r_hat, bins = localize_body_bin(target, np.zeros_like(target), OCC_BW, 1.0, 12.0)
        assert r_hat == range_m_to_bin(r, NSC, OCC_BW)
        assert r_hat in bins


def test_localization_changes_with_data() -> None:
    a, _ = localize_body_bin(_single_path_cfr(3.0), np.zeros((NSC, 1)), OCC_BW, 1.0, 12.0)
    b, _ = localize_body_bin(_single_path_cfr(8.0), np.zeros((NSC, 1)), OCC_BW, 1.0, 12.0)
    assert a != b


def test_no_ground_truth_parameter() -> None:
    params = set(inspect.signature(localize_body_bin).parameters)
    forbidden = {"true_bin", "truth", "ground_truth", "gt", "target_position", "position"}
    assert params.isdisjoint(forbidden), params
