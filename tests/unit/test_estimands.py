"""Regression tests for the approved F2 receive-gain correction (TASK 2 follow-up).

F2 run used rx = 20 dB; under the fixed-aperture F2 definition the rx gain follows
the aperture, so F2_corrected = F2_run + (20 - G_rx,F2[fc]), with G_rx,F2 = the F2
transmit (aperture) gain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fr3isac.config import FairnessConfig
from fr3isac.fairness.normalize import normalize

NUMBERS = Path("results/commag/manuscript_numbers.json")
CARRIERS = (3.5e9, 12e9, 15e9, 28e9)

pytestmark = pytest.mark.skipif(not NUMBERS.exists(), reason="manuscript numbers not generated")


def _rx_gain_f2(fc):
    # monostatic: the F2 receive array is the transmit array (aperture gain)
    return normalize(FairnessConfig(regime="F2"), fc).bs.gain_db


def _row(numbers, field, fc, detector, regime):
    return next(r for r in numbers[field]
                if r["fc_hz"] == fc and r["detector"] == detector and r["regime"] == regime)


def test_f2_correction_matches_aperture_rx_gain():
    numbers = json.loads(NUMBERS.read_text())["numbers"]
    for field in ("required_eirp_dbm", "required_eirp_dbm_visible",
                  "required_eirp_dbm_per_placement_median"):
        for fc in CARRIERS:
            for det in ("D0", "D1"):
                f1 = _row(numbers, field, fc, det, "F1")["point"]
                f2 = _row(numbers, field, fc, det, "F2")["point"]
                expected = 20.0 - _rx_gain_f2(fc)
                assert abs((f2 - f1) - expected) <= 0.7, (field, fc, det, f2 - f1, expected)


def test_previous_per_placement_median_f2_reproduced():
    numbers = json.loads(NUMBERS.read_text())["numbers"]
    assert abs(_row(numbers, "required_eirp_dbm_per_placement_median", 12e9, "D0", "F2")["point"]
               - (-54.85)) <= 0.5
    assert abs(_row(numbers, "required_eirp_dbm_per_placement_median", 12e9, "D1", "F2")["point"]
               - (-57.71)) <= 0.5


def test_primary_nonfinite_states():
    import numpy as np
    import pandas as pd

    from fr3isac.analysis.estimands import primary

    # NaN -> raise
    nan = pd.DataFrame({"scenario_id": [0], "required_eirp_dbm": [np.nan],
                        "forearm_visible": [True]})
    try:
        primary(nan, n_boot=10)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("NaN must raise")

    # +inf (no modeled motion return) is kept and can drive the quantile to +inf
    inf = pd.DataFrame({"scenario_id": [0, 1], "required_eirp_dbm": [-50.0, np.inf],
                        "forearm_visible": [True, True]})
    point, lo, hi, n = primary(inf, q=0.90, n_boot=20)
    assert point == np.inf and n == 2
