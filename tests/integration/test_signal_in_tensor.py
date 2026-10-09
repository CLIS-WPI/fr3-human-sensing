"""C2 signal-in-tensor check: per-subcarrier energy = (P_tx/N_sc)*|h_eff|^2."""

from __future__ import annotations

import numpy as np
import pytest

from fr3isac.sensing.rt_channel import detector_input_scale, radar_equation_power

pytestmark = pytest.mark.gpu

C = 299792458.0


def test_detector_input_scale_identity() -> None:
    h = np.array([3.0 + 4.0j, -1.0 + 0.5j])
    scale = detector_input_scale(2.0, 8)
    y = scale * h
    np.testing.assert_allclose(np.abs(y) ** 2, (2.0 / 8) * np.abs(h) ** 2, rtol=1e-12)


def test_signal_energy_matches_radar_equation() -> None:
    from fr3isac.experiments.validation_gates import _monostatic, _solve
    from fr3isac.sensing.rt_channel import cfr_passband, sensing_path_arrays

    fc, R, p_tx, n_sc = 12e9, 5.0, 1.0, 64
    a = sensing_path_arrays(_solve(_monostatic(R, fc)))
    h = cfr_passband(a["a"], a["tau"], fc, np.array([0.0]))
    y = detector_input_scale(p_tx, n_sc) * h
    energy = float(np.abs(y[0]) ** 2)
    expected = (p_tx / n_sc) * radar_equation_power(p_tx, fc, R, 1.0)
    assert abs(10 * np.log10(energy / expected)) <= 0.1

    # A double count (missing the 1/N_sc split) would be off by 10log10(N_sc).
    assert abs(10 * np.log10(energy / (p_tx * radar_equation_power(p_tx, fc, R, 1.0)))) > 10
