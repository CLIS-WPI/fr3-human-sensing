"""Regression: articulated fast path must use BS-relative ranges (BS off origin).

The gate scenes put tx=rx at the origin, so they could not catch the
``_r1a_fast`` origin-vs-BS bug. This test uses the Tier-1 room whose BS is at
(0.6, 0.6, 2.8) and checks the fast method against direct RT.
"""

from __future__ import annotations

import numpy as np
import pytest

from fr3isac.experiments import localization as m3
from fr3isac.sensing.rt_slowtime import reconstruction_nmse

pytestmark = pytest.mark.gpu


def test_bs_is_not_at_origin() -> None:
    assert np.linalg.norm(m3._bs_position()) > 0.1


def test_articulated_fast_matches_direct_with_nonorigin_bs() -> None:
    fc, regime = 12e9, "F1"
    root = (4.0, 4.0, m3.TARGET_HEIGHT_M)
    motion = m3.TEST_MOTIONS[0]
    offsets = (np.arange(m3.NSC) - m3.NSC // 2) * m3.SCS
    gain = m3._regime_gain_db(fc, regime)
    times = np.arange(m3.NSLOW) * (m3.CPI_S / m3.NSLOW)
    samples = m3.analytic_arm_trajectory(root, times, amplitude_m=motion[0],
                                         omega_rad_s=2 * np.pi * motion[1])
    direct = np.stack([m3._solve_cfr(fc, "R1-A", s, offsets, gain) for s in samples], axis=1)
    fast = m3._r1a_fast(fc, root, motion, offsets, gain)
    nmse = reconstruction_nmse(fast, direct)
    assert nmse < 1e-3, f"fast-vs-direct NMSE too high: {nmse}"

    # The origin-only range model would be wrong by the BS offset; show it differs.
    origin_ranges = np.linalg.norm(np.asarray(samples[0].forearm_position))
    bs_ranges = np.linalg.norm(np.asarray(samples[0].forearm_position) - m3._bs_position())
    assert abs(origin_ranges - bs_ranges) > 0.5
