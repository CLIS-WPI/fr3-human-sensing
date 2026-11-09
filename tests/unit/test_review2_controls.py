import numpy as np

from fr3isac.experiments import rerun_v2_core as core
from fr3isac.experiments.review2_controls import const_fraction, residual_bank


def test_residual_template_matches_onset_record():
    cfg = core.load_cfg("configs/rerun_v2.yaml")
    st = cfg["slow_time"]
    fc, a, f, p, n0 = 12e9, 0.2, 1.0, 0.7, 40
    t = np.arange(st["window_samples"]) / st["prf_hz"]
    t0 = n0 / st["prf_hz"] - 1e-9
    d = np.where(t > t0, a * (np.sin(2 * np.pi * f * (t - t0) + p) - np.sin(p)), 0.0)
    x = np.exp(1j * 4 * np.pi * fc / core.C * d) - 1.0
    bank, kept = residual_bank(fc, [(a, f, p)], cfg, offsets=np.array([n0]))
    assert kept == [(a, f, p, n0)]
    cap = np.abs(np.vdot(bank[0], x)) ** 2 / (np.vdot(bank[0], bank[0]).real * np.vdot(x, x).real)
    assert cap > 0.999


def test_const_fraction():
    assert np.isclose(const_fraction(np.ones(8))[0], 1.0)
    assert np.isclose(const_fraction(np.exp(2j * np.pi * np.arange(8) / 8))[0], 0.0)
