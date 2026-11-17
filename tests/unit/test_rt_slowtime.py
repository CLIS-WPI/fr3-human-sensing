"""Unit tests for C3 slow-time helpers (no GPU / no Sionna solver needed)."""

from __future__ import annotations

import numpy as np

from fr3isac.sensing.rt_slowtime import interpolate_cfr, reconstruction_nmse

FC = 3.5e9


def _track(n_frames: int, tau0: float = 3.0e-8, dtau: float = 1e-11):
    return {
        "key": ((16,), (0,)),
        "samples": {
            i: {"a": 1.0 + 0.0j, "tau": tau0 + i * dtau, "vertex": np.zeros(3)}
            for i in range(n_frames)
        },
    }


def test_interpolate_cfr_linear_delay_matches_analytic() -> None:
    anchor = np.arange(4) * 1e-3
    query = np.array([0.5e-3, 1.5e-3, 2.5e-3])
    H = interpolate_cfr([_track(4)], anchor, query, FC, np.array([0.0]))
    tau_q = 3.0e-8 + np.array([0.5, 1.5, 2.5]) * 1e-11
    expected = np.exp(-1j * 2 * np.pi * FC * tau_q)
    np.testing.assert_allclose(H.ravel(), expected, rtol=1e-9)


def test_reconstruction_nmse_zero_for_identical() -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((8, 5)) + 1j * rng.standard_normal((8, 5))
    assert reconstruction_nmse(ref, ref) == 0.0


def test_reconstruction_nmse_positive_for_offset() -> None:
    ref = np.ones((4, 3), dtype=np.complex128)
    fast = np.zeros((4, 3), dtype=np.complex128)
    assert reconstruction_nmse(fast, ref) == 1.0


def test_synthesize_cfr_constant_velocity_doppler() -> None:
    import numpy as np

    from fr3isac.sensing.rt_slowtime import synthesize_cfr

    fc, prf, nslow = 28e9, 1280.0, 128
    dt = 1.0 / prf
    v, r0 = 3.3, 5.0
    times = np.arange(nslow) * dt
    amp = np.ones((1, nslow), dtype=np.complex128)
    H = synthesize_cfr(fc, np.array([0.0]), amp, (r0 + v * times)[None, :])
    x = H.sum(axis=0)
    nfft = 8 * nslow
    X = np.fft.fftshift(np.fft.fft(x * np.hanning(nslow), n=nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=dt))
    peak = float(f[np.argmax(np.abs(X))])
    truth = -2 * v / (299792458.0 / fc)
    assert abs(peak - truth) <= prf / nslow
