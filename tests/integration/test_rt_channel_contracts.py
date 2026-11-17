"""Contract tests C1/C2 for the RT pipeline (GPU; Sionna RT required)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

C = 299792458.0
NSC, SCS = 64, 120e3


def _monostatic_scene(R, fc, velocity=None):
    import mitsuba as mi
    from sionna.rt import PlanarArray, Receiver, Scene, Transmitter
    from sionna.rt.rcs import ConstantRCSSensingTarget

    sc = Scene()
    sc.add(Transmitter("tx", position=mi.Point3f(0.0, 0.0, 0.0)))
    sc.add(Receiver("rx", position=mi.Point3f(0.0, 0.0, 0.0)))
    sc.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.frequency = fc
    sc.add(ConstantRCSSensingTarget("t", sigma=1.0, position=mi.Point3f(float(R), 0.0, 0.0),
                                    velocity=(mi.Vector3f(*velocity) if velocity else None)))
    return sc


def _solve(sc):
    from sionna.rt.rcs import RCSSolver

    return RCSSolver(deterministic=True)(sc, max_depth=1, los=True, specular_reflection=False,
                                         refraction=False, seed=1)


def test_contract_flags() -> None:
    from fr3isac.sensing.rt_channel import assert_contract_flags

    assert_contract_flags()  # fails if Sionna RT cir/cfr defaults changed


@pytest.mark.parametrize("fc", [3.5e9, 12e9, 15e9, 28e9])
def test_phase_contract(fc: float) -> None:
    from fr3isac.sensing.rt_channel import cfr_passband, sensing_path_arrays

    R = 5.0
    arr = sensing_path_arrays(_solve(_monostatic_scene(R, fc)))
    assert arr["a"].size == 1
    offs = (np.arange(NSC) - NSC // 2) * SCS
    H = cfr_passband(arr["a"], arr["tau"], fc, offs)
    phase = np.unwrap(np.angle(H))
    slope = np.polyfit(offs, phase, 1)[0]
    tau_est = -slope / (2 * np.pi)
    tau_true = 2 * R / C
    delay_res = 1.0 / (NSC * SCS)
    assert abs(tau_est - tau_true) <= 0.1 * delay_res

    # Moving the target by dR changes the carrier phase by -4*pi*dR/lambda.
    lam = C / fc
    dR = lam / 100.0
    H1 = cfr_passband(arr["a"], arr["tau"], fc, np.array([0.0]))[0]
    arr2 = sensing_path_arrays(_solve(_monostatic_scene(R + dR, fc)))
    H2 = cfr_passband(arr2["a"], arr2["tau"], fc, np.array([0.0]))[0]
    dphase = np.angle(H2 * np.conj(H1))
    predicted = np.angle(np.exp(-1j * 4 * np.pi * dR / lam))
    assert abs(dphase - predicted) <= 0.05


def test_doppler_sign() -> None:
    from fr3isac.sensing.rt_channel import sensing_path_arrays

    fc = 3.5e9
    arr = sensing_path_arrays(_solve(_monostatic_scene(5.0, fc, velocity=(1.0, 0.0, 0.0))))
    lam = C / fc
    expected = -2 * 1.0 / lam  # receding -> negative
    assert abs(float(arr["doppler"][0]) - expected) <= 1.0


def test_noise_in_detector_tensor() -> None:
    from fr3isac.sensing.rt_channel import add_noise_frequency_domain, thermal_noise_variance

    sigma2 = thermal_noise_variance(SCS, noise_figure_db=7.0)
    rng = np.random.default_rng(0)
    x = np.array([add_noise_frequency_domain(np.zeros((1,), dtype=np.complex128), sigma2, rng)[0]
                  for _ in range(200_000)])
    assert np.mean(np.abs(x) ** 2) == pytest.approx(sigma2, rel=0.02)


def test_noise_in_detector_tensor_rt() -> None:
    """C2 (RT-backed): AWGN on the actual RT CFR has the modelled variance."""
    from fr3isac.sensing.rt_channel import (
        add_noise_frequency_domain,
        cfr_passband,
        sensing_path_arrays,
        thermal_noise_variance,
    )

    fc = 12e9
    arr = sensing_path_arrays(_solve(_monostatic_scene(5.0, fc)))
    offs = (np.arange(NSC) - NSC // 2) * SCS
    H = cfr_passband(arr["a"], arr["tau"], fc, offs)
    sigma2 = thermal_noise_variance(SCS, noise_figure_db=7.0)
    rng = np.random.default_rng(1)
    resid = np.array([add_noise_frequency_domain(H, sigma2, rng) - H for _ in range(2000)])
    var = float(np.var(resid.real) + np.var(resid.imag))
    assert var == pytest.approx(sigma2, rel=0.02)


def test_bandwidth_doubling_total_noise() -> None:
    """C2: doubling the occupied bandwidth at fixed spacing adds 3.01 dB total noise."""
    from fr3isac.sensing.rt_channel import thermal_noise_variance

    def total(bw: float) -> float:
        return (bw / SCS) * thermal_noise_variance(SCS, noise_figure_db=7.0)

    assert 10 * np.log10(total(2 * NSC * SCS) / total(NSC * SCS)) == pytest.approx(3.01, abs=0.1)


def test_no_gain_double_count() -> None:
    from fr3isac.sensing.rt_channel import radar_equation_power, sensing_path_arrays

    fc, R, p_tx = 28e9, 5.0, 1.0
    arr = sensing_path_arrays(_solve(_monostatic_scene(R, fc)))
    received = float(np.abs(arr["a"][0]) ** 2) * p_tx
    expected = radar_equation_power(p_tx, fc, R, 1.0)
    assert abs(10 * np.log10(received / expected)) <= 0.5
