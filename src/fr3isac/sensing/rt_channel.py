"""Ray-traced sensing channel helpers.

Builds the frequency-domain channel used by the sensing detector from Sionna RT path
coefficients, using the passband form so the carrier phase is applied exactly once
and no library delay or power normalization is introduced.

Conventions:
* ``paths.a`` is a ``(real, imag)`` tuple; ``paths.tau`` is the absolute delay (s).
* ``H[k] = sum_p a_p * exp(-j 2 pi (f_c + Delta_f_k) tau_p)``.
* per-subcarrier noise variance ``k T B_sc NF``.
* beamforming weights are unit norm; antenna gain is never added on top of the channel.
"""

from __future__ import annotations

import inspect

import numpy as np

from fr3isac.config import SPEED_OF_LIGHT_M_S
from fr3isac.phy.noise import BOLTZMANN_J_PER_K

# Guarded defaults of the installed Sionna RT library routines. If a future
# version changes these, ``assert_contract_flags`` fails so the documented
# behaviour cannot silently drift.
_EXPECTED_CFR_DEFAULTS = {"normalize_delays": True, "normalize": False}
_EXPECTED_CIR_DEFAULTS = {"normalize_delays": True}


def assert_contract_flags() -> None:
    """Fail if the installed Sionna RT ``cir``/``cfr`` defaults changed."""
    from sionna.rt import Paths

    for name, expected in (("cfr", _EXPECTED_CFR_DEFAULTS), ("cir", _EXPECTED_CIR_DEFAULTS)):
        params = inspect.signature(getattr(Paths, name)).parameters
        for key, value in expected.items():
            if key in params and params[key].default != value:
                raise AssertionError(
                    f"Paths.{name}.{key} default changed from {value} to {params[key].default}"
                )


def sensing_path_arrays(paths) -> dict[str, np.ndarray]:
    """Return per-path arrays for the first rx/tx link, valid paths only."""
    from fr3isac.rt.engine import paths_a_complex, paths_tau, paths_valid

    a = paths_a_complex(paths)
    tau = paths_tau(paths)
    valid = paths_valid(paths)
    # a: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths(, time)]
    a_link = a[0, 0, 0, 0]
    if a_link.ndim == 2:
        a_link = a_link[:, 0]
    if a_link.size == 0 or tau.size == 0 or valid.size == 0:
        return {"a": np.array([]), "tau": np.array([]), "doppler": None}
    tau_link = tau.reshape(-1, tau.shape[-1])[0]
    valid_link = valid.reshape(-1, valid.shape[-1])[0]
    doppler = getattr(paths, "doppler", None)
    dop_link = None
    if doppler is not None:
        d = np.asarray(doppler)
        dop_link = d.reshape(-1, d.shape[-1])[0]
    return {"a": a_link[valid_link], "tau": tau_link[valid_link],
            "doppler": (dop_link[valid_link] if dop_link is not None else None)}


def cfr_passband(a_paths: np.ndarray, tau_paths: np.ndarray, fc_hz: float,
                 subcarrier_offsets_hz: np.ndarray) -> np.ndarray:
    """Passband CFR ``H[k] = sum_p a_p exp(-j2pi (f_c + df_k) tau_p)``."""
    a = np.asarray(a_paths, dtype=np.complex128).ravel()
    tau = np.asarray(tau_paths, dtype=np.float64).ravel()
    f = (fc_hz + np.asarray(subcarrier_offsets_hz, dtype=np.float64).ravel())
    return np.exp(-1j * 2 * np.pi * np.outer(f, tau)) @ a


def thermal_noise_variance(subcarrier_bandwidth_hz: float, noise_figure_db: float = 0.0,
                           temperature_k: float = 290.0) -> float:
    """Per-subcarrier noise variance ``k T B_sc NF`` (complex sample variance).

    ``B_sc`` is the subcarrier bandwidth (subcarrier spacing). Total in-band noise
    over ``N`` subcarriers is ``N`` times this value, so doubling the occupied
    bandwidth at fixed spacing raises the total noise by 3.01 dB.
    """
    return BOLTZMANN_J_PER_K * temperature_k * subcarrier_bandwidth_hz * 10 ** (noise_figure_db / 10.0)


def add_noise_frequency_domain(H: np.ndarray, sigma2: float, rng: np.random.Generator) -> np.ndarray:
    """Add circular complex AWGN with total power ``sigma2`` per sample."""
    H = np.asarray(H, dtype=np.complex128)
    return H + np.sqrt(sigma2 / 2.0) * (
        rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))


def additive_interference_variance(sigma2_noise: float, inr_db: float) -> float:
    """C4: additive Gaussian interference power INR dB above thermal noise."""
    return sigma2_noise * 10.0 ** (inr_db / 10.0)


def effective_channel(H: np.ndarray, w_t: np.ndarray, w_r: np.ndarray) -> complex:
    """``h_eff = w_r^H H w_t`` with unit-norm weights (H [num_rx_ant, num_tx_ant])."""
    w_t = np.asarray(w_t, dtype=np.complex128).ravel()
    w_r = np.asarray(w_r, dtype=np.complex128).ravel()
    if abs(np.linalg.norm(w_t) - 1.0) > 1e-9 or abs(np.linalg.norm(w_r) - 1.0) > 1e-9:
        raise ValueError("beamforming weights must have unit norm")
    return complex(w_r.conj() @ np.asarray(H, dtype=np.complex128) @ w_t)


def detector_input_scale(p_tx_w: float, num_subcarriers: int) -> float:
    """C2: per-subcarrier amplitude scale ``sqrt(P_tx / N_sc)`` of the detector tensor.

    Total conducted power ``P_tx`` is split equally over the active sensing
    subcarriers, so the detector-input tensor has per-subcarrier signal energy
    ``(P_tx / N_sc) * |h_eff|^2``.
    """
    if p_tx_w < 0 or num_subcarriers < 1:
        raise ValueError("p_tx_w >= 0 and num_subcarriers >= 1 required")
    return float(np.sqrt(p_tx_w / num_subcarriers))


def radar_equation_power(p_tx_w: float, fc_hz: float, range_m: float, sigma_m2: float) -> float:
    """Monostatic received target power for unit-norm single-element weights."""
    lam = SPEED_OF_LIGHT_M_S / fc_hz
    return p_tx_w * sigma_m2 * lam**2 / ((4.0 * np.pi) ** 3 * range_m**4)


__all__ = [
    "assert_contract_flags",
    "sensing_path_arrays",
    "cfr_passband",
    "thermal_noise_variance",
    "add_noise_frequency_domain",
    "additive_interference_variance",
    "effective_channel",
    "detector_input_scale",
    "radar_equation_power",
]
