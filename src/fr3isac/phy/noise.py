"""Receiver noise models."""

from __future__ import annotations

import numpy as np

BOLTZMANN_J_PER_K = 1.380649e-23


def thermal_noise_power_w(
    bandwidth_hz: float,
    noise_figure_db: float = 0.0,
    temperature_k: float = 290.0,
) -> float:
    """k * T * B * NF."""
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth must be positive")
    nf_linear = 10.0 ** (noise_figure_db / 10.0)
    return BOLTZMANN_J_PER_K * temperature_k * bandwidth_hz * nf_linear


def thermal_noise_power_dbm(
    bandwidth_hz: float,
    noise_figure_db: float = 0.0,
    temperature_k: float = 290.0,
) -> float:
    return float(10.0 * np.log10(thermal_noise_power_w(bandwidth_hz, noise_figure_db, temperature_k) / 1e-3))


def sample_awgn(
    shape: tuple[int, ...],
    noise_power_w: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circularly-symmetric complex AWGN with total power ``noise_power_w``.

    Variance per complex sample is ``noise_power_w`` (real+imag each half).
    """
    sigma = np.sqrt(noise_power_w / 2.0)
    return sigma * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )


__all__ = [
    "BOLTZMANN_J_PER_K",
    "thermal_noise_power_w",
    "thermal_noise_power_dbm",
    "sample_awgn",
]
