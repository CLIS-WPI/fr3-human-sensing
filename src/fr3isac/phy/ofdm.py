"""OFDM numerology and path-to-waveform conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fr3isac.config import SPEED_OF_LIGHT_M_S


@dataclass(frozen=True)
class OFDMNumerology:
    fft_size: int
    occupied_subcarriers: int
    subcarrier_spacing_hz: float
    cp_length: int
    num_ofdm_symbols: int

    @property
    def sample_rate_hz(self) -> float:
        return self.fft_size * self.subcarrier_spacing_hz

    @property
    def occupied_bandwidth_hz(self) -> float:
        return self.occupied_subcarriers * self.subcarrier_spacing_hz

    @property
    def symbol_duration_s(self) -> float:
        return 1.0 / self.subcarrier_spacing_hz

    @property
    def cp_duration_s(self) -> float:
        return self.cp_length / self.sample_rate_hz

    @property
    def subcarrier_frequencies_hz(self) -> np.ndarray:
        """Baseband subcarrier offsets of the occupied band (Hz)."""
        n = self.occupied_subcarriers
        k = np.arange(-(n // 2), n - (n // 2))
        return k * self.subcarrier_spacing_hz


def numerology_from_bandwidth(
    bandwidth_hz: float,
    subcarrier_spacing_hz: float,
    cp_length: int = 72,
    num_ofdm_symbols: int = 14,
    min_fft_size: int = 128,
) -> OFDMNumerology:
    """Choose a power-of-two FFT that fits the occupied bandwidth."""
    raw = bandwidth_hz / subcarrier_spacing_hz
    fft_size = 1
    while fft_size < raw:
        fft_size *= 2
    fft_size = max(fft_size, min_fft_size)
    occupied = int(round(raw))
    if occupied > fft_size:
        raise ValueError("occupied subcarriers exceed FFT size")
    return OFDMNumerology(
        fft_size=fft_size,
        occupied_subcarriers=occupied,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        cp_length=cp_length,
        num_ofdm_symbols=num_ofdm_symbols,
    )


def subcarrier_frequencies_absolute_hz(fc_hz: float, numerology: OFDMNumerology) -> np.ndarray:
    return fc_hz + numerology.subcarrier_frequencies_hz


def paths_to_cfr(
    path_coefficients: np.ndarray,
    path_delays_s: np.ndarray,
    freqs_hz: np.ndarray,
) -> np.ndarray:
    """Frequency response of a specular multipath channel.

    ``H(f) = sum_p a_p * exp(-j 2*pi*f*tau_p)`` where ``a_p`` already includes
    antenna/propagation amplitude and phase. Paths are summed coherently, which
    is the correct composition for the specular paths produced by PathSolver.
    """
    a = np.asarray(path_coefficients).ravel().astype(np.complex128)
    tau = np.asarray(path_delays_s).ravel().astype(np.float64)
    f = np.asarray(freqs_hz).ravel().astype(np.float64)
    if a.shape[0] != tau.shape[0]:
        raise ValueError("path_coefficients and path_delays_s must have equal length")
    phase = np.exp(-1j * 2.0 * np.pi * np.outer(f, tau))  # [F, P]
    return phase @ a  # [F]


def free_space_channel_coefficient(fc_hz: float, distance_m: float) -> complex:
    """Friis narrowband channel coefficient for isotropic unit-gain antennas."""
    if distance_m <= 0:
        raise ValueError("distance must be positive")
    return complex(SPEED_OF_LIGHT_M_S / (4.0 * np.pi * fc_hz * distance_m))


def free_space_path_gain(fc_hz: float, distance_m: float) -> float:
    return float(abs(free_space_channel_coefficient(fc_hz, distance_m)) ** 2)


def net_goodput_mbps(useful_bits: float, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        raise ValueError("elapsed_s must be positive")
    return useful_bits / elapsed_s / 1e6


__all__ = [
    "OFDMNumerology",
    "numerology_from_bandwidth",
    "subcarrier_frequencies_absolute_hz",
    "paths_to_cfr",
    "free_space_channel_coefficient",
    "free_space_path_gain",
    "net_goodput_mbps",
]
