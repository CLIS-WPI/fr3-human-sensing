"""Data-driven sensing localization (no ground-truth inputs).

The body range is localized from the measured channel only: the
``target scene - empty-room reference`` range profile (IFFT over subcarriers) peaks
at the body range bin. No true target position or bin is passed in.
"""

from __future__ import annotations

import numpy as np

from fr3isac.config import SPEED_OF_LIGHT_M_S


def range_profile(cfr: np.ndarray) -> np.ndarray:
    """Power range profile from a CFR ``[n_subcarriers, n_slow]`` (IFFT over subcarriers)."""
    cfr = np.asarray(cfr, dtype=np.complex128)
    prof = np.abs(np.fft.ifft(cfr, axis=0)) ** 2
    return prof.sum(axis=1) if prof.ndim == 2 else prof


def range_bin_to_m(bin_index: float, num_subcarriers: int, occupied_bw_hz: float) -> float:
    """Monostatic range of a range-profile bin."""
    return bin_index * SPEED_OF_LIGHT_M_S / (2.0 * occupied_bw_hz)


def range_m_to_bin(range_m: float, num_subcarriers: int, occupied_bw_hz: float) -> int:
    return int(round(range_m * 2.0 * occupied_bw_hz / SPEED_OF_LIGHT_M_S))


def localize_body_bin(
    target_cfr: np.ndarray,
    empty_cfr: np.ndarray,
    occupied_bw_hz: float,
    range_min_m: float,
    range_max_m: float,
    neighborhood: int = 1,
) -> tuple[int, list[int]]:
    """Localize the body range bin from ``target - empty`` (data-driven).

    Returns the peak bin and the search neighborhood ``[r-neighborhood, r+neighborhood]``.
    """
    nsc = np.asarray(target_cfr).shape[0]
    change = np.asarray(target_cfr, dtype=np.complex128) - np.asarray(empty_cfr, dtype=np.complex128)
    prof = range_profile(change)
    lo = max(0, range_m_to_bin(range_min_m, nsc, occupied_bw_hz))
    hi = min(nsc - 1, range_m_to_bin(range_max_m, nsc, occupied_bw_hz))
    hi = max(hi, lo + 1)
    seg = prof[lo:hi]
    r = lo + int(np.argmax(seg))
    bins = [b for b in range(r - neighborhood, r + neighborhood + 1) if 0 <= b < nsc]
    return r, bins


__all__ = [
    "range_profile",
    "range_bin_to_m",
    "range_m_to_bin",
    "localize_body_bin",
]
