"""Transient limb-motion onset detector (coherent change + template bank).

Detects the onset of transient limb motion within the deadline: it localizes the
body range from the static/pre-onset return, forms a clutter-subtracted coherent
change over a range neighborhood, and correlates it with a chirp template bank
built from the declared motion class. The matched filter integrates the transient
coherently over the CPI, and the detection threshold is calibrated on noise-only
windows at the target false-alarm rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fr3isac.config import SPEED_OF_LIGHT_M_S

DETECTOR_VERSION_V3 = "p3.4-1"

# Template bank spans the declared transient motion class. Parameters are the
# analytic arm amplitudes/omegas plus onset phases; no production trajectory is
# used to build the bank.
TEMPLATE_AMPLITUDES_M = (0.15, 0.22, 0.28)
TEMPLATE_OMEGAS_RAD_S = (2 * np.pi * 0.6, 2 * np.pi * 0.8, 2 * np.pi * 1.0)
TEMPLATE_PHASES = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)


def limb_chirp_template(fc_hz: float, amplitude_m: float, omega_rad_s: float,
                        phase0: float, num_slow: int, cpi_s: float) -> np.ndarray:
    """Unit-amplitude chirp template ``exp(j*4*pi/lambda * A sin(omega t + phase0))``.

    Derived from the analytic arm trajectory ``x(t)=A sin(omega t)`` so the
    instantaneous Doppler is ``2 v(t)/lambda`` with ``v(t)=A omega cos(omega t)``.
    """
    lam = SPEED_OF_LIGHT_M_S / fc_hz
    t = np.arange(num_slow) * (cpi_s / num_slow)
    return np.exp(1j * (4.0 * np.pi / lam) * amplitude_m * np.sin(omega_rad_s * t + phase0))


def template_bank(fc_hz: float, num_slow: int, cpi_s: float) -> np.ndarray:
    """Return ``[num_templates, num_slow]`` complex template bank (frozen class)."""
    templates = [
        limb_chirp_template(fc_hz, a, w, p, num_slow, cpi_s)
        for a in TEMPLATE_AMPLITUDES_M
        for w in TEMPLATE_OMEGAS_RAD_S
        for p in TEMPLATE_PHASES
    ]
    return np.asarray(templates)


def localize_body_range(ref_rd_power: np.ndarray, neighborhood: int = 1) -> tuple[int, list[int]]:
    """Preregistered body-range localization from the static/pre-onset RD map."""
    power = np.asarray(ref_rd_power, dtype=float)
    rng = np.arange(power.shape[0])
    r = int(rng[np.argmax(power[:, :].sum(axis=1))])
    bins = [b for b in range(r - neighborhood, r + neighborhood + 1) if 0 <= b < power.shape[0]]
    return r, bins


def matched_change_statistic(
    x_cur: np.ndarray,
    x_ref: np.ndarray,
    templates: np.ndarray,
    sigma2: float,
    eps: float = 1e-30,
) -> float:
    """Max normalized coherent change correlation over the template bank.

    ``stat = max_t |<x_cur - x_ref, t>|^2 / (||t||^2 * sigma2)``.
    """
    diff = np.asarray(x_cur, dtype=np.complex128) - np.asarray(x_ref, dtype=np.complex128)
    norm = np.sum(np.abs(templates) ** 2, axis=1)
    corr = np.abs(templates.conj() @ diff) ** 2 / np.maximum(norm * max(sigma2, eps), eps)
    return float(np.max(corr))


@dataclass
class TransientDetectorV3:
    carrier_hz: float
    cpi_s: float
    hop_s: float
    num_slow: int
    neighborhood: int = 1
    threshold: float = float("nan")
    version: str = DETECTOR_VERSION_V3

    @property
    def num_templates(self) -> int:
        return len(TEMPLATE_AMPLITUDES_M) * len(TEMPLATE_OMEGAS_RAD_S) * len(TEMPLATE_PHASES)

    @property
    def effective_search_dimension(self) -> int:
        return (2 * self.neighborhood + 1) * self.num_templates

    def templates(self) -> np.ndarray:
        return template_bank(self.carrier_hz, self.num_slow, self.cpi_s)

    def statistic(self, x_cur: np.ndarray, x_ref: np.ndarray) -> float:
        sigma2 = float(np.var(x_ref - np.mean(x_ref)))
        return matched_change_statistic(x_cur, x_ref, self.templates(), sigma2)


__all__ = [
    "DETECTOR_VERSION_V3",
    "TEMPLATE_AMPLITUDES_M",
    "TEMPLATE_OMEGAS_RAD_S",
    "TEMPLATE_PHASES",
    "limb_chirp_template",
    "template_bank",
    "localize_body_range",
    "matched_change_statistic",
    "TransientDetectorV3",
]
