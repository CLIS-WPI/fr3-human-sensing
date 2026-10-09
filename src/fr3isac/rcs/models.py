"""RCS models and the aggregate-scattering budget.

Tiers:
* R0  native standardized human (no normalization);
* R1-P one point at the root with a matched aggregate scattering budget;
* R1-A articulated segments under the same aggregate budget;
* R2  frequency-aware model (conditional; not used as validated truth in P0).

The Sionna RCS callable signature is ``f(k_i, k_s, seed) -> Float`` and does
**not** receive carrier frequency, so frequency-aware behaviour must be bound
through a closure or precomputed table (section 15.2). ``frequency_bound_rcs``
implements that closure for a simple power-law model; P0 uses constant-RCS
targets, so R2 is present but not promoted to headline evidence.
"""

from __future__ import annotations

import numpy as np

from fr3isac.config import SPEED_OF_LIGHT_M_S


def segment_power_budget_weights(weights: list[float]) -> list[float]:
    """Normalize non-negative weights so their power-RCS values sum to one."""
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return list(w / total)


def aggregate_rcs_budget(total_sigma_m2: float, weights: list[float]) -> list[float]:
    """Per-segment power-RCS values that sum to ``total_sigma_m2`` (section 15.4)."""
    if total_sigma_m2 <= 0:
        raise ValueError("total_sigma_m2 must be positive")
    return [total_sigma_m2 * wi for wi in segment_power_budget_weights(weights)]


def r2_power_law_sigma(sigma_ref_m2: float, fc_hz: float, fc_ref_hz: float, exponent: float) -> float:
    """Frequency-aware scaling ``sigma(fc) = sigma_ref * (fc/fc_ref)**exponent``.

    ``exponent`` is a placeholder to be fixed only after independent evidence;
    a constant RCS corresponds to ``exponent = 0``. This is a sensitivity model,
    not validated physical truth.
    """
    if fc_ref_hz <= 0 or fc_hz <= 0:
        raise ValueError("frequencies must be positive")
    return float(sigma_ref_m2 * (fc_hz / fc_ref_hz) ** exponent)


def frequency_bound_rcs(sigma_m2: float):
    """Return a Sionna-compatible constant RCS callable bound to a value.

    The returned callable matches ``f(k_i, k_s, seed) -> drjit.Float`` and is
    independent of the directions, which is the R1 constant-scattering model.
    """

    def _rcs(k_i, k_s, seed):
        # Keep the drjit/tensor type of the input by adding a zero derived from it.
        zero = k_i[0] * 0.0
        return zero + float(sigma_m2)

    return _rcs


__all__ = [
    "segment_power_budget_weights",
    "aggregate_rcs_budget",
    "r2_power_law_sigma",
    "frequency_bound_rcs",
    "SPEED_OF_LIGHT_M_S",
]
