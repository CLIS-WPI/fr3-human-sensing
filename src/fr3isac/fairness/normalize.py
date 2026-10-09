"""Cross-band fairness normalizers F1/F2.

This module is deliberately pure arithmetic on units (no Sionna dependency) so
that the fairness contract can be unit-tested and audited independently of the
ray tracer.

Model (documented, provisional engineering choice):

* A single antenna element has ``element_gain_db`` (default 0 dBi, isotropic).
* A planar array of ``n`` half-wavelength-spaced elements has array gain
  ``element_gain_db + 10 log10(n) + 10 log10(aperture_efficiency)``.
* Physical aperture of a half-wavelength planar array is ``n * (lambda/2)^2``.
* F1 (gain normalized): EIRP and receive *gain* are held fixed; the number of
  elements (hence aperture) is whatever realizes those gains at each carrier.
* F2 (aperture constrained): physical apertures are held fixed; the number of
  half-wavelength elements (hence gain, beamwidth and EIRP) follows from the
  carrier. A declared EIRP cap is enforced by reporting the overage rather than
  silently clipping the science.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fr3isac.config import SPEED_OF_LIGHT_M_S, FairnessConfig

ELEMENT_GAIN_DB = 0.0


def wavelength_m(fc_hz: float) -> float:
    return SPEED_OF_LIGHT_M_S / fc_hz


def array_gain_db(num_elements: int, aperture_efficiency: float) -> float:
    if num_elements < 1:
        raise ValueError("num_elements must be >= 1")
    return ELEMENT_GAIN_DB + 10.0 * math.log10(num_elements) + 10.0 * math.log10(
        aperture_efficiency
    )


def elements_for_gain(gain_db: float, aperture_efficiency: float) -> int:
    """Smallest element count realizing at least ``gain_db``."""
    required = 10.0 ** ((gain_db - ELEMENT_GAIN_DB - 10.0 * math.log10(aperture_efficiency)) / 10.0)
    return max(1, int(math.ceil(required - 1e-9)))


def elements_for_aperture(aperture_m2: float, fc_hz: float) -> int:
    """Half-wavelength elements fitting in a physical aperture."""
    lam = wavelength_m(fc_hz)
    cell = (lam / 2.0) ** 2
    return max(1, int(math.floor(aperture_m2 / cell + 1e-9)))


def aperture_for_elements(num_elements: int, fc_hz: float) -> float:
    lam = wavelength_m(fc_hz)
    return num_elements * (lam / 2.0) ** 2


def approx_half_power_beamwidth_deg(num_elements_1d: int, fc_hz: float) -> float:
    """Broadside HPBW of a uniform 1-D half-wavelength array (radians approx).

    ``Theta ~= 0.886 * lambda / (N * d)`` with ``d = lambda/2`` gives
    ``Theta ~= 1.772 / N`` radians.
    """
    n = max(1, num_elements_1d)
    return math.degrees(1.772 / n)


@dataclass(frozen=True)
class ArraySpec:
    fc_hz: float
    num_elements: int
    gain_db: float
    aperture_m2: float
    beamwidth_deg: float
    conducted_power_dbm: float
    eirp_dbm: float
    eirp_cap_exceeded_db: float = 0.0


@dataclass(frozen=True)
class FairnessResult:
    regime: str
    fc_hz: float
    bs: ArraySpec
    ue: ArraySpec


def _conducted_from_eirp(eirp_dbm: float, gain_db: float) -> float:
    return eirp_dbm - gain_db


def _eirp_from_conducted(conducted_dbm: float, gain_db: float) -> float:
    return conducted_dbm + gain_db


def _cap_overage(eirp_dbm: float, cap_dbm: float | None) -> float:
    if cap_dbm is None:
        return 0.0
    return max(0.0, eirp_dbm - cap_dbm)


def normalize(cfg: FairnessConfig, fc_hz: float) -> FairnessResult:
    """Return the normalized BS/UE array specs for one carrier and regime."""
    if cfg.regime == "F1":
        # Fix EIRP and receive gains; derive element counts and apertures.
        tx_gain = cfg.eirp_dbm - cfg.conducted_power_dbm
        bs_n = min(cfg.max_elements, elements_for_gain(tx_gain, cfg.aperture_efficiency))
        bs_gain = array_gain_db(bs_n, cfg.aperture_efficiency)
        ue_n = min(cfg.max_elements, elements_for_gain(cfg.ue_rx_gain_db, cfg.aperture_efficiency))
        ue_gain = array_gain_db(ue_n, cfg.aperture_efficiency)
        bs_conducted = cfg.conducted_power_dbm
        bs_eirp = cfg.eirp_dbm
    elif cfg.regime == "F2":
        # Fix physical apertures; derive element counts, gains and EIRP.
        bs_n = min(cfg.max_elements, elements_for_aperture(cfg.bs_aperture_m2, fc_hz))
        bs_gain = array_gain_db(bs_n, cfg.aperture_efficiency)
        ue_n = min(cfg.max_elements, elements_for_aperture(cfg.ue_aperture_m2, fc_hz))
        ue_gain = array_gain_db(ue_n, cfg.aperture_efficiency)
        bs_conducted = cfg.conducted_power_dbm
        bs_eirp = _eirp_from_conducted(bs_conducted, bs_gain)
    else:  # pragma: no cover - config validator restricts to F1/F2
        raise ValueError(f"unsupported fairness regime {cfg.regime!r}")

    bs_over = _cap_overage(bs_eirp, cfg.eirp_cap_dbm)
    bs1d = max(1, int(round(math.sqrt(bs_n))))
    ue1d = max(1, int(round(math.sqrt(ue_n))))

    return FairnessResult(
        regime=cfg.regime,
        fc_hz=fc_hz,
        bs=ArraySpec(
            fc_hz=fc_hz,
            num_elements=bs_n,
            gain_db=bs_gain,
            aperture_m2=aperture_for_elements(bs_n, fc_hz),
            beamwidth_deg=approx_half_power_beamwidth_deg(bs1d, fc_hz),
            conducted_power_dbm=bs_conducted,
            eirp_dbm=bs_eirp,
            eirp_cap_exceeded_db=bs_over,
        ),
        ue=ArraySpec(
            fc_hz=fc_hz,
            num_elements=ue_n,
            gain_db=ue_gain,
            aperture_m2=aperture_for_elements(ue_n, fc_hz),
            beamwidth_deg=approx_half_power_beamwidth_deg(ue1d, fc_hz),
            conducted_power_dbm=cfg.conducted_power_dbm,
            eirp_dbm=_eirp_from_conducted(cfg.conducted_power_dbm, ue_gain),
        ),
    )


__all__ = [
    "ELEMENT_GAIN_DB",
    "wavelength_m",
    "array_gain_db",
    "elements_for_gain",
    "elements_for_aperture",
    "aperture_for_elements",
    "approx_half_power_beamwidth_deg",
    "ArraySpec",
    "FairnessResult",
    "normalize",
]
