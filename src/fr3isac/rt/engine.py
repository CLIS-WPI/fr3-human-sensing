"""Background ray-tracing engine: PathSolver wrapper and path diagnostics.

Sionna path tensor conventions (verified against Sionna 2.2.0 in P0):

* ``paths.a`` is a ``(real, imag)`` tuple of tensors with shape
  ``[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, (num_time_steps)]``.
* ``paths.tau`` / ``paths.valid`` / ``paths.doppler`` have shape
  ``[num_rx, num_tx, num_paths, (num_time_steps)]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sionna.rt import Paths, PathSolver, Scene

from fr3isac.config import SPEED_OF_LIGHT_M_S, ExperimentConfig
from fr3isac.phy.ofdm import free_space_channel_coefficient


def paths_a_complex(paths: Paths) -> np.ndarray:
    """Coherent complex path coefficients as a complex numpy array."""
    a = paths.a
    if isinstance(a, tuple):
        re, im = a
        return np.asarray(re) + 1j * np.asarray(im)
    arr = np.asarray(a)
    if np.iscomplexobj(arr):
        return arr
    raise TypeError(f"unexpected paths.a type {type(a)!r}")


def paths_tau(paths: Paths) -> np.ndarray:
    return np.asarray(paths.tau)


def paths_doppler(paths: Paths) -> np.ndarray:
    return np.asarray(paths.doppler)


def paths_valid(paths: Paths) -> np.ndarray:
    return np.asarray(paths.valid)


def _first_link_1d(arr: np.ndarray) -> np.ndarray:
    """Return the first rx/tx link from a ``[num_rx, num_tx, num_paths(, time)]`` tensor.

    ``tau``/``valid``/``doppler`` have the path axis in the last (or second to
    last, if a time axis is present) position.
    """
    if arr.size == 0:
        return arr.reshape(-1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 3:  # [num_rx, num_tx, num_paths]
        return arr.reshape(-1, arr.shape[-1])[0]
    if arr.ndim == 4:  # [num_rx, num_tx, num_paths, num_time]
        return arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0][:, 0]
    raise ValueError(f"unexpected path tensor shape {arr.shape}")


def link_valid(paths: Paths) -> np.ndarray:
    return _first_link_1d(paths_valid(paths))


def link_tau(paths: Paths) -> np.ndarray:
    return _first_link_1d(paths_tau(paths))


def link_doppler(paths: Paths) -> np.ndarray:
    return _first_link_1d(paths_doppler(paths))


def element_coefficients(paths: Paths) -> np.ndarray:
    """Complex coefficient for the first receive/transmit antenna elements."""
    a = paths_a_complex(paths)
    # a: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths(, time)]
    per_path = a[0, 0, 0, 0]
    if per_path.ndim == 2:
        per_path = per_path[:, 0]
    return per_path


def link_coefficients(paths: Paths) -> np.ndarray:
    """Effective single-stream coefficients: coherent sum over antenna elements."""
    a = paths_a_complex(paths)
    summed = a.sum(axis=(0, 1, 2, 3))
    if summed.ndim == 2:
        summed = summed[:, 0]
    return summed


def valid_path_coefficients(paths: Paths, mode: str = "element") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(a, tau)`` for valid paths only, flattened to 1-D."""
    a = element_coefficients(paths) if mode == "element" else link_coefficients(paths)
    tau = link_tau(paths)
    valid = link_valid(paths)
    if a.shape[0] != valid.shape[0]:
        raise AssertionError(f"path coefficient/validity length mismatch: {a.shape} vs {valid.shape}")
    return a[valid], tau[valid]


@dataclass(frozen=True)
class PathDiagnostics:
    num_valid_paths: int
    path_gain_db: float
    coherent_gain_db: float
    rms_delay_spread_s: float
    mean_delay_s: float
    min_delay_s: float
    max_delay_s: float


def path_diagnostics(paths: Paths, eps: float = 1e-30) -> PathDiagnostics:
    a, tau = valid_path_coefficients(paths, mode="link")
    power = np.abs(a) ** 2
    total_power = float(power.sum())
    coherent = float(abs(a.sum()) ** 2)
    if total_power > eps and tau.size:
        weights = power / total_power
        mean_delay = float(np.sum(weights * tau))
        rms = float(np.sqrt(np.sum(weights * (tau - mean_delay) ** 2)))
    else:
        mean_delay = 0.0
        rms = 0.0
    return PathDiagnostics(
        num_valid_paths=int(a.size),
        path_gain_db=float(10.0 * np.log10(total_power + eps)),
        coherent_gain_db=float(10.0 * np.log10(coherent + eps)),
        rms_delay_spread_s=rms,
        mean_delay_s=mean_delay,
        min_delay_s=float(tau.min()) if tau.size else float("nan"),
        max_delay_s=float(tau.max()) if tau.size else float("nan"),
    )


def run_background_paths(scene: Scene, config: ExperimentConfig, seed: int) -> Paths:
    solver = PathSolver(deterministic=config.run.deterministic)
    rt = config.rt
    return solver(
        scene,
        max_depth=rt.max_depth,
        samples_per_src=rt.samples_per_tx,
        synthetic_array=rt.synthetic_array,
        los=rt.los,
        specular_reflection=rt.specular_reflection,
        diffuse_reflection=rt.diffuse_reflection,
        refraction=rt.refraction,
        diffraction=rt.diffraction,
        seed=int(seed),
    )


def free_space_comparison(
    paths: Paths,
    fc_hz: float,
    distance_m: float,
    delay_tol_s: float = 1e-11,
    gain_tol_db: float = 0.05,
) -> dict[str, object]:
    """Compare a single-LOS-path channel against Friis + geometric delay."""
    a, tau = valid_path_coefficients(paths, mode="element")
    if a.size != 1:
        raise AssertionError(f"expected exactly one LOS path, got {a.size}")
    measured_tau = float(tau[0])
    analytic_tau = distance_m / SPEED_OF_LIGHT_M_S
    measured_gain = float(abs(a[0]) ** 2)
    analytic_gain = float(abs(free_space_channel_coefficient(fc_hz, distance_m)) ** 2)
    gain_err_db = float(10.0 * np.log10(measured_gain / analytic_gain))
    return {
        "measured_tau_s": measured_tau,
        "analytic_tau_s": analytic_tau,
        "delay_abs_err_s": abs(measured_tau - analytic_tau),
        "delay_ok": abs(measured_tau - analytic_tau) <= delay_tol_s,
        "measured_gain": measured_gain,
        "analytic_gain": analytic_gain,
        "gain_err_db": gain_err_db,
        "gain_ok": abs(gain_err_db) <= gain_tol_db,
    }


__all__ = [
    "paths_a_complex",
    "paths_tau",
    "paths_doppler",
    "paths_valid",
    "link_valid",
    "link_tau",
    "link_doppler",
    "element_coefficients",
    "link_coefficients",
    "valid_path_coefficients",
    "PathDiagnostics",
    "path_diagnostics",
    "run_background_paths",
    "free_space_comparison",
]
