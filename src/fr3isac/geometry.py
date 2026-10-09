"""Pure-geometry helpers (no ray tracer): target skeleton, motion and LoS tests.

Used by the localization and estimand modules. Kept dependency-light (numpy only)
so the estimand recomputation runs on CPU without Sionna.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCENE_JSON = Path("scenes/tier1_primary/scene.json")

# Target skeleton (metres, relative to the sampled target position).
SEGMENT_OFFSETS: dict[str, tuple[float, float, float]] = {
    "torso": (0.0, 0.0, 0.0),
    "right_forearm": (-0.20, 0.35, -0.20),
}
DEFAULT_SEGMENT_WEIGHTS: dict[str, float] = {"torso": 0.7, "right_forearm": 0.3}

TARGET_HEIGHT_M = 1.4
ROI_X = (2.0, 7.5)
ROI_Y = (1.5, 6.5)

NSLOW = 128
CPI_S = 0.10
PRF_SLOW_HZ = 1280.0
DEV_MOTIONS = ((0.18, 0.7), (0.20, 0.5))
TEST_MOTIONS = ((0.18, 0.9), (0.25, 0.7))


def load_descriptor(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or SCENE_JSON).read_text(encoding="utf-8"))


def blockers(descriptor: dict) -> list[dict]:
    """Scene objects that can occlude a segment (floor and ceiling excluded)."""
    return [o for o in descriptor["objects"] if o["name"] not in ("floor", "ceiling")]


def bs_position(descriptor: dict | None = None) -> np.ndarray:
    return np.asarray((descriptor or load_descriptor())["bs"]["position_m"], dtype=float)


def segment_aabb_intersect(p0, p1, center, size) -> bool:
    """Slab test: does the segment p0->p1 intersect the axis-aligned box?"""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    lo = np.asarray(center, dtype=float) - np.asarray(size, dtype=float) / 2.0
    hi = np.asarray(center, dtype=float) + np.asarray(size, dtype=float) / 2.0
    d = p1 - p0
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return False
            continue
        t1 = (lo[i] - p0[i]) / d[i]
        t2 = (hi[i] - p0[i]) / d[i]
        t1, t2 = min(t1, t2), max(t1, t2)
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return False
    return True


@dataclass(frozen=True)
class ArmTrajectorySample:
    t_s: float
    torso_position: tuple[float, float, float]
    torso_velocity: tuple[float, float, float]
    forearm_position: tuple[float, float, float]
    forearm_velocity: tuple[float, float, float]


def analytic_arm_trajectory(
    root_position: tuple[float, float, float],
    times_s: np.ndarray,
    amplitude_m: float = 0.25,
    omega_rad_s: float = 2.0 * np.pi * 0.5,
) -> list[ArmTrajectorySample]:
    """Deterministic arm swing: torso static, forearm moving along x.

    Forearm position ``x = root_x + x0 + A sin(omega t)`` with the analytic
    velocity derivative. Synthetic verification motion, not recorded motion.
    """
    torso_off = SEGMENT_OFFSETS["torso"]
    fa_off = SEGMENT_OFFSETS["right_forearm"]
    torso_pos = tuple(root_position[i] + torso_off[i] for i in range(3))
    samples: list[ArmTrajectorySample] = []
    for t in np.asarray(times_s, dtype=float):
        dx = amplitude_m * np.sin(omega_rad_s * t)
        dvx = amplitude_m * omega_rad_s * np.cos(omega_rad_s * t)
        fa_pos = (root_position[0] + fa_off[0] + dx,
                  root_position[1] + fa_off[1],
                  root_position[2] + fa_off[2])
        samples.append(ArmTrajectorySample(float(t), torso_pos, (0.0, 0.0, 0.0), fa_pos,
                                           (float(dvx), 0.0, 0.0)))
    return samples


def static_sample(root_position, amplitude_m: float = 0.1, omega_rad_s: float = 2 * np.pi * 0.5):
    return analytic_arm_trajectory(root_position, np.array([0.0]), amplitude_m, omega_rad_s)[0]


__all__ = [
    "SCENE_JSON", "SEGMENT_OFFSETS", "DEFAULT_SEGMENT_WEIGHTS", "TARGET_HEIGHT_M",
    "ROI_X", "ROI_Y", "NSLOW", "CPI_S", "PRF_SLOW_HZ", "DEV_MOTIONS", "TEST_MOTIONS",
    "load_descriptor", "blockers", "bs_position", "segment_aabb_intersect",
    "ArmTrajectorySample", "analytic_arm_trajectory", "static_sample",
]
