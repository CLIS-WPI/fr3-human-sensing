"""Target builders: native 3GPP baseline, matched point, articulated segments."""

from __future__ import annotations

import mitsuba as mi
import numpy as np
from sionna.rt.rcs import ConstantRCSSensingTarget, SensingTarget, TR38901SensingTarget

from fr3isac.config import TargetConfig
from fr3isac.geometry import (
    DEFAULT_SEGMENT_WEIGHTS,
    SEGMENT_OFFSETS,
    ArmTrajectorySample,
    analytic_arm_trajectory,
)
from fr3isac.rcs.models import aggregate_rcs_budget, segment_power_budget_weights


def _p3(v: tuple[float, float, float]) -> mi.Point3f:
    return mi.Point3f(float(v[0]), float(v[1]), float(v[2]))


def _v3(v: tuple[float, float, float]) -> mi.Vector3f:
    return mi.Vector3f(float(v[0]), float(v[1]), float(v[2]))


def build_point_target(
    name: str,
    position: tuple[float, float, float],
    sigma_m2: float,
    velocity: tuple[float, float, float] | None = None,
    xpr_db: float | None = None,
) -> ConstantRCSSensingTarget:
    return ConstantRCSSensingTarget(
        name=name,
        sigma=float(sigma_m2),
        xpr_db=xpr_db,
        position=_p3(position),
        velocity=_v3(velocity) if velocity is not None else None,
    )


def build_human_target(
    name: str,
    position: tuple[float, float, float],
    config: TargetConfig,
    velocity: tuple[float, float, float] | None = None,
) -> TR38901SensingTarget:
    """Native standardized human baseline (R0). One scattering point."""
    return TR38901SensingTarget(
        name=name,
        object_type=config.object_type,
        model_type=2,  # single-scattering-point human model
        position=_p3(position),
        velocity=_v3(velocity) if velocity is not None else None,
        random_sigma_s=config.random_sigma_s,
        random_xpr=config.random_xpr,
        random_phases=config.random_phases,
    )


def single_scattering_point_count(target: SensingTarget | TR38901SensingTarget) -> int:
    """Number of scattering points actually attached to a Sionna sensing target."""
    model = target.scattering_model
    lcs = np.asarray(model.spst.lcs_positions)
    return int(lcs.reshape(-1, 3).shape[0])


def segment_sigmas(total_sigma_m2: float, segments: list[str]) -> dict[str, float]:
    weights = [DEFAULT_SEGMENT_WEIGHTS.get(s, 1.0 / len(segments)) for s in segments]
    weights = segment_power_budget_weights(weights)
    return dict(zip(segments, aggregate_rcs_budget(total_sigma_m2, weights), strict=False))


def build_articulated_segments(
    root_position: tuple[float, float, float],
    total_sigma_m2: float,
    config: TargetConfig,
    moving: dict[str, tuple[float, float, float]] | None = None,
) -> list[ConstantRCSSensingTarget]:
    """Build one torso plus the configured segments with a matched RCS budget.

    ``moving`` maps segment names to velocities; all other segments are static.
    Positions are offset from ``root_position`` by the P0 skeleton. Each segment
    is an independent sensing target so limb motion is not conflated with root
    translation (section 14.1).
    """
    moving = moving or {}
    sigmas = segment_sigmas(total_sigma_m2, config.segments)
    targets: list[ConstantRCSSensingTarget] = []
    for seg in config.segments:
        offset = SEGMENT_OFFSETS.get(seg, (0.0, 0.0, 1.0))
        pos = tuple(root_position[i] + offset[i] for i in range(3))
        targets.append(
            build_point_target(f"{seg}", pos, sigmas[seg], velocity=moving.get(seg))
        )
    return targets


__all__ = [
    "SEGMENT_OFFSETS",
    "DEFAULT_SEGMENT_WEIGHTS",
    "build_point_target",
    "build_human_target",
    "single_scattering_point_count",
    "segment_sigmas",
    "build_articulated_segments",
    "ArmTrajectorySample",
    "analytic_arm_trajectory",
]
