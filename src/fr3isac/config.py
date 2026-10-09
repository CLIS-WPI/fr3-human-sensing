"""Typed, validated experiment configuration.

A single configuration fully describes a run. Fields are grouped to mirror the
plan's mandatory config groups. Free-form unvalidated dictionaries are not
allowed for production runs.

Numerical values here are *provisional P0 engineering placeholders* until the
freeze gate after the pilot. They are
never treated as frozen scientific constants.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fr3isac import CONFIG_SCHEMA_VERSION

SPEED_OF_LIGHT_M_S = 299_792_458.0


class StrictModel(BaseModel):
    """Base model: reject unknown keys and validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# run / environment
# --------------------------------------------------------------------------- #
class RunConfig(StrictModel):
    experiment_id: str
    tag: str = ""
    output_root: Path = Path("outputs")
    deterministic: bool = True
    master_seed: int = 0
    cache_policy: Literal["read_write", "read_only", "refresh"] = "read_write"
    telemetry: bool = True


class GpuBinding(StrictModel):
    """Host GPU identity.

    ``requested_uuid`` is part of the reproducibility contract and enters the
    config hash. The ``observed_*`` fields are runtime facts filled by the
    health check and are deliberately excluded from the scientific hash.
    """

    host_label: str
    requested_uuid: str
    observed_uuid: str | None = None
    model_name: str | None = None
    driver_version: str | None = None
    memory_mb: int | None = None
    mig: str | None = None


class EnvironmentConfig(StrictModel):
    gpu: GpuBinding
    backend_variant: str = "cuda"
    gpu_required: bool = True
    expected_package_versions: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
class SceneConfig(StrictModel):
    name: str
    path: Path | None = None
    tier: Literal["tier0_calibration", "tier1_primary", "tier2_replication"] = "tier0_calibration"
    scale: float = 1.0
    unit: Literal["m", "cm", "mm"] = "m"
    material_map_version: str = "p0-v1"
    region_label: str = "calibration"
    geometry_seed: int = 0
    scene_hash: str | None = None


# --------------------------------------------------------------------------- #
# radio / fairness
# --------------------------------------------------------------------------- #
class RadioConfig(StrictModel):
    fc_hz: float
    bandwidth_hz: float
    subcarrier_spacing_hz: float = 120e3
    carrier_list_hz: list[float] = Field(default_factory=list)
    noise_figure_db: float = 7.0
    temperature_k: float = 290.0

    @field_validator("fc_hz", "bandwidth_hz", "subcarrier_spacing_hz")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @field_validator("carrier_list_hz")
    @classmethod
    def _positive_list(cls, v: list[float]) -> list[float]:
        if any(c <= 0 for c in v):
            raise ValueError("carrier frequencies must be positive")
        return v


class FairnessConfig(StrictModel):
    """F1/F2 normalizers."""

    regime: Literal["F1", "F2"] = "F1"
    eirp_dbm: float = 40.0
    conducted_power_dbm: float = 30.0
    eirp_cap_dbm: float | None = None
    bs_rx_gain_db: float = 20.0
    ue_rx_gain_db: float = 0.0
    bs_aperture_m2: float = 0.25
    ue_aperture_m2: float = 0.0025
    aperture_efficiency: float = 0.7
    bs_pattern: str = "isotropic"
    ue_pattern: str = "isotropic"
    polarization: str = "constant_cpm"
    beam_policy: str = "fixed_boresight"
    observation_time_s: float = 0.01
    max_elements: int = 4096
    noise_figure_db: float = 7.0

    @model_validator(mode="after")
    def _check_regime(self) -> FairnessConfig:
        if self.regime == "F1" and self.eirp_dbm is None:
            raise ValueError("F1 requires an EIRP target")
        if self.regime == "F2" and self.conducted_power_dbm is None:
            raise ValueError("F2 requires conducted power / aperture policy")
        return self


# --------------------------------------------------------------------------- #
# tx/rx
# --------------------------------------------------------------------------- #
Vec3 = tuple[float, float, float]


class RadioNodeConfig(StrictModel):
    name: str
    position_m: Vec3
    orientation_deg: Vec3 = (0.0, 0.0, 0.0)
    array_rows: int = 1
    array_cols: int = 1
    element_spacing_wavelengths: float = 0.5
    beam_index: int = 0


class TxRxConfig(StrictModel):
    bs: RadioNodeConfig
    ue: RadioNodeConfig
    sampling_policy: str = "replay_across_bands"


# --------------------------------------------------------------------------- #
# ray tracing
# --------------------------------------------------------------------------- #
class RTConfig(StrictModel):
    max_depth: int = 3
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    diffraction: bool = False
    refraction: bool = False
    samples_per_tx: int = 10_000
    synthetic_array: bool = True
    cell_size_m: float = 1.0
    cell_height_m: float = 1.0


class SensingRTConfig(StrictModel):
    max_depth: int = 1
    samples_per_src: int = 10_000
    buffer_size: int = 1_000_000
    los: bool = True
    specular_reflection: bool = True
    refraction: bool = True
    diffraction: bool = False
    diffuse_reflection: bool = False
    deterministic: bool = True


# --------------------------------------------------------------------------- #
# target
# --------------------------------------------------------------------------- #
class TargetConfig(StrictModel):
    model: Literal["native", "matched_point", "articulated"] = "native"
    object_type: str = "human"
    target_model_version: str = "p0-v1"
    motion_id: str = "static"
    rcs_tier: Literal["R0", "R1-P", "R1-A", "R2"] = "R0"
    rcs_model_version: str = "p0-v1"
    aggregate_rcs_budget_m2: float | None = None
    normalization_ensemble: str = "fixed_aspect_pose_v1"
    phase_policy: Literal["continuous_absolute_delay", "snapshot_anchors"] = (
        "continuous_absolute_delay"
    )
    frequency_binding: Literal["native", "band_bound"] = "native"
    validation_version: str = "none"
    random_sigma_s: bool = True
    random_xpr: bool = True
    random_phases: bool = True
    segments: list[str] = Field(default_factory=lambda: ["torso", "right_forearm"])
    arm_amplitude_m: float = 0.25
    arm_omega_rad_s: float = 3.141592653589793


# --------------------------------------------------------------------------- #
# phy
# --------------------------------------------------------------------------- #
class PHYConfig(StrictModel):
    fft_size: int = 1024
    occupied_subcarriers: int = 816
    cp_length: int = 72
    num_ofdm_symbols: int = 14
    mcs_index: int = 10
    csi: Literal["perfect", "estimated"] = "perfect"
    equalizer: str = "lmmse"
    pilot_overhead_fraction: float = 0.143
    guard_overhead_fraction: float = 0.07
    beam_training_overhead_fraction: float = 0.0
    sensing_overhead_fraction: float = 0.1
    goodput_denominator: Literal["elapsed_observation_time"] = "elapsed_observation_time"
    num_replays: int = 4

    @model_validator(mode="after")
    def _check_grid(self) -> PHYConfig:
        if self.occupied_subcarriers > self.fft_size:
            raise ValueError("occupied_subcarriers exceeds fft_size")
        if not 0.0 <= self.pilot_overhead_fraction < 1.0:
            raise ValueError("pilot_overhead_fraction must be in [0,1)")
        return self


# --------------------------------------------------------------------------- #
# sensing
# --------------------------------------------------------------------------- #
class SensingConfig(StrictModel):
    cpi_s: float = 0.01
    detector: str = "matched_filter_range_doppler"
    window: str = "full_cpi"
    pfa_target: float = 1e-3
    pfa_unit: Literal["per_cell", "per_frame", "per_decision_window"] = "per_decision_window"
    calibration_split: str = "held_out_trajectories"
    deadline_s: float = 0.02
    miss_limit: float = 0.1
    clutter_model: str = "none"
    residual_self_interference_db: float | None = None
    phase_time_origin: str = "absolute_common"
    decision_hop_s: float = 0.01
    search_range_min_m: float = 0.0
    search_range_max_m: float | None = None
    search_vmin_mps: float = 0.1
    search_vmax_mps: float = 5.0
    pfa_calibration_windows: int = 10000
    pfa_validation_windows: int = 10000


# --------------------------------------------------------------------------- #
# sampling / analysis / validation
# --------------------------------------------------------------------------- #
class SamplingConfig(StrictModel):
    scenario_count: int = 10
    placement_distribution: str = "uniform_room_v1"
    distance_min_m: float = 3.0
    distance_max_m: float = 8.0
    target_height_m: float = 1.5
    trajectory_ids: list[str] = Field(default_factory=list)
    crn_policy: Literal["common_random_numbers"] = "common_random_numbers"


class AnalysisConfig(StrictModel):
    goodput_floor_mbps: float = 100.0
    miss_ceiling: float = 0.1
    resource_budget: str = "declared_shared_schedule_v1"
    feasible_set_rule: str = "meets_all_frozen_requirements"
    tie_policy: str = "min_cost_with_tolerance"
    regret_reference: str = "articulated"
    cluster_ids: str = "scenario_id"
    confidence_level: float = 0.95
    bootstrap_samples: int = 10_000
    contrasts: list[str] = Field(default_factory=lambda: ["FR3_vs_FR1", "FR3_vs_FR2"])
    crossover_variables: list[str] = Field(default_factory=lambda: ["distance_m", "eirp_dbm"])
    adequacy_miss_margin: float = 0.05
    adequacy_goodput_rel_margin: float = 0.05
    cost_tolerance_rel: float = 0.05


class ValidationConfig(StrictModel):
    benchmark_id: str = "none"
    benchmark_version: str = "none"
    benchmark_hash: str = "none"
    source: str = "none"
    rights: str = "unknown"
    acquisition_status: Literal["planned", "unknown", "unsuitable", "usable", "acquired"] = (
        "planned"
    )
    coordinate_transform: str = "identity_pending"
    time_transform: str = "identity_pending"
    carrier_waveform_compatible: bool = False
    calibration_split: str = "none"
    test_split: str = "none"
    observables: list[str] = Field(default_factory=list)
    tolerances: dict[str, float] = Field(default_factory=dict)
    uncertainty: str = "unquantified"
    applicability_domain: str = "unvalidated"
    per_observable_status: Literal["planned", "incomplete", "pass", "partial", "fail"] = "planned"
    scope_decision: Literal["pending", "deployment_validated", "model_sensitivity"] = "pending"
    scope_decision_rationale: str = ""


# --------------------------------------------------------------------------- #
# root
# --------------------------------------------------------------------------- #
# Runtime-only fields excluded from the scientific config hash.
_RUNTIME_EXCLUDE: dict[str, set[str]] = {
    "environment": {"gpu": {"observed_uuid", "model_name", "driver_version", "memory_mb", "mig"}},
    "scene": {"scene_hash"},
}


class ExperimentConfig(StrictModel):
    schema_version: str = CONFIG_SCHEMA_VERSION
    run: RunConfig
    environment: EnvironmentConfig
    scene: SceneConfig
    radio: RadioConfig
    tx_rx: TxRxConfig
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    rt: RTConfig = Field(default_factory=RTConfig)
    sensing_rt: SensingRTConfig = Field(default_factory=SensingRTConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    phy: PHYConfig = Field(default_factory=PHYConfig)
    sensing: SensingConfig = Field(default_factory=SensingConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @model_validator(mode="after")
    def _check_bandwidth(self) -> ExperimentConfig:
        if self.radio.bandwidth_hz > self.radio.fc_hz:
            raise ValueError("bandwidth_hz cannot exceed fc_hz")
        return self

    def canonical_dict(self) -> dict[str, Any]:
        """Full config as JSON-native data with runtime-only fields removed."""
        data = json.loads(self.model_dump_json())
        for section, excl in _RUNTIME_EXCLUDE.items():
            if section not in data:
                continue
            if isinstance(excl, set):
                for key in excl:
                    data[section].pop(key, None)
            else:
                for key, keys in excl.items():
                    if key in data[section]:
                        for k in keys:
                            data[section][key].pop(k, None)
        return data

    def config_hash(self) -> str:
        """Stable SHA-256 of canonical scientific config (short first 16 hex)."""
        blob = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, indent=2)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping, got {type(raw).__name__}")
    return ExperimentConfig.model_validate(raw)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    """Write a config back to YAML in a canonical, stable order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            json.loads(config.model_dump_json()),
            fh,
            sort_keys=True,
            default_flow_style=False,
        )


__all__ = [
    "SPEED_OF_LIGHT_M_S",
    "RunConfig",
    "GpuBinding",
    "EnvironmentConfig",
    "SceneConfig",
    "RadioConfig",
    "FairnessConfig",
    "RadioNodeConfig",
    "TxRxConfig",
    "RTConfig",
    "SensingRTConfig",
    "TargetConfig",
    "PHYConfig",
    "SensingConfig",
    "SamplingConfig",
    "AnalysisConfig",
    "ValidationConfig",
    "ExperimentConfig",
    "load_config",
    "dump_config",
]
