"""Shared test helpers."""

from __future__ import annotations

from typing import Any

from fr3isac.config import ExperimentConfig


def minimal_config_dict(**overrides: Any) -> dict[str, Any]:
    """A minimal but fully-formed P0 config used by unit tests.

    Only top-level group keys are deep-merged one level; nested overrides should
    be supplied whole.
    """
    base: dict[str, Any] = {
        "run": {"experiment_id": "unit-test", "master_seed": 1234},
        "environment": {
            "gpu": {
                "host_label": "GPU1",
                "requested_uuid": "GPU-<redacted>",
            },
            "expected_package_versions": {"sionna": "2.2.0", "torch": "2.9.1"},
        },
        "scene": {"name": "tier0_free_space"},
        "radio": {"fc_hz": 3.5e9, "bandwidth_hz": 100e6, "carrier_list_hz": [3.5e9, 12e9, 28e9]},
        "tx_rx": {
            "bs": {"name": "bs", "position_m": [0.0, 0.0, 3.0]},
            "ue": {"name": "ue", "position_m": [10.0, 0.0, 1.5]},
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def make_config(**overrides: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(minimal_config_dict(**overrides))
