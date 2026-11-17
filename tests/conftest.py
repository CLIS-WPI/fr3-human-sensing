"""Shared pytest fixtures for the FR3 ISAC test-suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from fr3isac.config import ExperimentConfig
from tests.helpers import minimal_config_dict

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(minimal_config_dict())


@pytest.fixture()
def repo_root() -> Path:
    return REPO_ROOT
