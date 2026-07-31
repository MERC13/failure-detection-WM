"""Shared fixtures. Tests wrap the G0 gate's own check functions rather than reimplementing
the comparisons, so a test can never drift from the gate that guards the invariant.
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def cfg() -> dict:
    config = yaml.safe_load((REPO_ROOT / "configs" / "env.yaml").read_text())
    # MUJOCO_GL must be set before MuJoCo is imported by any check.
    os.environ.setdefault("MUJOCO_GL", config["render_backend"])
    return config


@pytest.fixture(scope="session")
def seed(cfg) -> int:
    return int(cfg["seed"])


@pytest.fixture(scope="session")
def task_and_init_states(cfg):
    from scripts.verify_env import load_suite

    suite = load_suite(cfg)
    task_id = int(cfg["task_id"])
    return suite.get_task(task_id), suite.get_task_init_states(task_id)
