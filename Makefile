# The venv lives outside the repo: the repo sits on a /mnt/c 9p mount where
# installing torch/mujoco is dramatically slower than on native ext4.
VENV ?= $(HOME)/.venvs/failure-detection-WM
PY   := $(VENV)/bin/python

# MuJoCo needs an offscreen GL backend. On this WSL2 host, egl resolves through
# WSLg's libd3d12 + mesa d3d12_dri; osmesa is the software fallback.
export MUJOCO_GL ?= egl

.PHONY: env-check test test-slow lint

env-check:
	$(PY) scripts/verify_env.py --config configs/env.yaml

test:
	$(PY) -m pytest tests/ -v -m "not slow"

test-slow:
	$(PY) -m pytest tests/ -v

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
