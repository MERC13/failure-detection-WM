# The venv lives outside the repo: the repo sits on a /mnt/c 9p mount where
# installing torch/mujoco is dramatically slower than on native ext4.
VENV ?= $(HOME)/.venvs/failure-detection-WM
PY   := $(VENV)/bin/python

# MuJoCo needs an offscreen GL backend. On this WSL2 host, egl resolves through
# WSLg's libd3d12 + mesa d3d12_dri; osmesa is the software fallback.
export MUJOCO_GL ?= egl

.PHONY: env-check g1-check g2-check g3-check test test-slow lint

env-check:
	$(PY) scripts/verify_env.py --config configs/env.yaml

# POLICY_RUN is the train_policy.py run_id whose measured success rate the gate reads;
# COLLECT_RUN is the collect_demos.py run_id whose full-population replay result it reads.
g1-check:
	$(PY) scripts/verify_g1.py --config configs/encoder.yaml \
		--policy-run $(POLICY_RUN) --collect-run $(COLLECT_RUN)

# DYN_RUN is the train_dynamics.py run_id holding dynamics.pt; PROBE_RUN is the
# probe_representation.py run_id supplying SPEC.md §4.2's goal-distance Spearman rho.
g2-check:
	$(PY) scripts/verify_g2.py --config configs/dynamics.yaml \
		--dyn-run $(DYN_RUN) --probe-run $(PROBE_RUN)

# RUN is the run_replay.py run_id whose replayed outcomes the gate reads. The gate never
# replays anything itself: it scores what the simulator already recorded.
g3-check:
	$(PY) scripts/verify_g3.py --config configs/search.yaml --run $(RUN)

test:
	$(PY) -m pytest tests/ -v -m "not slow"

test-slow:
	$(PY) -m pytest tests/ -v

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
