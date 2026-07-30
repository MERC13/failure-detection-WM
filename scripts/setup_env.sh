#!/usr/bin/env bash
# Bootstrap the pinned environment on a new machine, then run the G0 gate.
#
#   bash scripts/setup_env.sh
#
# Idempotent: safe to re-run. Every step checks before acting.
#
# This encodes the gotchas Phase 0 already paid for. Read PROGRESS.md for the full
# story; the short version is in the comments at each step.

set -euo pipefail

VENV="${VENV:-$HOME/.venvs/failure-detection-WM}"
DATA_DIR="${DATA_DIR:-$HOME/data/libero}"
SUITE="${SUITE:-libero_object}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\n=== %s\n' "$1"; }

# --- 1. System build toolchain -----------------------------------------------
# robosuite 1.4.0 hard-depends on egl_probe, a C extension. Neither egl_probe nor
# hf-egl-probe publishes a wheel for any platform, so a compiler is mandatory --
# there is no --no-deps shortcut that keeps EGL device probing working.
say "Checking build toolchain"
MISSING=()
for tool in gcc g++ cmake make; do
    command -v "$tool" >/dev/null 2>&1 || MISSING+=("$tool")
done
[ -f /usr/include/EGL/egl.h ] || MISSING+=("EGL headers")

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "MISSING: ${MISSING[*]}"
    echo
    echo "Install them first (needs a real terminal for the sudo password):"
    echo
    echo "  sudo apt update && sudo apt install -y build-essential cmake \\"
    echo "      libegl1-mesa-dev libgl1-mesa-dev libosmesa6 python3-venv"
    echo
    exit 1
fi
echo "OK: gcc g++ cmake make + EGL headers present"

# --- 2. Virtualenv ------------------------------------------------------------
say "Virtualenv at $VENV"
if [ ! -x "$VENV/bin/python" ]; then
    # python3-venv is often absent on Debian/Ubuntu (ensurepip missing); fall back
    # to virtualenv, which needs no root.
    python3 -m venv "$VENV" 2>/dev/null || {
        echo "python3 -m venv failed (ensurepip likely missing); using virtualenv"
        python3 -m pip install --user --quiet virtualenv
        python3 -m virtualenv "$VENV"
    }
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip
echo "OK: $("$PY" --version), pip $("$PY" -m pip --version | awk '{print $2}')"

# --- 3. torch, matched to the hardware ---------------------------------------
# On a GPU box the default PyPI wheel is the CUDA build and is what we want. On a
# CPU-only box the same install drags in ~5 GB of unusable CUDA wheels, so pull
# from the CPU index instead.
say "Installing torch"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    "$PY" -m pip install --quiet torch==2.13.0 torchvision==0.28.0
else
    echo "No GPU detected -- installing CPU-only torch"
    "$PY" -m pip install --quiet torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cpu
fi
"$PY" -c "import torch; print(f'OK: torch {torch.__version__}, cuda_available={torch.cuda.is_available()}')"

# --- 4. Everything else -------------------------------------------------------
# requirements.txt carries the reasons for each pin. The two that bite:
#   robosuite must stay 1.4.x  (1.5 removed SingleArmEnv, which LIBERO imports)
#   mujoco must stay <= 3.3.7  (MjData.qM became .M in 3.4.0; the OSC controller
#                               crashes on first env construction otherwise)
say "Installing pinned requirements"
"$PY" -m pip install --quiet -r "$REPO_ROOT/requirements.txt"
"$PY" -m pip install --quiet transformers  # frozen DINOv2 encoder (Phase 1)
echo "OK"

# --- 5. LIBERO's machine-global config ---------------------------------------
# Importing libero with no ~/.libero/config.yaml runs input() and blocks -- under a
# non-interactive runner it dies with EOFError. Write it directly instead.
say "Provisioning ~/.libero/config.yaml"
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
L="$SITE/libero/libero"
mkdir -p "$HOME/.libero" "$DATA_DIR"
if [ -f "$HOME/.libero/config.yaml" ]; then
    echo "Already exists, leaving it alone:"
else
    cat > "$HOME/.libero/config.yaml" <<EOF
assets: $L/assets
benchmark_root: $L
bddl_files: $L/bddl_files
datasets: $DATA_DIR
init_states: $L/init_files
EOF
    echo "Written:"
fi
cat "$HOME/.libero/config.yaml"

# --- 6. Demos ----------------------------------------------------------------
# One suite only. libero_object alone is 7.0 GB; all 130 tasks would be ~90 GB.
say "Downloading $SUITE demos to $DATA_DIR"
if [ -d "$DATA_DIR/$SUITE" ] && [ "$(ls -A "$DATA_DIR/$SUITE" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Already present ($(du -sh "$DATA_DIR/$SUITE" | cut -f1)), skipping"
else
    "$PY" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id='yifengzhu-hf/LIBERO-datasets', repo_type='dataset',
                  allow_patterns='$SUITE/*', local_dir='$DATA_DIR')
EOF
fi
du -sh "$DATA_DIR/$SUITE"

# --- 7. G0 -------------------------------------------------------------------
# I1 and I5 were proven on the machine Phase 0 ran on and nowhere else. A GPU box
# renders through hardware EGL rather than software rasterization -- a different
# code path -- so the gate is re-run here rather than assumed.
say "Running G0 gate"
cd "$REPO_ROOT"
make env-check

say "Setup complete. Next: make test-slow && make lint"
