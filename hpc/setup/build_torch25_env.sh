#!/bin/bash
# Build (or repair) the col775_t25 conda env on IITD HPC so it satisfies the
# assignment spec: Python 3.10 + PyTorch 2.5.0+cu118 + torchvision 0.20.0+cu118.
#
# Strategy: clone the already-working `col775_a2` env (torch 2.3.1+cu121) so we
# inherit all the non-torch deps (numpy, scikit-learn, PIL, paramiko, ...), then
# swap the CUDA-12 runtime wheels for CUDA-11 runtime wheels and force-reinstall
# torch/torchvision from the local cu118 wheels.
#
# Requires:
#   $HOME/torch25_wheels/                         ← torch/torchvision cu118 wheels (uploaded from local repo)
#   $HOME/torch25_wheels/cu11_deps/               ← nvidia-*-cu11 wheels (also uploaded; IITD proxy blocks PyPI)
#
# Usage:  bash hpc_scripts/build_torch25_env.sh
set -euo pipefail

SRC="$HOME/.conda/envs/col775_a2"
DST="$HOME/.conda/envs/col775_t25"
WHEELS="$HOME/torch25_wheels"
CU11_DEPS="$WHEELS/cu11_deps"
PY="$DST/bin/python"
PIP="$PY -m pip"

if [ ! -d "$SRC" ]; then
    echo "[build] source env $SRC missing — run setup_hpc.sh first"; exit 1
fi
if [ ! -d "$CU11_DEPS" ] || [ -z "$(ls -A "$CU11_DEPS" 2>/dev/null)" ]; then
    echo "[build] $CU11_DEPS is missing/empty. Upload the nvidia-*-cu11 wheels first:"
    echo "        pip download (on a machine with PyPI access) the packages listed"
    echo "        below into torch25_wheels/cu11_deps, then SFTP that dir to HPC."
    echo "        Required: nvidia-cuda-runtime-cu11, nvidia-cublas-cu11, nvidia-cudnn-cu11,"
    echo "        nvidia-cuda-cupti-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cufft-cu11,"
    echo "        nvidia-curand-cu11, nvidia-cusolver-cu11, nvidia-cusparse-cu11,"
    echo "        nvidia-nccl-cu11, nvidia-nvtx-cu11."
    exit 1
fi

echo "=== (re)building $DST from clone of $SRC ==="
rm -rf "$DST"
cp -r "$SRC" "$DST"
# Rewrite shebangs / conda-meta references that still point at the source env.
grep -rIl "envs/col775_a2\b" "$DST" 2>/dev/null \
    | xargs -r sed -i "s|envs/col775_a2\b|envs/col775_t25|g" || true

echo "=== BEFORE:"
$PY -c "import sys; print('py', sys.version.split()[0])"

echo "=== drop CUDA-12 runtime wheels inherited from source env ==="
$PIP uninstall -y \
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
    nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
    nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 triton \
    torch torchvision 2>&1 | tail -30 || true

echo "=== install CUDA-11 runtime wheels from $CU11_DEPS ==="
$PIP install --no-index --find-links "$CU11_DEPS" \
    nvidia-cuda-runtime-cu11 nvidia-cublas-cu11 nvidia-cudnn-cu11 \
    nvidia-cuda-cupti-cu11  nvidia-cuda-nvrtc-cu11 nvidia-cufft-cu11 \
    nvidia-curand-cu11      nvidia-cusolver-cu11   nvidia-cusparse-cu11 \
    nvidia-nccl-cu11        nvidia-nvtx-cu11

echo "=== install torch 2.5.0+cu118 + torchvision 0.20.0+cu118 from local wheels ==="
$PIP install --no-deps "$WHEELS/torch-2.5.0+cu118-cp310-cp310-linux_x86_64.whl"
$PIP install --no-deps "$WHEELS/torchvision-0.20.0+cu118-cp310-cp310-linux_x86_64.whl"

echo "=== AFTER (CPU import only — CUDA check needs a GPU node) ==="
$PY -c "
import sys, torch, torchvision, numpy, PIL
print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('numpy', numpy.__version__)
print('PIL', PIL.__version__)
print('cuda available (login-node, expected False):', torch.cuda.is_available())
"
echo "=== DONE. Next: submit hpc/submit_probe.pbs (or gpu_probe.pbs) to verify CUDA on an A100. ==="
