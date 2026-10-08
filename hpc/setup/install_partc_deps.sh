#!/bin/bash
# Install Part C extra dependencies into the col775_t25 conda env.
# Must be run on the HPC LOGIN node (compute nodes can't reach the proxy).
# Idempotent.

set -e

CONDA_ENV="${1:-col775_t25}"
PROXY_HOST="10.10.78.62"
PROXY_PORT="3128"
PROXY_USER="${HPC_PROXY_USER:-mt1230785}"
PROXY_PASS="${HPC_PROXY_PASS:?HPC_PROXY_PASS env var must be set to your IITD HPC password}"
PROXY_URL="http://${PROXY_USER}:${PROXY_PASS}@${PROXY_HOST}:${PROXY_PORT}"

echo "[partc-deps] env=$CONDA_ENV"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export no_proxy="localhost,127.0.0.1,.iitd.ac.in,.iitd.ernet.in"

source "$HOME/condaBaseSetup"
conda activate "$CONDA_ENV"

python - <<'PY'
import sys
print("python :", sys.executable)
import torch
print("torch  :", torch.__version__)
PY

# transformers and cleanfid are the only new deps for Part C.
# Pin to versions compatible with torch 2.5.
pip install --no-input \
    "transformers>=4.40,<4.50" \
    "clean-fid==0.1.35" \
    "tqdm"

python - <<'PY'
import transformers, cleanfid
print("transformers :", transformers.__version__)
print("cleanfid     :", "ok (imported)")
PY

echo "[partc-deps] done."
