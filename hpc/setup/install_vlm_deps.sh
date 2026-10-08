#!/bin/bash
# Install Part B (VLM) extra dependencies into the col775_t25 conda env.
# Must be run on the HPC LOGIN node. Idempotent.
#
# Deps:
#   - peft        (LoRA)
#   - sacrebleu   (BLEU eval)
#   - accelerate  (HF generate/train helpers)
#   - transformers upgrade to a Qwen3-2507-supporting version, if the local wheel
#     is present under $WORKDIR/torch25_wheels/vlm/.
#
# Offline install — all wheels must already be in $WORKDIR/torch25_wheels/vlm/.

set -e

CONDA_ENV="${1:-col775_t25}"
WORKDIR="${HOME}/scratch/col775_a2_hpc"
WHEELS="${WORKDIR}/torch25_wheels/vlm"

PROXY_HOST="10.10.78.62"
PROXY_PORT="3128"
PROXY_USER="${HPC_PROXY_USER:-mt1230785}"
PROXY_PASS="${HPC_PROXY_PASS:?HPC_PROXY_PASS env var must be set to your IITD HPC password}"
PROXY_URL="http://${PROXY_USER}:${PROXY_PASS}@${PROXY_HOST}:${PROXY_PORT}"

echo "[vlm-deps] env=$CONDA_ENV  wheels=$WHEELS"

export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export no_proxy="localhost,127.0.0.1,.iitd.ac.in,.iitd.ernet.in"

source "$HOME/condaBaseSetup"
conda activate "$CONDA_ENV"

python - <<'PY'
import sys, torch, transformers
print("python       :", sys.executable)
print("torch        :", torch.__version__)
print("transformers :", transformers.__version__)
PY

if [ ! -d "$WHEELS" ]; then
    echo "[vlm-deps] ERROR: wheels dir not found: $WHEELS" >&2
    echo "[vlm-deps] Upload wheels first via hpc_upload.py" >&2
    exit 1
fi

# Install offline. --no-index --find-links keeps it deterministic; if a dep
# is missing we surface it here rather than silently hitting PyPI.
# Note: transformers upgrade is optional; only install if the user staged it.
pip install --no-input --no-index --find-links "$WHEELS" \
    peft sacrebleu accelerate safetensors tokenizers || \
pip install --no-input --find-links "$WHEELS" \
    peft sacrebleu accelerate safetensors tokenizers

# Upgrade transformers IFF a newer wheel is present in the vlm wheels dir.
if ls "$WHEELS"/transformers-*.whl >/dev/null 2>&1; then
    echo "[vlm-deps] upgrading transformers from staged wheel"
    pip install --no-input --force-reinstall --no-deps --no-index --find-links "$WHEELS" \
        transformers
fi

python - <<'PY'
import transformers, peft, sacrebleu, accelerate
print("transformers :", transformers.__version__)
print("peft         :", peft.__version__)
print("sacrebleu    :", sacrebleu.__version__)
print("accelerate   :", accelerate.__version__)

# Try to instantiate Qwen3 config to catch version mismatch EARLY.
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",
                                     cache_dir=None, trust_remote_code=True,
                                     local_files_only=True)
    print("Qwen3 config  :", cfg.model_type, cfg.hidden_size)
except Exception as e:
    print("Qwen3 config load deferred (cache may not be synced yet):", e)
PY

echo "[vlm-deps] done."
