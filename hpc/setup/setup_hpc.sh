#!/bin/bash
# One-shot setup on IITD HPC login node.
#
# Tailored for user mt1230785 (maths/btech), who already ran
# /home/apps/skeleton/oneTimeHPCAccountEnvSetup.sh.
#
# Run from the repo root AFTER uploading col775_a2_hpc/ to $HOME/scratch/:
#     cd $HOME/scratch/col775_a2_hpc
#     bash setup_hpc.sh
#
# Idempotent: re-running skips steps already done.

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRATCH_DATA="/scratch/scai/phd/aiz228170/COL775-A2-2026/dataset/A2_dataset"
CONDA_ENV="col775_a2"
MINICONDA_BASE="/home/apps/miniconda3/24.71"
PROXY_HOST="10.10.78.62"
PROXY_PORT="3128"
# IITD proxy requires HTTP Basic Auth (kerberos-backed). Credentials default to
# the user's HPC creds; override via env vars if needed.
PROXY_USER="${HPC_PROXY_USER:-mt1230785}"
PROXY_PASS="${HPC_PROXY_PASS:?HPC_PROXY_PASS env var must be set to your IITD HPC password}"
PROXY_URL="http://${PROXY_USER}:${PROXY_PASS}@${PROXY_HOST}:${PROXY_PORT}"
PROXY_URL_REDACTED="http://${PROXY_USER}:***@${PROXY_HOST}:${PROXY_PORT}"

echo "[setup] repo root  : $REPO_ROOT"
echo "[setup] miniconda  : $MINICONDA_BASE"
echo "[setup] proxy      : $PROXY_URL_REDACTED"

# ---------------------------------------------------------------------------
# 0. Set proxy for any internet-bound commands run from this shell.
# ---------------------------------------------------------------------------
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export no_proxy="localhost,127.0.0.1,.iitd.ac.in,.iitd.ernet.in"

# ---------------------------------------------------------------------------
# 1. Strip any Windows CRLF line endings (Windows -> Linux artifact).
# ---------------------------------------------------------------------------
echo "[setup] stripping CRLF from shell + pbs scripts"
for f in setup_hpc.sh fetch_captions.sh hpc/*.pbs hpc/*.sh; do
    if [ -f "$f" ]; then
        sed -i 's/\r$//' "$f"
    fi
done

# ---------------------------------------------------------------------------
# 2. Install $HOME/condaBaseSetup so PBS jobs can `conda activate`.
# ---------------------------------------------------------------------------
if [ ! -s "$HOME/condaBaseSetup" ]; then
    echo "[setup] writing $HOME/condaBaseSetup from /home/apps/skeleton/minicondaBaseEnv"
    cp /home/apps/skeleton/minicondaBaseEnv "$HOME/condaBaseSetup"
else
    echo "[setup] $HOME/condaBaseSetup already exists"
fi

# Activate base conda for this session.
source "$HOME/condaBaseSetup"

# ---------------------------------------------------------------------------
# 3. Tell conda + pip to use the proxy. Persisted to ~/.condarc and ~/.pip.
# ---------------------------------------------------------------------------
echo "[setup] configuring conda + pip proxy"
conda config --set proxy_servers.http  "$PROXY_URL"
conda config --set proxy_servers.https "$PROXY_URL"
conda config --set ssl_verify true

mkdir -p "$HOME/.pip"
cat > "$HOME/.pip/pip.conf" <<EOF
[global]
proxy = ${PROXY_URL}
trusted-host = pypi.org
               files.pythonhosted.org
               download.pytorch.org
               repo.anaconda.com
EOF

# ---------------------------------------------------------------------------
# 4. Create conda env (Python 3.10 + GLIBC-2.17 workaround per Piazza FAQ).
# ---------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "[setup] conda env '$CONDA_ENV' already exists, skipping create"
else
    echo "[setup] creating conda env '$CONDA_ENV' from environment.yml (~5-10 min)"
    conda env create -f environment.yml
fi

# ---------------------------------------------------------------------------
# 5. Dataset symlinks. Code expects <repo_root>/A2_dataset/{Part_A,Part_Aa}.
# ---------------------------------------------------------------------------
if [ ! -d "$SCRATCH_DATA" ]; then
    echo "[error] scratch dataset not found at $SCRATCH_DATA"
    echo "        verify with: ls -l $SCRATCH_DATA"
    exit 1
fi

mkdir -p "$REPO_ROOT/A2_dataset"
mkdir -p "$REPO_ROOT/A2_dataset/Part_Aa/Clevr_official/captions"

link() {
    local target="$1"
    local link_path="$2"
    if [ -e "$link_path" ] || [ -L "$link_path" ]; then
        echo "[link] exists: $link_path"
    else
        ln -s "$target" "$link_path"
        echo "[link] $link_path -> $target"
    fi
}

link "$SCRATCH_DATA/Part_A"                                "$REPO_ROOT/A2_dataset/Part_A"
link "$SCRATCH_DATA/Part_Aa/Probe-Datasets"                "$REPO_ROOT/A2_dataset/Part_Aa/Probe-Datasets"
link "$SCRATCH_DATA/Part_Aa/Clevr_official/images"         "$REPO_ROOT/A2_dataset/Part_Aa/Clevr_official/images"

# ---------------------------------------------------------------------------
# 6. Fetch Drive-hosted Part_Aa captions (retrieval.py needs this).
# ---------------------------------------------------------------------------
CAP_JSON="$REPO_ROOT/A2_dataset/Part_Aa/Clevr_official/captions/clevr_val_captions.json"
if [ ! -f "$CAP_JSON" ]; then
    echo "[setup] fetching Part_Aa captions from Drive (login-node only)"
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        conda activate "$CONDA_ENV"
        bash "$REPO_ROOT/fetch_captions.sh"
        conda deactivate
    fi
else
    echo "[setup] captions already at $CAP_JSON"
fi

# ---------------------------------------------------------------------------
# 7. Output directories.
# ---------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/checkpoints" "$REPO_ROOT/embeddings" "$REPO_ROOT/plots"

echo ""
echo "=========================================================="
echo "setup complete."
echo ""
echo "next:"
echo "  1. (optional) sanity-run on interactive A100 (~Rs 1):"
echo "       bash hpc/submit_interactive.sh col775.mt1230785.course 1"
echo "       # then in the session:"
echo "       cd \$HOME/scratch/col775_a2_hpc"
echo "       source \$HOME/condaBaseSetup && conda activate col775_a2"
echo "       nvidia-smi"
echo "       python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
echo ""
echo "  2. submit a 1-epoch test (cheap sanity check):"
echo "       qsub hpc/submit_test.pbs"
echo ""
echo "  3. submit full training:"
echo "       qsub hpc/submit_clip.pbs"
echo "       qsub hpc/submit_dino.pbs"
echo "=========================================================="
