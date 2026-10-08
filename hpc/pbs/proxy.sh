#!/bin/bash
# Kerberos-authenticated CNTLM proxy for compute-node internet.
# Only needed if a batch job must reach the internet (HuggingFace, pip, wandb).
# The training + probe pipeline does NOT need this — all data is pre-staged
# and pip installs happen on the login node via setup_hpc.sh.
#
# Usage (inside a PBS script, foreground-backgrounded):
#     ./hpc/proxy.sh &
#     export http_proxy="http://proxy62.iitd.ac.in:3128"
#     export https_proxy="http://proxy62.iitd.ac.in:3128"
#
# Program-code → proxy number map (from HPC docs):
#     mtech  -> 62   (proxy62.iitd.ac.in)
#     btech  -> 63
#     phd    -> 61
#
# This expects your Kerberos password to be readable from $HOME/.k5login or
# stored via `kinit` before the job is submitted. Consult HPC FAQ for details.

KRB_USER="${USER}@IITD.AC.IN"

# If cached credentials from `kinit` on the login node are still valid they
# propagate to the compute node automatically. Re-run kinit from the login
# node before `qsub` if you see auth failures.
if command -v klist >/dev/null 2>&1; then
    klist -s || echo "[proxy] no valid kerberos ticket; run 'kinit' on login node before qsub"
fi

# Typical pattern: launch the proxy binary provided by HPC admins.
# The actual binary path varies — check with `which cntlm` on a login node.
if command -v cntlm >/dev/null 2>&1; then
    cntlm -c "$HOME/.cntlm.conf" -f &
else
    echo "[proxy] cntlm not found on PATH; ask HPC admin for setup instructions"
fi
