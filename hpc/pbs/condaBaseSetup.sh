#!/bin/bash
# Template for $HOME/condaBaseSetup — sourced by PBS batch scripts because
# `conda activate` does not work in non-interactive shells without this.
#
# HOW TO POPULATE (on the HPC login node, ONCE):
#     1. Install Anaconda: bash Anaconda3-*.sh  (accept default ~/anaconda3 + yes to conda init)
#     2. exec bash  (or re-login) — this writes a `>>> conda initialize >>>` block into ~/.bashrc
#     3. Copy that block into $HOME/condaBaseSetup:
#        sed -n '/>>> conda initialize >>>/,/<<< conda initialize <<</p' ~/.bashrc > $HOME/condaBaseSetup
#     4. chmod 644 $HOME/condaBaseSetup
#
# After that, any batch script can just do:
#     source $HOME/condaBaseSetup
#     conda activate col775_a2
#
# This file is only a *placeholder/documentation* in the repo; the real thing
# lives in your $HOME and is written by setup_hpc.sh.
