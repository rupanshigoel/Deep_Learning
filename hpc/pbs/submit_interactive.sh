#!/bin/bash
# Convenience wrapper: request an interactive A100 session for debugging.
# Usage:  bash hpc/submit_interactive.sh [hours]
# Default: 1h on icelake A100 in standard queue (~Rs 1).
set -e

HOURS="${1:-1}"
PROJECT="col775.mt1230785.course"

echo "requesting 1x A100 (icelake) for ${HOURS}h on project ${PROJECT}..."
qsub -I \
    -P "${PROJECT}" \
    -q standard \
    -l select=1:ncpus=4:ngpus=1:centos=icelake \
    -l walltime=${HOURS}:00:00
