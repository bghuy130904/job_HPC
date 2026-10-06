#!/bin/bash
# Compatibility wrapper; submit job_SIE4x4_DH.sh directly for SLURM directives.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$REPO_ROOT/benchmarks/sie4x4/jobs/orca/job_SIE4x4_DH.sh" "$@"
