#!/bin/bash
#SBATCH --job-name=SIE_obdh
#SBATCH --output=SIE_%x_%j.out
#SBATCH --error=SIE_%x_%j.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
set -euo pipefail
# Submit from repository root; SLURM copies this script, so do not infer root from $0.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIE_ROOT="$REPO_ROOT/benchmarks/sie4x4"
OUTDIR="${OUTDIR:-$REPO_ROOT/results/sie4x4/obdh/${SLURM_JOB_ID:-local}}"
export JOB_SCRATCH_PATH="${JOB_SCRATCH_PATH:-/scratch/${USER}/sie-${SLURM_JOB_ID:-$$}}"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR" "$OUTDIR"
# Activate your environment before sbatch, or provide VENV explicitly.
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
python "$SIE_ROOT/methods/obdh/calc_OBDH.py"   --outdir "$OUTDIR" --basis "${BASIS:-aug-cc-pVDZ}" --grid "${GRID:-4}"   --threads "$OMP_NUM_THREADS" "$@" > "$OUTDIR/driver.log" 2>&1
