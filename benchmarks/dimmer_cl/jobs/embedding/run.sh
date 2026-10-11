#!/bin/bash
#SBATCH --job-name=dimmer_cl
#SBATCH --output=dimmer_cl-%j.out
#SBATCH --error=dimmer_cl-%j.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem=100G

set -euo pipefail
# sbatch copies this file; use the submission directory or an explicit repo path.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_DIR/benchmarks/dimmer_cl/methods/embedding/intE.py" ]]; then
    echo 'Submit from the job_HPC root or set REPO_DIR to that root.' >&2
    exit 2
fi
export METHOD="${METHOD:-obmp2}"
case "$METHOD" in mp2|obmp2) ;; *) echo 'METHOD must be mp2 or obmp2' >&2; exit 2;; esac
export RUN_DIR="${RUN_DIR:-$REPO_DIR/results/dimmer_cl/$METHOD/${SLURM_JOB_ID:-manual}}"
mkdir -p "$RUN_DIR"
export JOB_SCRATCH_PATH="${JOB_SCRATCH_PATH:-/scratch/${SLURM_JOB_ID:-dimmer_cl_$$}}"
export TMPDIR="$JOB_SCRATCH_PATH"
export PYSCF_SCRATCH="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR"
module load python3.9
source "${VENV_PATH:-/home/giahuy/.venv}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export BASIS="${BASIS:-def2-QZVPPD}"
export MEMORY="${MEMORY:-90000}"
python -u "$REPO_DIR/benchmarks/dimmer_cl/methods/embedding/intE.py" > "$RUN_DIR/scan.log" 2>&1
