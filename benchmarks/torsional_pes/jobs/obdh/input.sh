#!/bin/bash
#SBATCH --job-name=torsion_obdh
#SBATCH --output=torsion_obdh_%j.out
#SBATCH --error=torsion_obdh_%j.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem=50G
set -euo pipefail
REPO_DIR="${REPO_DIR:-/home/giahuy/Code/job}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/giahuy/Result/torsional_pes/obdh/${SLURM_JOB_ID:-local}}"
export JOB_SCRATCH_PATH="/scratch/${SLURM_JOB_ID:-torsion_obdh_$$}"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR" "$OUTPUT_DIR"
module load python3.9
source /home/giahuy/.venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
for molecule in ethanol 1-butanol; do
    python -u "$REPO_DIR/benchmarks/torsional_pes/methods/obdh/obdh.py" \
        --molecule "$molecule" --threads "$OMP_NUM_THREADS" --max-memory 40000 \
        --output-dir "$OUTPUT_DIR" "$@"
done
