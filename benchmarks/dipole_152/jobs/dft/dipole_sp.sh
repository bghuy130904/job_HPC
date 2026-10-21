#!/bin/bash
#SBATCH --job-name=dft_dipole_sp
#SBATCH --output=/data/giahuy/Result/DFT/dipole_moments/_output/dipole_sp_%A_%a.out
#SBATCH --error=/data/giahuy/Result/DFT/dipole_moments/_error/dipole_sp_%A_%a.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH --array=0-70%3
set -euo pipefail
start=$(date +%s)
CODE_ROOT="${CODE_ROOT:-/home/giahuy/Code/job}"
INPUT_FILE="$CODE_ROOT/benchmarks/dipole_152/methods/dft/calc_dipmomt_dft.py"
INPUT_JSON="$CODE_ROOT/benchmarks/dipole_152/inputs/sp_inputs.json"
OUTPUT_DIR="${OUTPUT_DIR:-/data/giahuy/Result/DFT/dipole_moments/sp_v2}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/dipole_${SLURM_ARRAY_TASK_ID:-all}.txt"
printf 'START %s job=%s task=%s host=%s\nCode: %s\nOutput: %s\nLoading Python environment...\n' \
    "$(date -Is)" "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-all}" "$(hostname)" \
    "$INPUT_FILE" "$OUTPUT_DIR" >> "$OUTPUT_FILE"
export PYTHONUNBUFFERED=1
export JOB_SCRATCH_PATH="/scratch/${SLURM_JOB_ID}"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR"
module load python3.9
source /home/giahuy/.venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u "$INPUT_FILE" \
    --input "$INPUT_JSON" --outdir "$OUTPUT_DIR" \
    --ref-json "$CODE_ROOT/benchmarks/dipole_152/references/ref_sp.json" \
    --methods pbe0 b3lyp b2plyp \
    --basis "${BASIS:-aug-cc-pCVQZ}" \
    --alpha "${ALPHA_X:-0.53}" "${ALPHA_C:-0.39}" \
    --scf-policy "${SCF_POLICY:-multiguess}" \
    --dipole "${DIPOLE_MODE:-dm}" \
    --fields 0.0001 \
    --scf-cycles 200 --scf-tol 1e-12 --scf-grad-tol 1e-7 \
    --ob-cycles 300 --ob-tol 1e-10 \
    --grid 4 --scf-aux none --mp2-aux auto \
    --threads "$SLURM_CPUS_PER_TASK" --max-memory 40000 \
    --resume "$@" >> "$OUTPUT_FILE" 2>&1
end=$(date +%s)
runtime=$((end-start))
printf 'Hoàn tất. Output: %s. Thời gian: %02d:%02d:%02d\n' \
    "$OUTPUT_FILE" $((runtime/3600)) $(((runtime%3600)/60)) $((runtime%60))
