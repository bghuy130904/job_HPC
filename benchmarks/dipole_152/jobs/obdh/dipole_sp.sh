#!/bin/bash
#SBATCH --job-name=wftdipolesp
#SBATCH --output=/data/giahuy/Result/OBDH/dipole_moments/_output/dipole_sp_%A_%a.out
#SBATCH --error=/data/giahuy/Result/OBDH/dipole_moments/_error/dipole_sp_%A_%a.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH --array=0-70%3
set -euo pipefail
# Change this pair only. Use: bash dipole_sp.sh --submit
ALPHA=(0.50 0.40)
ALPHA_X="${ALPHA_X:-${ALPHA[0]}}"
ALPHA_C="${ALPHA_C:-${ALPHA[1]}}"
for value in "$ALPHA_X" "$ALPHA_C"; do
    if [[ ! "$value" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
        printf 'Alpha must be a decimal between 0 and 1: %s\n' "$value" >&2
        exit 2
    fi
done
ALPHA_TAG="ax_${ALPHA_X}_ac_${ALPHA_C}"
for argument in "$@"; do
    case "$argument" in
        --alpha|--alpha=*)
            printf 'Set ALPHA=(a_x a_c) or ALPHA_X/ALPHA_C so the job name and folder match.\n' >&2
            exit 2 ;;
    esac
done
RESULT_ROOT="${RESULT_ROOT:-/data/giahuy/Result/OBDH/dipole_moments}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/sp_scaling/$ALPHA_TAG}"
if [[ "${1:-}" == "--submit" ]]; then
    shift
    # Freeze alpha/output in the submitted environment; SBATCH cannot expand Bash variables.
    export ALPHA_X ALPHA_C RESULT_ROOT OUTPUT_DIR
    mkdir -p "$RESULT_ROOT/_output" "$RESULT_ROOT/_error"
    exec sbatch --export=ALL --job-name="wftdipolesp_${ALPHA_TAG}" \
        --output="$RESULT_ROOT/_output/dipole_sp_%A_%a.out" \
        --error="$RESULT_ROOT/_error/dipole_sp_%A_%a.err" "$0" "$@"
fi
if [[ "${1:-}" == "--merge" ]]; then
    shift
    export OUTPUT_DIR
    exec bash "$(dirname -- "${BASH_SOURCE[0]}")/merge_sp.sh" "$@"
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'Use: bash %s --submit (run) or --merge (report)\n' "$0" >&2
    exit 2
fi
start=$(date +%s)
CODE_ROOT="${CODE_ROOT:-/home/giahuy/Code/job}"
INPUT_FILE="$CODE_ROOT/benchmarks/dipole_152/methods/obdh/calc_dipmomt_obdh.py"
INPUT_JSON="$CODE_ROOT/benchmarks/dipole_152/inputs/sp_inputs.json"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/dipole_${SLURM_ARRAY_TASK_ID:-all}.txt"
printf 'START %s job=%s task=%s host=%s\nCode: %s\nOutput: %s\nLoading Python environment...\n' \
    "$(date -Is)" "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-all}" "$(hostname)" \
    "$INPUT_FILE" "$OUTPUT_DIR" >> "$OUTPUT_FILE"
printf 'Alpha: (%s, %s)\n' "$ALPHA_X" "$ALPHA_C" >> "$OUTPUT_FILE"
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
    --methods uhf ump2 obmp2 obdh \
    --basis "${BASIS:-aug-cc-pCVQZ}" \
    --alpha "$ALPHA_X" "$ALPHA_C" \
    --scf-policy "${SCF_POLICY:-multiguess}" \
    --dipole "${DIPOLE_MODE:-both}" \
    --fields 0.0001 \
    --scf-cycles 200 --scf-tol 1e-10 --scf-grad-tol 1e-4 \
    --ob-cycles 300 --ob-tol 1e-8 \
    --grid 4 --scf-aux none --mp2-aux auto \
    --threads "$SLURM_CPUS_PER_TASK" --max-memory 47000 \
    --resume "$@" >> "$OUTPUT_FILE" 2>&1
end=$(date +%s)
runtime=$((end-start))
printf 'Hoàn tất. Output: %s. Thời gian: %02d:%02d:%02d\n' \
    "$OUTPUT_FILE" $((runtime/3600)) $(((runtime%3600)/60)) $((runtime%60))
