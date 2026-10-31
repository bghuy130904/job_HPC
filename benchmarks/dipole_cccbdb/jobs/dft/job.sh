#!/bin/bash
#SBATCH --job-name=dft_dipole_cccbdb
#SBATCH --output=/data/giahuy/Result/DFT/dipole_cccbdb/_output/dipole_%A_%a.out
#SBATCH --error=/data/giahuy/Result/DFT/dipole_cccbdb/_error/dipole_%A_%a.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH --array=0-19%3
set -euo pipefail

# Same launch convention as dipole_152/jobs/obdh/dipole_sp.sh.
# Change this pair only for an OB scaling run, then: bash job.sh --submit
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
        --alpha|--alpha=*|--outdir|--outdir=*)
            printf 'Set ALPHA=(a_x a_c), ALPHA_X/ALPHA_C or OUTPUT_DIR so names and parameters agree.\n' >&2
            exit 2 ;;
    esac
done
CODE_ROOT="${CODE_ROOT:-/home/giahuy/Code/job}"
INPUT_FILE="$CODE_ROOT/benchmarks/dipole_cccbdb/methods/dft/calc_dipmomt_dft.py"
INPUT_JSON="$CODE_ROOT/benchmarks/dipole_cccbdb/inputs/cccbdb_inputs.json"
REF_JSON="$CODE_ROOT/benchmarks/dipole_cccbdb/references/ref_cccbdb.json"
RESULT_ROOT="${RESULT_ROOT:-/data/giahuy/Result/DFT/dipole_cccbdb}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/v2}"

# Expanded 20-species input has a different fingerprint from the original ten.
# Use ARRAY_SPEC=10-19%3 with --submit to calculate only the additions.
ARRAY_SPEC="${ARRAY_SPEC:-0-19%3}"

if [[ "${1:-}" == "--submit" ]]; then
    shift
    # Freeze the pair and destination for this submitted array.
    export ALPHA_X ALPHA_C RESULT_ROOT OUTPUT_DIR CODE_ROOT
    mkdir -p "$RESULT_ROOT/_output" "$RESULT_ROOT/_error"
    exec sbatch --export=ALL --array="$ARRAY_SPEC" --job-name="dft_cccbdb_${ALPHA_TAG}" \
        --output="$RESULT_ROOT/_output/dipole_%A_%a.out" \
        --error="$RESULT_ROOT/_error/dipole_%A_%a.err" "$0" "$@"
fi
if [[ "${1:-}" == "--merge" ]]; then
    shift
    if [[ ! -f "$OUTPUT_DIR/dipole.meta.json" ]]; then
        printf 'Missing metadata: %s/dipole.meta.json\n' "$OUTPUT_DIR" >&2
        exit 2
    fi
    module load python3.9
    source /home/giahuy/.venv/bin/activate
    exec python -u "$INPUT_FILE" --outdir "$OUTPUT_DIR" --merge \
        --ref-json "$REF_JSON" "$@" >> "$OUTPUT_DIR/merge.txt" 2>&1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'Use: bash %s --submit (run) or --merge (report)\n' "$0" >&2
    exit 2
fi

start=$(date +%s)
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/dipole_${SLURM_ARRAY_TASK_ID:-all}.txt"
printf 'START %s job=%s task=%s host=%s\nCode: %s\nInput: %s\nReference: %s\nOutput: %s\nAlpha: (%s, %s)\nLoading Python environment...\n' \
    "$(date -Is)" "$SLURM_JOB_ID" "${SLURM_ARRAY_TASK_ID:-all}" "$(hostname)" \
    "$INPUT_FILE" "$INPUT_JSON" "$REF_JSON" "$OUTPUT_DIR" "$ALPHA_X" "$ALPHA_C" >> "$OUTPUT_FILE"
export PYTHONUNBUFFERED=1
export JOB_SCRATCH_PATH="/scratch/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID:-all}"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR"
module load python3.9
source /home/giahuy/.venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u "$INPUT_FILE" \
    --input "$INPUT_JSON" --outdir "$OUTPUT_DIR" --ref-json "$REF_JSON" \
    --methods pbe0 b3lyp b2plyp \
    --basis "${BASIS:-aug-cc-pCVQZ}" \
    --alpha "$ALPHA_X" "$ALPHA_C" \
    --scf-policy "${SCF_POLICY:-standard}" \
    --dipole "${DIPOLE_MODE:-dm}" --fields 0.0001 \
    --scf-cycles 200 --scf-tol 1e-10 --scf-grad-tol 1e-4 \
    --ob-cycles 300 --ob-tol 1e-8 \
    --grid 4 --scf-aux none --mp2-aux auto \
    --threads "$SLURM_CPUS_PER_TASK" --max-memory "${MAX_MEMORY:-47000}" \
    --resume "$@" >> "$OUTPUT_FILE" 2>&1

end=$(date +%s)
runtime=$((end-start))
printf 'Hoàn tất. Output: %s. Thời gian: %02d:%02d:%02d\n' \
    "$OUTPUT_FILE" $((runtime/3600)) $(((runtime%3600)/60)) $((runtime%60))
