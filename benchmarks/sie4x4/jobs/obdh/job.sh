#!/bin/bash
#SBATCH --job-name=obdh_sie4x4
#SBATCH --output=/data/giahuy/Result/OBDH/SIE_DFT/_output/sie4x4_%A_%a.out
#SBATCH --error=/data/giahuy/Result/OBDH/SIE_DFT/_error/sie4x4_%A_%a.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem=60000
#SBATCH --array=0-80%4
set -euo pipefail

# Submit once: bash job.sh --submit
# Task 0=(0.1,0.1), 8=(0.1,0.9), 9=(0.2,0.1), 80=(0.9,0.9).
# ARRAY_SPEC=0-8%2 restricts the scan; default: all 81 pairs, 3 at a time.
CODE_ROOT="${CODE_ROOT:-/home/giahuy/Code/job}"
RESULT_ROOT="${RESULT_ROOT:-/data/giahuy/Result/OBDH/SIE_DFT}"
SCAN_ROOT="${SCAN_ROOT:-$RESULT_ROOT/scaling_v2}"
TMP_ROOT="${TMP_ROOT:-/data/giahuy/pyscf_tmp/sie4x4}"
# Each worker owns its HDF5 scratch files; NFS file locking hangs on c3.
# Export before Python imports so primary and neighbor workers inherit it.
export HDF5_USE_FILE_LOCKING=FALSE
ARRAY_SPEC="${ARRAY_SPEC:-0-80%3}"
NEIGHBOR_STEP="${NEIGHBOR_STEP:-0.1}"
INPUT_FILE="$CODE_ROOT/benchmarks/sie4x4/methods/obdh/calc_OBDH_scaling.py"
INPUT_JSON="${INPUT_JSON:-$CODE_ROOT/benchmarks/sie4x4/inputs/input.json}"
JOB_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for argument in "$@"; do
    case "$argument" in
        --alpha|--alpha=*|--outdir|--outdir=*)
            printf 'Alpha/output are assigned by the array task. Use ARRAY_SPEC or SCAN_ROOT.\n' >&2
            exit 2 ;;
    esac
done
if [[ "${1:-}" == "--submit" ]]; then
    shift
    export CODE_ROOT RESULT_ROOT SCAN_ROOT TMP_ROOT NEIGHBOR_STEP INPUT_JSON
    mkdir -p "$RESULT_ROOT/_output" "$RESULT_ROOT/_error" "$SCAN_ROOT"
    array_id=$(sbatch --parsable --export=ALL --array="$ARRAY_SPEC" \
        --output="$RESULT_ROOT/_output/sie4x4_%A_%a.out" \
        --error="$RESULT_ROOT/_error/sie4x4_%A_%a.err" "$JOB_DIRECTORY/job.sh" "$@")
    array_id="${array_id%%;*}"
    printf 'Scaling array: %s (%s). Primary and diagnostic results are separate.\n' "$array_id" "$ARRAY_SPEC"
    summary_id=$(sbatch --parsable --export=ALL --dependency="afterany:$array_id" \
        --output="$RESULT_ROOT/_output/sie4x4_summary_%j.out" \
        --error="$RESULT_ROOT/_error/sie4x4_summary_%j.err" "$JOB_DIRECTORY/merge.sh")
    printf 'Summary job: %s (after all tasks, including failures).\nResults: %s\n' "$summary_id" "$SCAN_ROOT"
    exit 0
fi
if [[ "${1:-}" == "--merge" ]]; then
    shift
    export CODE_ROOT RESULT_ROOT SCAN_ROOT TMP_ROOT
    exec bash "$JOB_DIRECTORY/merge.sh" "$@"
fi
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    printf 'Use: bash %s --submit [Python options]. No manual Alpha is needed.\n' "$0" >&2
    exit 2
fi
if [[ ! "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || (( SLURM_ARRAY_TASK_ID > 80 )); then
    printf 'Array task must be an integer from 0 to 80.\n' >&2
    exit 2
fi
ALPHAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
ALPHA_X="${ALPHAS[$((SLURM_ARRAY_TASK_ID / 9))]}"
ALPHA_C="${ALPHAS[$((SLURM_ARRAY_TASK_ID % 9))]}"
OUTPUT_DIR="$SCAN_ROOT/recalc_obdh_(${ALPHA_X},${ALPHA_C})"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/sie4x4_ax_${ALPHA_X}_ac_${ALPHA_C}.txt"
printf 'START %s job=%s array=%s task=%s host=%s\nCode: %s\nInput: %s\nOutput: %s\nAlpha: (%s, %s)\n' \
    "$(date -Is)" "$SLURM_JOB_ID" "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" "$SLURM_ARRAY_TASK_ID" "$(hostname)" \
    "$INPUT_FILE" "$INPUT_JSON" "$OUTPUT_DIR" "$ALPHA_X" "$ALPHA_C" >> "$OUTPUT_FILE"

# Same /data convention as dipole_cccbdb/jobs/ccsd/job.sh; no node-local scratch.
TASK_TMP_ROOT="$TMP_ROOT/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/task_$SLURM_ARRAY_TASK_ID"
mkdir -p "$TASK_TMP_ROOT"
JOB_TMP_DIR=$(mktemp -d "$TASK_TMP_ROOT/run.XXXXXXXX")
export TMPDIR="$JOB_TMP_DIR"
export PYSCF_TMPDIR="$JOB_TMP_DIR"
cleanup() {
    status=$?
    trap - EXIT
    printf 'END %s exit=%s temporary_directory=%s\n' "$(date -Is)" "$status" "$JOB_TMP_DIR" >> "$OUTPUT_FILE"
    if [[ "${KEEP_TMP:-0}" == 1 ]]; then
        printf 'KEEP_TMP=1: retained this task temporary directory.\n' >> "$OUTPUT_FILE"
    elif [[ "$JOB_TMP_DIR" == "$TASK_TMP_ROOT"/run.* ]]; then
        rm -rf -- "$JOB_TMP_DIR"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'TMPDIR=%s\nPYSCF_TMPDIR=%s\nHDF5_USE_FILE_LOCKING=%s\nTemporary filesystem capacity:\n' "$TMPDIR" "$PYSCF_TMPDIR" "$HDF5_USE_FILE_LOCKING" >> "$OUTPUT_FILE"
df -h "$TMPDIR" >> "$OUTPUT_FILE"
df -i "$TMPDIR" >> "$OUTPUT_FILE"
export PYTHONUNBUFFERED=1
module load python3.9
source /home/giahuy/.venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u "$INPUT_FILE" \
    --input "$INPUT_JSON" --outdir "$OUTPUT_DIR" \
    --threads "$OMP_NUM_THREADS" --max-memory "${MAX_MEMORY:-55000}" \
    --alpha "$ALPHA_X" "$ALPHA_C" \
    --methods obdh obdh_1st --ob-init uhf dft \
    --confirm-c1-c4 --ob-restarts 2 --ob-restart-angle 0.05 \
    --ob-probe-directions 2 \
    --neighbor-mode auto --neighbor-root "$SCAN_ROOT" \
    --neighbor-step "$NEIGHBOR_STEP" --prepare-neighbor \
    --basis "${BASIS:-aug-cc-pVDZ}" --grid 4 \
    --scf-policy multiguess --scf-cycles 200 --scf-tol 1e-9 --scf-grad-tol 1e-6 \
    --ob-cycles 300 --ob-tol 1e-6 \
    "$@" >> "$OUTPUT_FILE" 2>&1
