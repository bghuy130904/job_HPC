#!/bin/bash
#SBATCH --job-name=obdh_sie4x4
#SBATCH --output=/data/giahuy/Result/OBDH/SIE_DFT/_output/sie4x4_%j.out
#SBATCH --error=/data/giahuy/Result/OBDH/SIE_DFT/_error/sie4x4_%j.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=6000
set -euo pipefail

# Change this pair only, then: bash job.sh --submit
# C1-C4 confirmation is enabled below, including automatic neighbor preparation.
ALPHA=(0.5 0.2)
NEIGHBOR_STEP="${NEIGHBOR_STEP:-0.1}"
ALPHA_X="${ALPHA_X:-${ALPHA[0]}}"
ALPHA_C="${ALPHA_C:-${ALPHA[1]}}"
for value in "$ALPHA_X" "$ALPHA_C"; do
    if [[ ! "$value" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
        printf 'Alpha must be a decimal between 0 and 1: %s\n' "$value" >&2
        exit 2
    fi
done
ALPHA_X=$(awk -v value="$ALPHA_X" 'BEGIN {printf "%.10g", value}')
ALPHA_C=$(awk -v value="$ALPHA_C" 'BEGIN {printf "%.10g", value}')
ALPHA_TAG="ax_${ALPHA_X}_ac_${ALPHA_C}"
for argument in "$@"; do
    case "$argument" in
        --alpha|--alpha=*|--outdir|--outdir=*)
            printf 'Set ALPHA=(a_x a_c), ALPHA_X/ALPHA_C or OUTPUT_DIR so names and input parameters agree.\n' >&2
            exit 2 ;;
    esac
done
RESULT_ROOT="${RESULT_ROOT:-/data/giahuy/Result/OBDH/SIE_DFT}"
BASE_OUTPUT_DIR="$RESULT_ROOT/scaling_v2/recalc_obdh_(${ALPHA_X},${ALPHA_C})"
if [[ -z "${OUTPUT_DIR+x}" ]]; then
    OUTPUT_DIR="$BASE_OUTPUT_DIR"
    # Keep earlier v2 data that did not run all checks; do not mix/relabel it.
    if [[ -f "$BASE_OUTPUT_DIR/config.json" ]] && \
       ! grep -Eq '"confirm_c1_c4"[[:space:]]*:[[:space:]]*true' "$BASE_OUTPUT_DIR/config.json"; then
        OUTPUT_DIR="${BASE_OUTPUT_DIR}_C1_C4"
    fi
fi
if [[ "${1:-}" == "--submit" ]]; then
    shift
    export ALPHA_X ALPHA_C RESULT_ROOT OUTPUT_DIR
    mkdir -p "$RESULT_ROOT/_output" "$RESULT_ROOT/_error"
    exec sbatch --export=ALL --job-name="obdh_sie_${ALPHA_TAG}" \
        --output="$RESULT_ROOT/_output/sie4x4_${ALPHA_TAG}_%j.out" \
        --error="$RESULT_ROOT/_error/sie4x4_${ALPHA_TAG}_%j.err" "$0" "$@"
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'Use: bash %s --submit [Python options]\n' "$0" >&2
    exit 2
fi
start=$(date +%s)
CODE_ROOT="${CODE_ROOT:-/home/giahuy/Code/job}"
INPUT_FILE="$CODE_ROOT/benchmarks/sie4x4/methods/obdh/calc_OBDH_scaling.py"
INPUT_JSON="${INPUT_JSON:-$CODE_ROOT/benchmarks/sie4x4/inputs/input.json}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/sie4x4_${ALPHA_TAG}.txt"
printf 'START %s job=%s host=%s\nCode: %s\nInput: %s\nOutput: %s\nAlpha: (%s, %s)\nLoading Python environment...\n' \
    "$(date -Is)" "$SLURM_JOB_ID" "$(hostname)" "$INPUT_FILE" "$INPUT_JSON" \
    "$OUTPUT_DIR" "$ALPHA_X" "$ALPHA_C" >> "$OUTPUT_FILE"
export PYTHONUNBUFFERED=1
export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR"
module load python3.9
source /home/giahuy/.venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u "$INPUT_FILE" \
    --input "$INPUT_JSON" --outdir "$OUTPUT_DIR" \
    --threads "$OMP_NUM_THREADS" --max-memory "${MAX_MEMORY:-55000}" \
    --alpha "$ALPHA_X" "$ALPHA_C" \
    --methods obdh obdh_1st --ob-init uhf dft \
    --confirm-c1-c4 --ob-restarts 2 --ob-restart-angle 0.05 \
    --ob-probe-directions 2 \
    --neighbor-mode auto --neighbor-root "$RESULT_ROOT/scaling_v2" \
    --neighbor-step "$NEIGHBOR_STEP" --prepare-neighbor \
    --basis "${BASIS:-aug-cc-pVDZ}" --grid 4 \
    --scf-policy multiguess --scf-cycles 200 --scf-tol 1e-9 --scf-grad-tol 1e-6 \
    --ob-cycles 300 --ob-tol 1e-6 \
    "$@" >> "$OUTPUT_FILE" 2>&1
end=$(date +%s)
runtime=$((end-start))
printf 'Hoàn tất. Output: %s. Thời gian: %02d:%02d:%02d\n' \
    "$OUTPUT_FILE" $((runtime/3600)) $(((runtime%3600)/60)) $((runtime%60))



