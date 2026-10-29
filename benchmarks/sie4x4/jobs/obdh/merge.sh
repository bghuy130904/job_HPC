#!/bin/bash
#SBATCH --job-name=sie4x4_summary
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail
CODE_ROOT="${CODE_ROOT:-/home/giahuy/Code/job}"
RESULT_ROOT="${RESULT_ROOT:-/data/giahuy/Result/OBDH/SIE_DFT}"
SCAN_ROOT="${SCAN_ROOT:-$RESULT_ROOT/scaling_v2}"
TMP_ROOT="${TMP_ROOT:-/data/giahuy/pyscf_tmp/sie4x4}"
mkdir -p "$TMP_ROOT" "$SCAN_ROOT/scan_summary"
SUMMARY_TMP_DIR=$(mktemp -d "$TMP_ROOT/summary.XXXXXXXX")
export TMPDIR="$SUMMARY_TMP_DIR" PYSCF_TMPDIR="$SUMMARY_TMP_DIR"
trap 'rm -rf -- "$SUMMARY_TMP_DIR"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
module load python3.9
source /home/giahuy/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u "$CODE_ROOT/benchmarks/sie4x4/tools/summarize_sie_v2.py" \
    --root "$SCAN_ROOT" "$@" >> "$SCAN_ROOT/scan_summary/summary.txt" 2>&1
