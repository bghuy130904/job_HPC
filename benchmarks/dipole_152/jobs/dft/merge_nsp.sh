#!/bin/bash
set -euo pipefail
module load python3.9
source /home/giahuy/.venv/bin/activate
CODE_ROOT="${CODE_ROOT:-/home/giahuy/Code/job}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/giahuy/Result/DFT/dipole_moments/nsp_v2}"
python -u "$CODE_ROOT/benchmarks/dipole_152/methods/dft/calc_dipmomt_dft.py" \
    --outdir "$OUTPUT_DIR" --merge \
    --ref-json "$CODE_ROOT/benchmarks/dipole_152/references/ref_nsp.json" \
    "$@" >> "$OUTPUT_DIR/merge.txt" 2>&1
