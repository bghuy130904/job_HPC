#!/bin/bash
set -euo pipefail
module load python3.9
source /home/giahuy/.venv/bin/activate
CODE_ROOT="${CODE_ROOT:-/home/giahuy/Code/job}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/giahuy/Result/CCSD/dipole_moments/sp_v2}"
python -u "$CODE_ROOT/benchmarks/dipole_152/methods/ccsd/calc_dipmomt_ccsd_ff.py" \
    --outdir "$OUTPUT_DIR" --merge \
    --ref-json "$CODE_ROOT/benchmarks/dipole_152/references/ref_sp.json" \
    "$@" >> "$OUTPUT_DIR/merge.txt" 2>&1
