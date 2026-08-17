#!/bin/bash

module load python3.9
source /home/giahuy/.venv/bin/activate

BASE="/data/giahuy/Result/OBDH/dipole_moments"
CODE="/home/giahuy/Code/job/OBDH/bench_dipole_152/_src/calc_dipmomt_obdh.py"

for tag in sp nsp; do
    [ -d "$BASE/$tag/rows" ] || continue
    python "$CODE" --outdir "$BASE/$tag" --merge \
           --output-xlsx "$BASE/$tag/dipole_obdh_${tag}.xlsx" \
           --ref-json "/home/giahuy/Code/job/CCSD/geometry/ref_ccsdt.json"
done