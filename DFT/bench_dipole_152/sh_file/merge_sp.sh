#!/bin/bash
# Merge chay vai giay, KHONG can compute node, KHONG can pyscf.
# Chay truc tiep tren login node:  bash merge_dft.sh
module load python3.9
source /home/giahuy/.venv/bin/activate

BASE="/data/giahuy/Result/DFT/dipole_moments"
CODE="/home/giahuy/Code/job/DFT/bench_dipole_152/_src/calc_dipmomt_dft.py"
REF="/home/giahuy/Code/job/DFT/bench_dipole_152/geometry/ref_ccsdt.json"

for tag in sp nsp; do
    [ -d "$BASE/$tag/rows" ] || continue
    python "$CODE" --outdir "$BASE/$tag" --merge \
           --output-xlsx "$BASE/$tag/dipole_dft_${tag}.xlsx" \
           --ref-json "$REF"
done