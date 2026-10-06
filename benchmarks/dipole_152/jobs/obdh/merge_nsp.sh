#!/bin/bash
#SBATCH --job-name=obdh_merge
#SBATCH --output=/data/giahuy/Result/OBDH/dipole_moments/_output/merge_%j.out
#SBATCH --error=/data/giahuy/Result/OBDH/dipole_moments/_error/merge_%j.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

CODE="/home/giahuy/Code/job/OBDH/bench_dipole_152/_src/obdh.py"
INPUT="/home/giahuy/Code/job/OBDH/bench_dipole_152/geometry/nsp_inputs.json"
REF="/home/giahuy/Code/job/OBDH/bench_dipole_152/geometry/ref_nsp.json"
OUTDIR="/data/giahuy/Result/OBDH/dipole_moments/nsp_scan"

module load python3.9
source /home/giahuy/.venv/bin/activate

python "$CODE" --input "$INPUT" --outdir "$OUTDIR" --ref "$REF" --merge --trim 3