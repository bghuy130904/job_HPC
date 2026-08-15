#!/bin/bash
#SBATCH --job-name=dipmomt_merge
#SBATCH --output=/data/giahuy/Result/CCSD/dipole_moments/_output/merge_%j.out
#SBATCH --error=/data/giahuy/Result/CCSD/dipole_moments/_error/merge_%j.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4000
#SBATCH --time=00:15:00

module load python3.9
source /home/giahuy/.venv/bin/activate

python /home/giahuy/Code/job/CCSD/_src/calc_dipmomt_ccsd.py \
       --input /home/giahuy/Code/job/CCSD/geometry/sp_inputs.json  \
       --outdir /data/giahuy/Result/CCSD/dipole_moments/sp --merge   \
       --output-xlsx /data/giahuy/Result/CCSD/dipole_moments/sp/dipole152_sp.xlsx
