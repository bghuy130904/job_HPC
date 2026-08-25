#!/bin/bash
#SBATCH --job-name=ccsd_dp_dm
#SBATCH --output=/data/giahuy/Result/CCSD/dipole_moments/_output/dipmomt_dm_%A_%a_%N.out
#SBATCH --error=/data/giahuy/Result/CCSD/dipole_moments/_error/dipmomt_dm_%A_%a_%N.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --threads-per-core=1
#SBATCH --mem=30G                     ### DM chi 1 lan CCSD/chat thay vi 7 -> 35G la thua
#SBATCH --time=28:00:00               ### FF mat 15.2 h o chat nang nhat (PPO) -> DM ~2 h
#SBATCH --array=0-70%6                ### 160 GB duoc chia tren c3 / 24G = 6 task dong thoi

CODE="/home/giahuy/Code/job/CCSD/bench_dipole_152/_src/calc_dipmomt_ccsd_ff.py"
INPUT="/home/giahuy/Code/job/CCSD/bench_dipole_152/geometry/sp_inputs.json"
OUTDIR="/data/giahuy/Result/CCSD/dipole_moments/sp_dm"

mkdir -p /data/giahuy/Result/CCSD/dipole_moments/_output
mkdir -p /data/giahuy/Result/CCSD/dipole_moments/_error
mkdir -p "$OUTDIR/rows"

# scratch bat buoc tren cluster nay
export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR"

module load python3.9
source /home/giahuy/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYSCF_MAX_MEMORY=$(( SLURM_MEM_PER_NODE * 75 / 100 ))
echo "SLURM cap ${SLURM_MEM_PER_NODE} MB -> PYSCF_MAX_MEMORY=${PYSCF_MAX_MEMORY} MB"

start=$(date +%s)

# --dipole dm: bo han nhanh finite field, chi chay 1 lan CCSD + solve_lambda.
# Cac cot mu_ff / ff_curv / diff_pct se de trong; ket qua nam o cot mu_dm.
python "$CODE" --input "$INPUT" --outdir "$OUTDIR" --dipole dm

end=$(date +%s); rt=$((end - start))
printf "Task %s xong. Thoi gian: %02d:%02d:%02d\n" \
       "$SLURM_ARRAY_TASK_ID" $((rt/3600)) $(((rt%3600)/60)) $((rt%60))