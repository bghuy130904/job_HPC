#!/bin/bash
#SBATCH --job-name=dipmomt_sp
#SBATCH --output=/data/giahuy/Result/CCSD/dipole_moments/_output/dipmomt_%A_%a.out
#SBATCH --error=/data/giahuy/Result/CCSD/dipole_moments/_error/dipmomt_%A_%a.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem=40G 
#SBATCH --time=24:00:00               ### CHỈNH theo giới hạn partition
#SBATCH --array=0-70%20               ### sp: 71 phân tử -> 0-70 ; nsp (81) -> đổi 0-80


CODE="/home/giahuy/Code/job/CCSD/_src/calc_dipmomt_ccsd.py"
INPUT="/home/giahuy/Code/job/CCSD/geometry/sp_inputs.json"
OUTDIR="/data/giahuy/Result/CCSD/dipole_moments/sp"

# thư mục cho log của SBATCH (SLURM KHÔNG tự tạo) + thư mục kết quả
mkdir -p /data/giahuy/Result/CCSD/dipole_moments/_output
mkdir -p /data/giahuy/Result/CCSD/dipole_moments/_error
mkdir -p "$OUTDIR/rows"

# scratch bắt buộc trên cluster này
export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR"

module load python3.9
source /home/giahuy/.venv/bin/activate

# threads khớp với cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# PySCF RAM ~ (mem-per-cpu * cpus) trừ headroom
export PYSCF_MAX_MEMORY=40000

start=$(date +%s)

# mỗi task chạy đúng 1 phân tử theo SLURM_ARRAY_TASK_ID
python "$CODE" --input "$INPUT" --outdir "$OUTDIR"

end=$(date +%s); rt=$((end - start))
printf "Task %s xong. Thời gian: %02d:%02d:%02d\n" \
       "$SLURM_ARRAY_TASK_ID" $((rt/3600)) $(((rt%3600)/60)) $((rt%60))