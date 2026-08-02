#!/bin/bash
#SBATCH --job-name=obdh_scan
#SBATCH --output=/data/giahuy/Result/OBDH/dipole_moments/_output/obdh_%A_%a.out
#SBATCH --error=/data/giahuy/Result/OBDH/dipole_moments/_error/obdh_%A_%a.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1          ### node bật hyperthreading -> lấy core VẬT LÝ
#SBATCH --mem=32G                      ### tổng/‌task (KHÔNG dùng mem-per-cpu)
#SBATCH --time=24:00:00                ### 99 alpha/phân tử -> để rộng; chỉnh theo partition
#SBATCH --array=0-70%8                 ### sp: 71 chất -> 0-70 ; nsp (81) -> 0-80

CODE="/home/giahuy/Code/job/OBDH/bench_dipole_152/_src/obdh.py"
INPUT="/home/giahuy/Code/job/OBDH/bench_dipole_152/geometry/sp_inputs.json"
REF="/home/giahuy/Code/job/OBDH/bench_dipole_152/geometry/ref_sp.json"     # {ten: dipole_CCSD(T)}
OUTDIR="/data/giahuy/Result/OBDH/dipole_moments/sp_scan"

mkdir -p /data/giahuy/Result/OBDH/dipole_moments/_output
mkdir -p /data/giahuy/Result/OBDH/dipole_moments/_error
mkdir -p "$OUTDIR/rows"

export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"; mkdir -p "$TMPDIR"

module load python3.9
source /home/giahuy/.venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYSCF_MAX_MEMORY=28000          # chừa headroom dưới 32G

start=$(date +%s)
python "$CODE" --input "$INPUT" --outdir "$OUTDIR" --ref "$REF" \
       --alpha-min 0.01 --alpha-max 0.99 --alpha-step 0.05
end=$(date +%s); rt=$((end-start))
printf "Task %s: %02d:%02d:%02d\n" "$SLURM_ARRAY_TASK_ID" $((rt/3600)) $(((rt%3600)/60)) $((rt%60))