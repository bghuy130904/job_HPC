#!/bin/bash
#SBATCH --job-name=diss
#SBATCH --output=/data/giahuy/Result/OBDH/bench_dip_noneq_config/_output/diss_%A_%a_%N.out
#SBATCH --error=/data/giahuy/Result/OBDH/bench_dip_noneq_config/_error/diss_%A_%a_%N.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --threads-per-core=1
#SBATCH --mem=50G
#SBATCH --time=40:00:00
#SBATCH --array=0-1          ### 0 = FH, 1 = FCl (moi he chay LIEN TUC, khong chia nho)

CODE="/home/giahuy/Code/job/OBDH/bench_dip_noneq_config/_src/calc_dissociation.py"
INPUT="/home/giahuy/Code/job/OBDH/bench_dip_noneq_config/geometry/dissociation_inputs.json"
REF="/home/giahuy/Code/job/OBDH/bench_dip_noneq_config/geometry/dissociation_reference.json"
OUTDIR="/data/giahuy/Result/OBDH/bench_dip_noneq_config"

SYS=(FH FCl)
S=${SYS[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$OUTDIR/_output" "$OUTDIR/_error" "$OUTDIR/rows"

export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR"

module load python3.9
source /home/giahuy/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYSCF_MAX_MEMORY=$(( SLURM_MEM_PER_NODE * 75 / 100 ))
echo "he=$S | SLURM cap ${SLURM_MEM_PER_NODE} MB -> PYSCF_MAX_MEMORY=${PYSCF_MAX_MEMORY} MB"

start=$(date +%s)

# solver cua pyCMF in bang print() thuan, khong bit duoc bang verbose=0.
# Chuyen stdout sang file rieng de .out chi giu dong tom tat cua script.
python "$CODE" --input "$INPUT" --outdir "$OUTDIR" --system "$S" \
       --ref-json "$REF" --methods uhf ump2 obmp2 obdh \
    | grep -E "^\[|^  r=|MERGE|===|branch_switch|hysteresis|method|benchmark|UHF|UMP2|OBMP2|OBDH"

end=$(date +%s); rt=$((end - start))
printf "He %s xong. Thoi gian: %02d:%02d:%02d\n" "$S" $((rt/3600)) $(((rt%3600)/60)) $((rt%60))