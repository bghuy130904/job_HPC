#!/bin/bash
#SBATCH --job-name=obdh_sp
#SBATCH --output=/data/giahuy/Result/OBDH/dipole_moments/_output/obdh_%A_%a.out
#SBATCH --error=/data/giahuy/Result/OBDH/dipole_moments/_error/obdh_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --array=0-70%20          ### sp: 71 chat -> 0-70 ; nsp (81) -> 0-80
                                ### %4 vi tren c3 ban chi duoc chia ~160 GB:
                                ### 160/36 = 4 task dong thoi la toi da

CODE="/home/giahuy/Code/job/OBDH/bench_dipole_152/_src/calc_dipmomt_obdh.py"
INPUT="/home/giahuy/Code/job/OBDH/bench_dipole_152/geometry/sp_inputs.json"
OUTDIR="/data/giahuy/Result/OBDH/dipole_moments/sp_dm"

mkdir -p /data/giahuy/Result/OBDH/dipole_moments/_output
mkdir -p /data/giahuy/Result/OBDH/dipole_moments/_error
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
# de thap hon --mem (36G = 36864 MB) khoang 18%: PySCF khong tinh mang tam
# cua numpy va buffer tich phan vao max_memory
export PYSCF_MAX_MEMORY=$(( SLURM_MEM_PER_NODE * 75 / 100 ))
echo "SLURM cap ${SLURM_MEM_PER_NODE} MB -> PYSCF_MAX_MEMORY=${PYSCF_MAX_MEMORY} MB"

start=$(date +%s)

python "$CODE" --input "$INPUT" --outdir "$OUTDIR" \
       --methods uhf ump2 obmp2 obdh --dipole both \
    | grep -E "^\[|MERGE|status|grad_norm|E_spread|dE_HF"

end=$(date +%s); rt=$((end - start))
printf "Task %s xong. Thoi gian: %02d:%02d:%02d\n" \
       "$SLURM_ARRAY_TASK_ID" $((rt/3600)) $(((rt%3600)/60)) $((rt%60))