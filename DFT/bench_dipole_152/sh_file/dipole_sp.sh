#!/bin/bash
#SBATCH --job-name=dft_sp
#SBATCH --output=/data/giahuy/Result/DFT/dipole_moments/_output/dft_%A_%a.out
#SBATCH --error=/data/giahuy/Result/DFT/dipole_moments/_error/dft_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem=24G
#SBATCH --time=40:00:00
#SBATCH --array=0-70%20          ### sp: 71 chat -> 0-70 ; nsp (81) -> 0-80
                                ### %4: tren c3 ban duoc chia ~160 GB, 160/36 = 4

CODE="/home/giahuy/Code/job/DFT/bench_dipole_152/_src/calc_dipmomt_dft.py"
INPUT="/home/giahuy/Code/job/DFT/bench_dipole_152/geometry/sp_inputs.json"
OUTDIR="/data/giahuy/Result/DFT/dipole_moments/sp"

mkdir -p /data/giahuy/Result/DFT/dipole_moments/_output
mkdir -p /data/giahuy/Result/DFT/dipole_moments/_error
mkdir -p "$OUTDIR/rows"

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

python "$CODE" --input "$INPUT" --outdir "$OUTDIR" \
       --funcs pbe0 b3lyp b2plyp

end=$(date +%s); rt=$((end - start))
printf "Task %s xong. Thoi gian: %02d:%02d:%02d\n" \
       "$SLURM_ARRAY_TASK_ID" $((rt/3600)) $(((rt%3600)/60)) $((rt%60))