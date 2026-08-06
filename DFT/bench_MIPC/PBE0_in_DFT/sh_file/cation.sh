#!/bin/bash

#SBATCH --job-name=cation_3 ### Job name
#SBATCH --output=/data/giahuy/Result/DFT_in_DFT/PBE0_in_DFT/bench_MIPC/_output/cation_3.out          ### Standard output file
#SBATCH --error=/data/giahuy/Result/DFT_in_DFT/PBE0_in_DFT/bench_MIPC/_error/cation_3.err             ### Standard error file
#SBATCH --partition=Bigmem            ### queue
#SBATCH --nodes=1                     ### Number of nodes
#SBATCH --ntasks=1                    ### Number of tasks per node
#SBATCH --cpus-per-task=16            ### Number of CPU cores per task
#SBATCH --threads-per-core=1
#SBATCH --mem=100G

start=$(date +%s)
# input-file/code trong đường dẫn /home
INPUT_FILE="/home/giahuy/Code/job/DFT/bench_MIPC/PBE0_in_DFT/_src/cation.py"
# Ghi output trực tiếp ra /data
OUTPUT_DIR="/data/giahuy/Result/DFT_in_DFT/PBE0_in_DFT/bench_MIPC/system_3"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/cation_3.txt"

# ====================================================================#
# LƯU Ý: 2 DÒNG COMMAND NÀY LÀ BẮT BUỘC PHẢI CÓ TRONG FILE SUBMIT JOB
# ====================================================================#
export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"
# ====================================================================#

#Load các Module như bình thường

module load python3.9
source /home/giahuy/.venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

PYTHONPATH= python $INPUT_FILE > $OUTPUT_FILE

echo "Job hoàn tất."
echo "Output đã được ghi trực tiếp vào: $OUTPUT_FILE"

#Ket thuc dem gio
end=$(date +%s)

runtime=$((end - start))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

printf "Thời gian: %02d:%02d:%02d (giờ:phút:giây)\n" $hours $minutes $seconds
