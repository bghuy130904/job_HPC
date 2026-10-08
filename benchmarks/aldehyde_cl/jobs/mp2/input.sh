#!/bin/bash

#SBATCH --job-name=mp2_conv_C12_aldehyde ### Job name
#SBATCH --output=aldehyde_mp2_%j.out
#SBATCH --error=aldehyde_mp2_%j.err
#SBATCH --partition=Bigmem            ### queue
#SBATCH --nodes=1                     ### Number of nodes
#SBATCH --ntasks=1                    ### Number of tasks per node
#SBATCH --cpus-per-task=12            ### Number of CPU cores per task
#SBATCH --threads-per-core=1
#SBATCH --mem=50G

start=$(date +%s)
# input-file/code trong đường dẫn /home
INPUT_FILE="/home/giahuy/Code/job/benchmarks/aldehyde_cl/methods/mp2/mp2.py"
# Ghi output trực tiếp ra /data
OUTPUT_DIR="/data/giahuy/Result/MP2_in_DFT/bench_conv_C12_aldehyde/"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/conv_C12_aldehyde.txt"

# ====================================================================#
# LƯU Ý: 2 DÒNG COMMAND NÀY LÀ BẮT BUỘC PHẢI CÓ TRONG FILE SUBMIT JOB
# ====================================================================#
export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR"
# ====================================================================#

#Load các Module như bình thường

module load python3.9
source /home/giahuy/.venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "$TMPDIR"

PYTHONPATH= python -u "$INPUT_FILE" \
    --xyz "/home/giahuy/Code/job/benchmarks/aldehyde_cl/inputs/aldehyde.xyz" \
    --basis cc-pvdz \
    --xc-env b3lyp \
    --threads "$SLURM_CPUS_PER_TASK" \
    --max-memory 45000 \
    --output "${OUTPUT_DIR%/}/mp2_aldehyde_cl_new.json" \
    --shells 0 \
    "$@" \
    > "$OUTPUT_FILE" 2>&1

status=$?
if [ "$status" -ne 0 ]; then
    echo "Tính toán thất bại, exit code: $status" >&2
    exit "$status"
fi

echo "Job hoàn tất."
echo "Output đã được ghi trực tiếp vào: $OUTPUT_FILE"

#Ket thuc dem gio
end=$(date +%s)

runtime=$((end - start))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

printf "Thời gian: %02d:%02d:%02d (giờ:phút:giây)\n" $hours $minutes $seconds

