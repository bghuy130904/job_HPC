#!/bin/bash

#SBATCH --job-name=obdh_conv_C12_aldehyde ### Job name
#SBATCH --output=/data/giahuy/Result/OBDH_in_DFT/bench_conv_C12_aldehyde/_output/aldehyde_obdh.out
#SBATCH --error=/data/giahuy/Result/OBDH_in_DFT/bench_conv_C12_aldehyde/_error/aldehyde_obdh.err
#SBATCH --partition=Bigmem            ### queue
#SBATCH --nodes=1                     ### Number of nodes
#SBATCH --ntasks=1                    ### Number of tasks per node
#SBATCH --cpus-per-task=12            ### Number of CPU cores per task
#SBATCH --threads-per-core=1
#SBATCH --mem=50G

start=$(date +%s)
# input-file/code trong đường dẫn /home
INPUT_FILE="/home/giahuy/Code/job/benchmarks/aldehyde_cl/methods/obdh/obdh.py"
# Ghi output trực tiếp ra /data
OUTPUT_DIR="/data/giahuy/Result/OBDH_in_DFT/bench_conv_C12_aldehyde/dissociation_C45_1.2R0/"
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

PYTHONPATH= python -u "$INPUT_FILE" \
    --xyz "/home/giahuy/Code/job/benchmarks/aldehyde_cl/inputs/aldehyde_C45_1.2R0.xyz" \
    --basis cc-pvdz \
    --charge 1 --spin 1\
    --alpha 0.5 0.4 \
    --threads "$SLURM_CPUS_PER_TASK" \
    --max-memory 40000 \
    --output "${OUTPUT_DIR%/}/obdh_aldehyde_cl_new.json" \
    --shells 0 1 \
    #--resume \
    "$@" \
    #>> "$OUTPUT_FILE" 2>&1
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

