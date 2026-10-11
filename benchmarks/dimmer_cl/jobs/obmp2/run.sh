#!/bin/bash

#SBATCH --job-name=intE_dimmer_cl ### Job name
#SBATCH --output=/data/giahuy/Result/OBMP2_in_DFT/dimmer_cl/_output/intE.out          ### Standard output file
#SBATCH --error=/data/giahuy/Result/OBMP2_in_DFT/dimmer_cl/_error/intE.err             ### Standard error file
#SBATCH --partition=Bigmem            ### queue
#SBATCH --nodes=1                     ### Number of nodes
#SBATCH --ntasks=1                    ### Number of tasks per node
#SBATCH --cpus-per-task=12            ### Number of CPU cores per task
#SBATCH --threads-per-core=1
#SBATCH --mem=100G

start=$(date +%s)
# input-file/code trong đường dẫn /home
INPUT_FILE="/home/giahuy/Code/job/benchmarks/dimmer_cl/methods/obmp2/intE.py"
export METHOD="${METHOD:-obmp2}"
# Ghi output trực tiếp ra /data
OUTPUT_DIR="/data/giahuy/Result/OBMP2_in_DFT/dimmer_cl/$METHOD/$SLURM_JOB_ID/"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/intE_dimmer_cl.txt"
export RUN_DIR="$OUTPUT_DIR"

# ====================================================================#
# LƯU Ý: 2 DÒNG COMMAND NÀY LÀ BẮT BUỘC PHẢI CÓ TRONG FILE SUBMIT JOB
# ====================================================================#
export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR"
export PYSCF_SCRATCH="$TMPDIR"
# ====================================================================#

#Load các Module như bình thường

module load python3.9
source /home/giahuy/.venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Tham số của driver intE được truyền qua biến môi trường
export BASIS="${BASIS:-def2-QZVPPD}"
export MEMORY="${MEMORY:-90000}"

PYTHONPATH= python -u "$INPUT_FILE" > "$OUTPUT_FILE" 2>&1

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
