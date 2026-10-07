#!/bin/bash

#SBATCH --job-name=dft_sie4x4 ### Job name
#SBATCH --output=/data/giahuy/Result/DFT/SIE_DFT/_output/sie4x4.out          ### Standard output file
#SBATCH --error=/data/giahuy/Result/DFT/SIE_DFT/_error/sie4x4.err             ### Standard error file
#SBATCH --partition=Bigmem            ### queue
#SBATCH --nodes=1                     ### Number of nodes
#SBATCH --ntasks=1                    ### Number of tasks per node
#SBATCH --cpus-per-task=10            ### Number of CPU cores per task
#SBATCH --mem-per-cpu=6000

start=$(date +%s)
# input-file/code trong đường dẫn /home
INPUT_FILE="/home/giahuy/Code/job/benchmarks/sie4x4/methods/dft/calc_DFT.py"
# Ghi output trực tiếp ra /data
OUTPUT_DIR="/data/giahuy/Result/DFT/SIE_DFT/$SLURM_JOB_ID"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/sie4x4.txt"

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

#Compile/run code
# Compile/run code
python "$INPUT_FILE" \
    --outdir "$OUTPUT_DIR" \
    --threads "$OMP_NUM_THREADS" \
    "$@" \
    --methods pbe pbe0 b3lyp dh_matched \
    --alpha 0.53 0.27 \
    > "$OUTPUT_FILE" 2>&1

echo "Job hoàn tất."
echo "Output đã được ghi trực tiếp vào: $OUTPUT_FILE"

#Ket thuc dem gio
end=$(date +%s)

runtime=$((end - start))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

printf "Thời gian: %02d:%02d:%02d (giờ:phút:giây)\n" $hours $minutes $seconds

