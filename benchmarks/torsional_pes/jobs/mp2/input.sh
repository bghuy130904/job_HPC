#!/bin/bash

#SBATCH --job-name=mp2_torsional_pes ### Job name
#SBATCH --output=/data/giahuy/Result/MP2_in_DFT/bench_torsional_pes/_output/torsional_mp2.out
#SBATCH --error=/data/giahuy/Result/MP2_in_DFT/bench_torsional_pes/_error/torsional_mp2.err
#SBATCH --partition=Bigmem            ### queue
#SBATCH --nodes=1                     ### Number of nodes
#SBATCH --ntasks=1                    ### Number of tasks per node
#SBATCH --cpus-per-task=12            ### Number of CPU cores per task
#SBATCH --threads-per-core=1
#SBATCH --mem=50G

start=$(date +%s)
# input-file/code trong đường dẫn /home
INPUT_FILE="/home/giahuy/Code/job/benchmarks/torsional_pes/methods/mp2/mp2.py"
# Ghi output trực tiếp ra /data
OUTPUT_DIR="/data/giahuy/Result/MP2_in_DFT/bench_torsional_pes/"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/torsional_pes.txt"

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

: > "$OUTPUT_FILE"
for molecule in ethanol 1-butanol; do
    PYTHONPATH= python -u "$INPUT_FILE" \
        --config "/home/giahuy/Code/job/benchmarks/torsional_pes/inputs/scan.json" \
        --molecule "$molecule" \
        --basis "6-311++G(2d,p)" \
        --xc-env "b3lyp" \
        --threads "$SLURM_CPUS_PER_TASK" \
        --max-memory 40000 \
        --output-dir "${OUTPUT_DIR%/}" \
        --shells 1 2 3 \
        "$@" \
        >> "$OUTPUT_FILE" 2>&1

    status=$?
    if [ "$status" -ne 0 ]; then
        echo "Tính toán thất bại, exit code: $status" >&2
        exit "$status"
    fi
done

echo "Job hoàn tất."
echo "Output đã được ghi trực tiếp vào: $OUTPUT_FILE"

#Ket thuc dem gio
end=$(date +%s)

runtime=$((end - start))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

printf "Thời gian: %02d:%02d:%02d (giờ:phút:giây)\n" $hours $minutes $seconds
