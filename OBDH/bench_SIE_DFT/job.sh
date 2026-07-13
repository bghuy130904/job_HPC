#!/bin/bash

#SBATCH --job-name=obdh_react_energy ### Job name
#SBATCH --output=/data/giahuy/Result/OBDH/SIE_DFT/_output/react_energy.out          ### Standard output file
#SBATCH --error=/data/giahuy/Result/OBDH/SIE_DFT/_error/react_energy.err             ### Standard error file
#SBATCH --partition=Bigmem            ### queue
#SBATCH --nodes=1                     ### Number of nodes
#SBATCH --ntasks=1                    ### Number of tasks per node
#SBATCH --cpus-per-task=10            ### Number of CPU cores per task
#SBATCH --mem-per-cpu=5000

start=$(date +%s)
# input-file/code trong đường dẫn /home
INPUT_FILE="/home/giahuy/Code/job/OBDH/bench_SIE_DFT/_src/calc_OBDH.py"
# Ghi output trực tiếp ra /data
OUTPUT_DIR="/data/giahuy/Result/OBDH/SIE_DFT/$SLURM_JOB_ID"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/react_energy.txt"

# ====================================================================#
# LƯU Ý: 2 DÒNG COMMAND NÀY LÀ BẮT BUỘC PHẢI CÓ TRONG FILE SUBMIT JOB
# ====================================================================#
export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"
# ====================================================================#

#Load các Module như bình thường

module load python3.9
source /home/giahuy/venvs_py3_9/bin/activate

export PYTHONPATH="~/venvs_py3_9/lib/python3.13/site-packages:$PYTHONPATH"
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

#Compile/run code
python $INPUT_FILE > $OUTPUT_FILE

echo "Job hoàn tất."
echo "Output đã được ghi trực tiếp vào: $OUTPUT_FILE"

#Ket thuc dem gio
end=$(date +%s)

runtime=$((end - start))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

printf "Thời gian: %02d:%02d:%02d (giờ:phút:giây)\n" $hours $minutes $seconds
