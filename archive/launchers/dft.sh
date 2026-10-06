#!/bin/bash

#SBATCH --job-name=dipmomt_sp ### Job name
#SBATCH --output=/data/giahuy/Result/DFT/dipole_moments/_output/dipmomt_sp.out          ### Standard output file
#SBATCH --error=/data/giahuy/Result/DFT/dipole_moments/_error/dipmomt_sp.err             ### Standard error file
#SBATCH --partition=normal            ### queue
#SBATCH --nodes=1                     ### Number of nodes
#SBATCH --ntasks=1                    ### Number of tasks per node
#SBATCH --cpus-per-task=13            ### Number of CPU cores per task
#SBATCH --mem-per-cpu=2500

start=$(date +%s)
# input-file/code trong đường dẫn /home
INPUT_FILE="/home/giahuy/Code/job/DFT/152_nsp.py"
# Ghi output trực tiếp ra /data
OUTPUT_DIR="/data/giahuy/Result/DFT/dipole_moments/$SLURM_JOB_ID"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/dipmomt_sp.txt"

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
export OPENBLAS_NUM_THREADS=13
export OMP_NUM_THREADS=13

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
