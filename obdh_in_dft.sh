#!/bin/bash

#SBATCH --job-name=product_CNT_Sn2 ### Job name
#SBATCH --output=/data/giahuy/Result/OBDH_in_DFT/Menshutkin/_output/product_CNT_Sn2.out          ### Standard output file
#SBATCH --error=/data/giahuy/Result/OBDH_in_DFT/Menshutkin/_error/product_CNT_Sn2.err             ### Standard error file
#SBATCH --partition=Bigmem            ### queue
#SBATCH --nodes=1                     ### Number of nodes
#SBATCH --ntasks=1                    ### Number of tasks per node
#SBATCH --cpus-per-task=20            ### Number of CPU cores per task
#SBATCH --mem-per-cpu=6000

start=$(date +%s)
# input-file/code trong đường dẫn /home
INPUT_FILE="/home/giahuy/Code/job/OBDH/obdh_in_dft.py"
# Ghi output trực tiếp ra /data
OUTPUT_DIR="/data/giahuy/Result/OBDH_in_DFT/Menshutkin/$SLURM_JOB_ID"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/product_CNT_Sn2.txt"

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
export OPENBLAS_NUM_THREADS=20
export OMP_NUM_THREADS=20

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
