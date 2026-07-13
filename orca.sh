#!/bin/bash
#SBATCH --job-name=2YNV_DLPNO
#SBATCH --output=/data/giahuy/Result/ORCA/DLPNO_2YNV/_output/2YNV_%x_%j.out
#SBATCH --error=/data/giahuy/Result/ORCA/DLPNO_2YNV/_error/2YNV_%x_%j.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=16                   # ORCA dùng ntasks (MPI), không phải cpus-per-task
##SBATCH --cpus-per-task=16
#SBATCH --mem=128G                     # Tổng RAM, ~4000MB x 16 core = 64GB + buffer


start=$(date +%s)

# ====================================================================#
# BẮT BUỘC: scratch path
# ====================================================================#
export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p $JOB_SCRATCH_PATH

# ====================================================================#
# Load modules
# ====================================================================#
module load mpi-4.1

export LD_LIBRARY_PATH="/home/giahuy/venvs_py3_9/lib/python3.13/site-packages/orca_5_0_4:$LD_LIBRARY_PATH"
ORCA_EXE="/home/giahuy/venvs_py3_9/lib/python3.13/site-packages/orca_5_0_4/orca"

# ====================================================================#
# Thư mục input/output
# Bạn đặt tất cả file .inp vào đây
# ====================================================================#
INP_DIR="/home/giahuy/Code/job/DLPNO_2YNV"
OUT_DIR="/data/giahuy/Result/ORCA/DLPNO_2YNV/$SLURM_JOB_ID"
DEBUG_DIR="/data/giahuy/Result/ORCA/DLPNO_2YNV/_debug/$SLURM_JOB_ID"
mkdir -p $OUT_DIR

# ====================================================================#
# Hàm chạy ORCA
# ====================================================================#
run_orca() {
    local inp="$1"
    local base
    base=$(basename "$inp" .inp)

    local work="$JOB_SCRATCH_PATH/$base"
    mkdir -p "$work"

    cp "$inp" "$work/"
    cd "$work" || exit 1

    echo ">>> Đang chạy: $base"
    echo ">>> Workdir: $work"

    "$ORCA_EXE" "$(basename "$inp")" > "$OUT_DIR/${base}.out" 2> "$OUT_DIR/${base}.err"
    status=$?

    # Nếu ORCA lỗi, copy toàn bộ scratch sang DEBUG_DIR rồi dừng
    if [ $status -ne 0 ]; then
        echo "ERROR: ORCA failed for $base with exit code $status"
        mkdir -p "$DEBUG_DIR/$base"
        rsync -av "$work/" "$DEBUG_DIR/$base/"
        echo "Debug files đã được lưu ở: $DEBUG_DIR/$base"
        exit $status
    fi

    # Nếu ORCA không in FINAL SINGLE POINT ENERGY, cũng lưu debug rồi dừng
    if ! grep -q "FINAL SINGLE POINT ENERGY" "$OUT_DIR/${base}.out"; then
        echo "ERROR: $base không có FINAL SINGLE POINT ENERGY"
        mkdir -p "$DEBUG_DIR/$base"
        rsync -av "$work/" "$DEBUG_DIR/$base/"
        echo "Debug files đã được lưu ở: $DEBUG_DIR/$base"
        exit 1
    fi

    # Nếu chạy thành công, chỉ copy file quan trọng về OUT_DIR
    cp -f "$work"/*.gbw "$OUT_DIR/" 2>/dev/null || true
    cp -f "$work"/*_property.txt "$OUT_DIR/" 2>/dev/null || true
    cp -f "$work"/*.densities "$OUT_DIR/" 2>/dev/null || true

    # Không copy .tmp, .loc, .mdcip.tmp về OUT_DIR
    # Xóa scratch của calculation này sau khi đã chạy thành công
    rm -rf "$work"

    echo ">>> Xong: $base"
}
# ====================================================================#
# Bước 1: Chạy Mg2+ monomer CHỈ 1 LẦN
# ====================================================================#
echo "====== Mg2+ monomer (dùng chung cho tất cả subclusters) ======"
run_orca "$INP_DIR/Mg.inp"
 
# ====================================================================#
# Bước 2: Chạy complex và ligands cho từng subcluster
# ====================================================================#
for N in 1 2 3 4 5; do
    echo "====== Subcluster N=$N ======"
    run_orca "$INP_DIR/N${N}_complex.inp"
    run_orca "$INP_DIR/N${N}_ligands.inp"
done
 
# ====================================================================#
# Bước 3: Extract energies và tính E_int
# Mg.out dùng chung cho tất cả N
# ====================================================================#
echo ""
echo "====== KẾT QUẢ INTERACTION ENERGIES ======"
export OUT_DIR
python3 << 'PYEOF'
import os
import re
 
OUT_DIR = os.environ.get("OUT_DIR", ".")
hartree2kcal = 627.509608
 
def get_energy(filepath):
    if not os.path.exists(filepath):
        print(f"  [MISSING] {filepath}")
        return None
    with open(filepath) as f:
        content = f.read()
    matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)", content)
    if matches:
        return float(matches[-1])
    print(f"  [NO ENERGY FOUND] {filepath}")
    return None
 
# Mg dùng chung
E_Mg = get_energy(f"{OUT_DIR}/Mg.out")
if E_Mg is None:
    print("ERROR: Không tìm thấy energy của Mg!")
    exit(1)
print(f"  E_Mg = {E_Mg:.6f} Hartree")
 
print("")
results = {}
for N in range(1, 6):
    E_complex = get_energy(f"{OUT_DIR}/N{N}_complex.out")
    E_ligands = get_energy(f"{OUT_DIR}/N{N}_ligands.out")
 
    if E_complex is None or E_ligands is None:
        continue
 
    e_int = (E_complex - E_Mg - E_ligands) * hartree2kcal
    results[N] = e_int
    print(f"  N={N}: E_complex={E_complex:.6f}  E_ligands={E_ligands:.6f}")
    print(f"       E_int = {e_int:.3f} kcal/mol")
 
print("")
print("  N  |  E_int (kcal/mol)")
print("  ---|-------------------")
for N, e in sorted(results.items()):
    print(f"  {N}  |  {e:.3f}")
PYEOF
 
# ====================================================================#
# Kết thúc đếm giờ
# ====================================================================#
echo ""
echo "Job hoàn tất."
end=$(date +%s)
runtime=$((end - start))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))
printf "Thời gian: %02d:%02d:%02d (giờ:phút:giây)\n" $hours $minutes $seconds

rm -rf "$JOB_SCRATCH_PATH"