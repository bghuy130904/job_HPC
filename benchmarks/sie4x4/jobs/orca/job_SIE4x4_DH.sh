#!/bin/bash
#SBATCH --job-name=SIE4x4_DH
#SBATCH --output=/data/giahuy/Result/ORCA/SIE4x4_DH/_output/SIE_%x_%j.out
#SBATCH --error=/data/giahuy/Result/ORCA/SIE4x4_DH/_error/SIE_%x_%j.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G

start=$(date +%s)

# ====================================================================#
# Scratch
# ====================================================================#
export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$JOB_SCRATCH_PATH"

# ====================================================================#
# Modules / ORCA
# ====================================================================#
module load mpi-4.1
export LD_LIBRARY_PATH="/home/giahuy/venvs_py3_9/lib/python3.13/site-packages/orca_5_0_4:$LD_LIBRARY_PATH"
ORCA_DIR="/home/giahuy/venvs_py3_9/lib/python3.13/site-packages/orca_5_0_4"
ORCA_EXE="$ORCA_DIR/orca"
MERGE_EXE="$ORCA_DIR/orca_mergefrag"

# ====================================================================#
# Thu muc
# ====================================================================#
INP_DIR="/home/giahuy/Code/job/SIE4x4_DH"      # chua toan bo *.inp da sinh
OUT_DIR="/data/giahuy/Result/ORCA/SIE4x4_DH/$SLURM_JOB_ID"
DEBUG_DIR="/data/giahuy/Result/ORCA/SIE4x4_DH/_debug/$SLURM_JOB_ID"
mkdir -p "$OUT_DIR" "$DEBUG_DIR"

FUNCTIONALS=(PBE B2PLYP B2GPPLYP DSDPBEP86 PWPB95)   # ten file (da bo dau '-')
SYSTEMS=(H2_plus_He He2_plus NH3_2_plus H2O_2_plus)
POINTS=(R_1.0 R_1.25 R_1.5 R_1.75 dissociation_limit)

# ====================================================================#
# run_orca <inp_path> [gbw_to_copy_in ...]
#   Chay trong scratch rieng; copy .out/.gbw ve OUT_DIR.
#   KHONG exit khi loi: SIE4x4 co the co diem khong hoi tu -> ghi nhan roi
#   di tiep, viec loc nghiem se do script Python o Buoc 3 xu ly.
# ====================================================================#
run_orca() {
    local inp="$1"; shift
    local base; base=$(basename "$inp" .inp)
    local work="$JOB_SCRATCH_PATH/$base"

    if [ -f "$OUT_DIR/${base}.out" ] && grep -q "FINAL SINGLE POINT ENERGY" "$OUT_DIR/${base}.out"; then
        echo "    [skip] $base (da co ket qua)"          # cho phep resume job
        return 0
    fi

    mkdir -p "$work"
    cp "$inp" "$work/"
    for extra in "$@"; do cp "$extra" "$work/" 2>/dev/null; done
    cd "$work" || return 1

    "$ORCA_EXE" "$(basename "$inp")" > "$OUT_DIR/${base}.out" 2> "$OUT_DIR/${base}.err"
    local status=$?

    if [ $status -ne 0 ] || ! grep -q "FINAL SINGLE POINT ENERGY" "$OUT_DIR/${base}.out"; then
        echo "    [FAIL] $base (exit=$status) -> debug"
        mkdir -p "$DEBUG_DIR/$base"
        rsync -a "$work/" "$DEBUG_DIR/$base/"
        cd "$JOB_SCRATCH_PATH" && rm -rf "$work"
        return 1
    fi

    cp -f "$work/${base}.gbw" "$OUT_DIR/" 2>/dev/null || true
    cd "$JOB_SCRATCH_PATH" && rm -rf "$work"
    echo "    [ok]   $base"
    return 0
}

# ====================================================================#
# Buoc 1-2: voi moi (functional, he, diem)
#   PBE  : chi chay _def
#   DH   : fragA -> fragB -> mergefrag -> _loc, va _def
# ====================================================================#
for F in "${FUNCTIONALS[@]}"; do
  for S in "${SYSTEMS[@]}"; do
    for P in "${POINTS[@]}"; do
      TAG="${F}_${S}_${P}"
      echo "=== $TAG ==="

      # (a) nghiem tu guess mac dinh -> thuong ra basin DELOCALIZED
      run_orca "$INP_DIR/${TAG}_def.inp"

      # (b) PBE khong can localized guess (chi 1 basin)
      [ "$F" = "PBE" ] && continue

      # (c) fragment guess -> basin LOCALIZED
      run_orca "$INP_DIR/${TAG}_fragA.inp" || continue
      run_orca "$INP_DIR/${TAG}_fragB.inp" || continue

      MW="$JOB_SCRATCH_PATH/merge_${TAG}"
      mkdir -p "$MW" && cd "$MW"
      cp "$OUT_DIR/${TAG}_fragA.gbw" "$OUT_DIR/${TAG}_fragB.gbw" . 2>/dev/null
      # orca_mergefrag <frag1.gbw> <frag2.gbw> <merged.gbw>
      # THU TU PHAI KHOP thu tu nguyen tu trong file _loc.inp (fragA truoc)
      "$MERGE_EXE" "${TAG}_fragA.gbw" "${TAG}_fragB.gbw" merged.gbw > merge.log 2>&1
      if [ ! -f merged.gbw ]; then
          echo "    [FAIL] mergefrag $TAG"
          cp merge.log "$DEBUG_DIR/merge_${TAG}.log" 2>/dev/null
          cd "$JOB_SCRATCH_PATH" && rm -rf "$MW"; continue
      fi
      cp merged.gbw "$OUT_DIR/${TAG}_merged.gbw"
      cd "$JOB_SCRATCH_PATH" && rm -rf "$MW"

      run_orca "$INP_DIR/${TAG}_loc.inp" "$OUT_DIR/${TAG}_merged.gbw"
    done
  done
done

# ====================================================================#
# Buoc 3: loc nghiem + tinh De  (trich <S^2>, spin pop, chon nghiem
#         HOP LE THAP NHAT giua _def va _loc)
# ====================================================================#
echo ""
echo "====== TRICH XUAT & TINH De ======"
export OUT_DIR
python3 "$INP_DIR/collect_SIE4x4.py"

end=$(date +%s); rt=$((end-start))
printf "\nThoi gian: %02d:%02d:%02d\n" $((rt/3600)) $(((rt%3600)/60)) $((rt%60))
rm -rf "$JOB_SCRATCH_PATH"
