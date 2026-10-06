#!/bin/bash
#SBATCH --job-name=SIE_ORCA
#SBATCH --output=SIE_%x_%j.out
#SBATCH --error=SIE_%x_%j.err
#SBATCH --partition=Bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIE_ROOT="$REPO_ROOT/benchmarks/sie4x4"
: "${ORCA_DIR:?Set ORCA_DIR to the ORCA installation directory}"
ORCA_EXE="$ORCA_DIR/orca"
MERGE_EXE="$ORCA_DIR/orca_mergefrag"
OUTDIR="${OUTDIR:-$REPO_ROOT/results/sie4x4/orca/${SLURM_JOB_ID:-local}}"
mkdir -p "$OUTDIR"
OUTDIR=$(cd "$OUTDIR" && pwd)
# Inputs generated for this run/config; do not reuse stale tracked .inp files.
INP_DIR="$OUTDIR/generated_inputs"
export JOB_SCRATCH_PATH="${JOB_SCRATCH_PATH:-/scratch/${USER}/sie-orca-${SLURM_JOB_ID:-$$}}"
export TMPDIR="$JOB_SCRATCH_PATH"
mkdir -p "$TMPDIR"
python "$SIE_ROOT/tools/gen_orca.py" --outdir "$INP_DIR" \
 --basis "${BASIS:-aug-cc-pVDZ}" --nproc "${SLURM_NTASKS:-16}"
failed=0
run_orca() {
 local base="$1"; local guess="${2:-}"; local work="$TMPDIR/$base"
 mkdir -p "$work"
 cp "$INP_DIR/$base.inp" "$work/"
 # MOREAD input always expects this exact basename.
 if [ -n "$guess" ]; then cp "$guess" "$work/merged.gbw"; fi
 if ! (cd "$work" && "$ORCA_EXE" "$base.inp" > "$OUTDIR/$base.out" 2> "$OUTDIR/$base.err"); then
  failed=$((failed+1)); return 1
 fi
 if ! grep -q 'ORCA TERMINATED NORMALLY' "$OUTDIR/$base.out"; then
  failed=$((failed+1)); return 1
 fi
 if [ -f "$work/$base.gbw" ]; then cp "$work/$base.gbw" "$OUTDIR/"; fi
}
FUNCTIONALS=(PBE B2PLYP B2GPPLYP DSDPBEP86 PWPB95)
SYSTEMS=(H2_plus_He He2_plus NH3_2_plus H2O_2_plus)
POINTS=(R_1.0 R_1.25 R_1.5 R_1.75 dissociation_limit)
for f in "${FUNCTIONALS[@]}"; do
 for s in "${SYSTEMS[@]}"; do
  for p in "${POINTS[@]}"; do
   tag="${f}_${s}_${p}"
   run_orca "${tag}_def" || true
   for variant in loc locB; do
    run_orca "${tag}_${variant}_fragA" || continue
    run_orca "${tag}_${variant}_fragB" || continue
    work="$TMPDIR/merge_${tag}_${variant}"
    mkdir -p "$work"
    cp "$OUTDIR/${tag}_${variant}_fragA.gbw" "$OUTDIR/${tag}_${variant}_fragB.gbw" "$work/"
    if ! (cd "$work" && "$MERGE_EXE" "${tag}_${variant}_fragA.gbw" "${tag}_${variant}_fragB.gbw" merged.gbw > merge.log 2>&1); then
     failed=$((failed+1)); continue
    fi
    if [ ! -f "$work/merged.gbw" ]; then failed=$((failed+1)); continue; fi
    run_orca "${tag}_${variant}" "$work/merged.gbw" || true
   done
  done
 done
done
# Keep failed scratch for inspection; no automatic destructive cleanup.
python "$SIE_ROOT/tools/collect_SIE4x4.py" --outdir "$OUTDIR" --recipes "$INP_DIR/recipes.json"
[ "$failed" -eq 0 ]
