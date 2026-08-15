#!/usr/bin/env python3
"""
Tính dipole moment tham chiếu mức (U)CCSD cho một bộ phân tử đọc từ JSON.

Cải tiến so với bản cũ:
  - mol.max_memory được set theo RAM thực SLURM cấp (tránh out-of-core oan).
  - DF thực sự cho CCSD (cc.CCSD(mf).density_fit()) — auxbasis RI-C tự chọn,
    KHÔNG ép def2-universal-jkfit (jkfit chỉ dành cho SCF, không hợp cho tương quan).
  - Ghi kết quả TĂNG DẦN: mỗi phân tử ra 1 file row riêng -> crash/hết giờ vẫn
    giữ được phần đã chạy; an toàn cho SLURM job array (không ghi đè lẫn nhau).
  - Chạy được 2 chế độ:
        * 1 phân tử/lần  -> dùng SLURM_ARRAY_TASK_ID hoặc --index
        * toàn bộ vòng lặp -> khi không có index
  - Thêm cột chẩn đoán: converged, <S^2>, dipole x/y/z, E(UHF), E(UCCSD).
  - --merge: gộp tất cả file row -> 1 xlsx + 1 csv.

Ví dụ:
    # array: mỗi task 1 phân tử
    python calc_dipmomt_ccsd.py --input sp_inputs.json --outdir /data/.../out
    # gộp cuối cùng
    python calc_dipmomt_ccsd.py --input sp_inputs.json --outdir /data/.../out --merge
"""

import os
import sys
import json
import time
import glob
import argparse
from functools import reduce

import numpy as np
import pandas as pd
from pyscf import gto, scf, cc

from pycmf.OBDH.stability import stabilize_scf


# ----------------------------------------------------------------------
# def compute_ccsd_dipole(mycc, mf, mol, unit="Debye"):
#     """rdm1 (MO) -> AO -> co với tích phân lưỡng cực. Hỗ trợ cả RHF và UHF."""
#     dm1 = mycc.make_rdm1()

#     if isinstance(mf.mo_coeff, (tuple, list)):
#         mo_a, mo_b = mf.mo_coeff
#         dm1a, dm1b = dm1
#         dm1_ao = reduce(np.dot, (mo_a, dm1a, mo_a.T)) \
#                + reduce(np.dot, (mo_b, dm1b, mo_b.T))
#     else:
#         dm1_ao = reduce(np.dot, (mf.mo_coeff, dm1, mf.mo_coeff.T))

#     with mol.with_common_orig((0, 0, 0)):
#         ao_dip = mol.intor_symmetric("int1e_r", comp=3)

#     el_dip = np.einsum("xij,ji->x", ao_dip, dm1_ao).real
#     nucl_dip = np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords())
#     mol_dip = nucl_dip - el_dip

#     if unit.upper() == "DEBYE":
#         mol_dip = mol_dip * 2.541746  # 1 a.u. = 2.541746 Debye
#     return mol_dip


# ----------------------------------------------------------------------
def run_one(mol_name, properties, max_memory):
    """Chạy 1 phân tử, trả về dict kết quả (không raise ra ngoài)."""
    t0 = time.time()
    row = {
        "molecule": mol_name,
        "charge": properties.get("charge"),
        "spin": properties.get("spin"),
        "converged": None,
        "S2": None,
        "E_UHF": None,
        "E_UCCSD": None,
        "dip_x": None, "dip_y": None, "dip_z": None,
        "dipole_debye": None,
        "walltime_s": None,
        "status": "ok",
    }
    try:
        pyscf_geom = [(a[0], tuple(a[1])) for a in properties["geometry"]]

        mol = gto.Mole()
        mol.atom = pyscf_geom
        mol.charge = properties["charge"]
        mol.spin = properties["spin"]        # 2S = Nalpha - Nbeta
        mol.basis = {"default": "aug-cc-pcvqz", "H": "aug-cc-pvqz"}
        mol.max_memory = max_memory          # <-- quan trọng
        mol.verbose = 0
        mol.build()

        # SCF: UHF + density fitting (jkfit đúng cho SCF)
        mf = scf.UHF(mol).density_fit(auxbasis="def2-universal-jkfit")
        mf.verbose = 0
        mf.max_cycle = 100
        mf.kernel()

        # Ổn định hóa nghiệm SCF
        mf = stabilize_scf(mf, max_macro_cycles=10, verbose=True)

        g = mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock(dm=mf.make_rdm1()))
        row["grad_norm"] = float(np.linalg.norm(g))
        row["converged"] = bool(mf.converged)
        if row["grad_norm"] > 1e-4:
            row["status"] = "SCF_not_converged"
        row["converged"] = bool(mf.converged)
        row["E_UHF"] = float(mf.e_tot)
        try:
            row["S2"] = float(mf.spin_square()[0])   # theo dõi spin contamination
        except Exception:
            pass

        # DF-CCSD: KHÔNG truyền auxbasis -> PySCF tự chọn RI-C phù hợp tương quan
        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        mycc.kernel()
        row["E_UCCSD"] = float(mycc.e_tot)           # total, không phải chỉ ecorr

        dm_ao = mycc.make_rdm1(ao_repr=True)                  # 1-RDM ở AO
        dipvec = scf.hf.dip_moment(mol, dm_ao, unit='Debye')  # vector [x, y, z]
        row["dip_x"], row["dip_y"], row["dip_z"] = [float(v) for v in dipvec]
        row["dipole_debye"] = float(np.linalg.norm(dipvec))

    except Exception as e:  # nuốt lỗi để 1 phân tử hỏng không giết cả job
        row["status"] = f"ERROR: {type(e).__name__}: {e}"

    row["walltime_s"] = round(time.time() - t0, 2)
    return row


# ----------------------------------------------------------------------
def write_row(outdir, idx, row):
    rows_dir = os.path.join(outdir, "rows")
    os.makedirs(rows_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in row["molecule"])
    path = os.path.join(rows_dir, f"{idx:03d}_{safe}.csv")
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def do_merge(outdir, output_xlsx):
    rows_dir = os.path.join(outdir, "rows")
    files = sorted(glob.glob(os.path.join(rows_dir, "*.csv")))
    if not files:
        print(f"[MERGE] Không tìm thấy file row nào trong {rows_dir}")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    os.makedirs(os.path.dirname(output_xlsx) or ".", exist_ok=True)
    df.to_excel(output_xlsx, index=False)
    df.to_csv(os.path.splitext(output_xlsx)[0] + ".csv", index=False)
    n_err = (df["status"] != "ok").sum()
    print(f"[MERGE] {len(df)} phân tử -> {output_xlsx}  (lỗi/không hội tụ: {n_err})")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="đường dẫn file JSON")
    ap.add_argument("--outdir", required=True, help="thư mục ghi kết quả")
    ap.add_argument("--index", type=int, default=None,
                    help="chạy 1 phân tử theo chỉ số (0-based)")
    ap.add_argument("--max-memory", type=int,
                    default=int(os.environ.get("PYSCF_MAX_MEMORY", "78000")),
                    help="RAM cho PySCF, MB (mặc định 78000)")
    ap.add_argument("--merge", action="store_true",
                    help="gộp các file row -> xlsx rồi thoát")
    ap.add_argument("--output-xlsx", default=None,
                    help="tên file xlsx khi merge (mặc định <outdir>/dipole_results.xlsx)")
    args = ap.parse_args()

    output_xlsx = args.output_xlsx or os.path.join(args.outdir, "dipole_results.xlsx")

    if args.merge:
        do_merge(args.outdir, output_xlsx)
        return

    with open(args.input) as f:
        data = json.load(f)
    items = list(data.items())  # thứ tự ổn định để index khớp giữa các task

    # xác định index: --index > SLURM_ARRAY_TASK_ID > (không có -> chạy hết)
    idx = args.index
    if idx is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if idx is not None:
        if idx < 0 or idx >= len(items):
            print(f"[SKIP] index {idx} ngoài phạm vi 0..{len(items)-1}")
            return
        name, props = items[idx]
        print(f"[{idx}] {name} ...", flush=True)
        row = run_one(name, props, args.max_memory)
        p = write_row(args.outdir, idx, row)
        print(f"[{idx}] {name}: {row['status']}, dipole={row['dipole_debye']}, "
              f"{row['walltime_s']}s -> {p}", flush=True)
    else:
        for i, (name, props) in enumerate(items):
            print(f"[{i}/{len(items)}] {name} ...", flush=True)
            row = run_one(name, props, args.max_memory)
            write_row(args.outdir, i, row)
            print(f"    -> {row['status']}, dipole={row['dipole_debye']}, "
                  f"{row['walltime_s']}s", flush=True)
        do_merge(args.outdir, output_xlsx)


if __name__ == "__main__":
    main()