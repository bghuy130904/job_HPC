#!/usr/bin/env python3
"""
Tính dipole moment tham chiếu mức (U)CCSD cho một bộ phân tử đọc từ JSON.

Thay đổi so với bản trước:
  - SCF chạy nhiều initial guess ('minao','atom','1e'), giữ nghiệm có năng
    lượng THẤP NHẤT. Bắt được trường hợp aufbau rơi vào trạng thái điện tử
    sai (vd C2H: minao -> 2-Pi sai, atom/1e -> 2-Sigma+ đúng, thấp hơn 15 mH).
  - Ghi cột E_spread_mH = (E_cao nhất - E_thấp nhất) giữa các guess.
    > 0.1 mHartree  => có trạng thái cạnh tranh, PHẢI xem tay.
  - Ghi cột grad_norm = ||dE/dkappa|| đo TRỰC TIẾP sau stabilize_scf.
    Đây là bằng chứng hội tụ duy nhất đáng tin; cờ mf.converged có thể cũ.
  - Import pyscf/pycmf theo kiểu lazy -> chế độ --merge chạy được ở bất kỳ
    môi trường nào chỉ cần pandas.
  - Thêm --only để chạy lại vài chất theo tên (vd sau khi sửa hình học).

Ví dụ:
    # SLURM array: mỗi task 1 phân tử
    python calc_dipmomt_ccsd.py --input sp_inputs.json --outdir /data/.../sp
    # chạy lại vài chất
    python calc_dipmomt_ccsd.py --input sp_inputs.json --outdir /data/.../sp \
           --only C2H PS O3
    # gộp
    python calc_dipmomt_ccsd.py --input sp_inputs.json --outdir /data/.../sp --merge
"""

import os
import sys
import json
import time
import glob
import argparse

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Ngưỡng
GRAD_TOL      = 1e-4    # ||g|| lớn hơn mức này  -> coi như CHƯA hội tụ
SPREAD_WARN   = 0.1     # mHartree, chênh lệch giữa các guess -> cảnh báo
INIT_GUESSES  = ('minao', 'atom', '1e')   # KHÔNG dùng 'huckel': với CN nó
                                          # chạy >25 phút trong khi các guess
                                          # khác chỉ mất ~25 giây.
BASIS = {"default": "aug-cc-pcvqz", "H": "aug-cc-pvqz"}


def _lazy_imports():
    """Nạp pyscf/pycmf khi thật sự cần. Giữ --merge độc lập với pyscf."""
    global gto, scf, cc, stabilize_scf
    from pyscf import gto, scf, cc                      # noqa: F401
    from pycmf.OBDH.stability import stabilize_scf      # noqa: F401
    # numpy>=2 bỏ numpy.linalg.linalg, một số bản pyscf soscf còn tham chiếu
    if not hasattr(np.linalg, 'linalg'):
        np.linalg.linalg = np.linalg


# ----------------------------------------------------------------------
def grad_norm(mf):
    """||gradient|| của SCF. Đo trực tiếp, không tin cờ mf.converged."""
    dm = mf.make_rdm1()
    fock = mf.get_fock(dm=dm)
    return float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock)))


def run_scf_multiguess(mol, verbose_stab=False):
    """
    Chạy UHF từ nhiều initial guess, mỗi lần đều:
        DIIS -> (nếu chưa hội tụ) Newton -> stabilize_scf
    Trả về (mf_tốt_nhất, spread_mHartree, danh_sách_kết_quả).

    Ba bước trên giải ba bài toán KHÁC nhau, không thay thế nhau được:
      DIIS          : tìm điểm dừng (nhanh, có thể kẹt)
      Newton        : ép về điểm dừng khi DIIS bò chậm (vd PS, hệ 2-Pi suy biến)
      stabilize_scf : rời khỏi điểm yên ngựa sang cực tiểu thật (vd O3, hạ 88 mH)
      nhiều guess   : thoát khỏi TRẠNG THÁI ĐIỆN TỬ sai (vd C2H, hạ 15 mH)
    Lưu ý: stability() chỉ có nghĩa khi đã ở điểm dừng, nên Newton phải
    chạy TRƯỚC stabilize_scf.
    """
    results = []
    best = None
    for guess in INIT_GUESSES:
        try:
            mf = scf.UHF(mol).density_fit(auxbasis="def2-universal-jkfit")
            mf.verbose = 0
            mf.init_guess = guess
            mf.max_cycle = 150
            mf.kernel()

            if not mf.converged:
                mf = mf.newton()
                mf.kernel(mf.mo_coeff, mf.mo_occ)

            mf = stabilize_scf(mf, max_macro_cycles=10, verbose=verbose_stab)

            g = grad_norm(mf)
            results.append({"guess": guess, "E": float(mf.e_tot), "grad": g})
            # chỉ nhận nghiệm đã hội tụ làm ứng viên
            if g <= GRAD_TOL and (best is None or mf.e_tot < best.e_tot - 1e-8):
                best = mf
        except Exception as e:
            results.append({"guess": guess, "err": f"{type(e).__name__}: {e}"})

    if best is None:
        # không guess nào cho nghiệm hội tụ -> lấy cái năng lượng thấp nhất
        ok = [r for r in results if "E" in r]
        if not ok:
            raise RuntimeError("tất cả initial guess đều lỗi")
        # chạy lại guess tốt nhất để lấy object mf
        gbest = min(ok, key=lambda r: r["E"])["guess"]
        mf = scf.UHF(mol).density_fit(auxbasis="def2-universal-jkfit")
        mf.verbose = 0
        mf.init_guess = gbest
        mf.max_cycle = 150
        mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            mf.kernel(mf.mo_coeff, mf.mo_occ)
        best = stabilize_scf(mf, max_macro_cycles=10, verbose=verbose_stab)

    energies = [r["E"] for r in results if "E" in r]
    spread = (max(energies) - min(energies)) * 1000.0 if len(energies) > 1 else 0.0
    return best, spread, results


# ----------------------------------------------------------------------
def run_one(mol_name, properties, max_memory):
    """Chạy 1 phân tử, trả về dict kết quả (không raise ra ngoài)."""
    t0 = time.time()
    row = {
        "molecule": mol_name,
        "charge": properties.get("charge"),
        "spin": properties.get("spin"),
        "nao": None,
        "converged": None,
        "grad_norm": None,
        "E_spread_mH": None,
        "n_guess_ok": None,
        "S2": None,
        "E_UHF": None,
        "E_UCCSD": None,
        "dip_x": None, "dip_y": None, "dip_z": None,
        "dipole_debye": None,
        "walltime_s": None,
        "status": "ok",
    }
    try:
        mol = gto.Mole()
        mol.atom = [(a[0], tuple(a[1])) for a in properties["geometry"]]
        mol.charge = properties["charge"]
        mol.spin = properties["spin"]          # 2S = Nalpha - Nbeta
        mol.basis = BASIS
        mol.max_memory = max_memory
        mol.verbose = 0
        mol.build()
        row["nao"] = int(mol.nao)

        mf, spread, results = run_scf_multiguess(mol)
        row["E_spread_mH"] = round(float(spread), 6)
        row["n_guess_ok"] = sum(1 for r in results if "E" in r)
        row["E_UHF"] = float(mf.e_tot)
        row["converged"] = bool(mf.converged)

        g = grad_norm(mf)
        row["grad_norm"] = g
        if g > GRAD_TOL:
            row["status"] = "SCF_not_converged"
        elif spread > SPREAD_WARN:
            row["status"] = f"WARN_multi_state(spread={spread:.2f}mH)"

        try:
            row["S2"] = float(mf.spin_square()[0])
        except Exception:
            pass

        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        mycc.kernel()
        if not mycc.converged and row["status"] == "ok":
            row["status"] = "CCSD_not_converged"
        row["E_UCCSD"] = float(mycc.e_tot)

        dm_ao = mycc.make_rdm1(ao_repr=True)
        dipvec = scf.hf.dip_moment(mol, dm_ao, unit='Debye', verbose=0)
        row["dip_x"], row["dip_y"], row["dip_z"] = [float(v) for v in dipvec]
        row["dipole_debye"] = float(np.linalg.norm(dipvec))

    except Exception as e:
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
    df.to_csv(os.path.splitext(output_xlsx)[0] + ".csv", index=False)
    try:
        df.to_excel(output_xlsx, index=False)
    except ImportError:
        print("[MERGE] thiếu openpyxl -> chỉ ghi CSV. Cài: pip install openpyxl")

    print(f"[MERGE] {len(df)} phân tử -> {output_xlsx}")
    print(f"        status: {dict(df['status'].value_counts())}")
    if "grad_norm" in df:
        n = (df.grad_norm > GRAD_TOL).sum()
        print(f"        grad_norm > {GRAD_TOL:g}: {n} chất   (max {df.grad_norm.max():.2e})")
    if "E_spread_mH" in df:
        flag = df[df.E_spread_mH > SPREAD_WARN]
        if len(flag):
            print(f"        !! {len(flag)} chất có trạng thái cạnh tranh (E_spread > {SPREAD_WARN} mH):")
            for _, r in flag.sort_values("E_spread_mH", ascending=False).iterrows():
                print(f"           {r['molecule']:10} spread = {r['E_spread_mH']:.3f} mH")
        else:
            print(f"        không chất nào có E_spread > {SPREAD_WARN} mH")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="đường dẫn file JSON")
    ap.add_argument("--outdir", required=True, help="thư mục ghi kết quả")
    ap.add_argument("--index", type=int, default=None,
                    help="chạy 1 phân tử theo chỉ số (0-based)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="chỉ chạy các chất có tên này (giữ nguyên chỉ số gốc)")
    ap.add_argument("--max-memory", type=int,
                    default=int(os.environ.get("PYSCF_MAX_MEMORY", "30000")),
                    help="RAM cho PySCF, MB. ĐỂ THẤP HƠN --mem của SLURM ~20%%")
    ap.add_argument("--merge", action="store_true",
                    help="gộp các file row -> xlsx rồi thoát")
    ap.add_argument("--output-xlsx", default=None)
    args = ap.parse_args()

    output_xlsx = args.output_xlsx or os.path.join(args.outdir, "dipole_results.xlsx")

    if args.merge:                    # không cần pyscf
        do_merge(args.outdir, output_xlsx)
        return

    _lazy_imports()

    with open(args.input) as f:
        data = json.load(f)
    items = list(data.items())        # thứ tự ổn định để index khớp giữa các task

    # --only: chạy đúng các chất được nêu, GIỮ NGUYÊN chỉ số gốc để tên file
    # row không lệch với các lần chạy trước
    if args.only:
        todo = [(i, n, p) for i, (n, p) in enumerate(items) if n in set(args.only)]
        missing = set(args.only) - {n for _, n, _ in todo}
        if missing:
            print(f"[CẢNH BÁO] không có trong input: {sorted(missing)}")
        if not todo:
            return
    else:
        idx = args.index
        if idx is None and "SLURM_ARRAY_TASK_ID" in os.environ:
            idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if idx is None:
            todo = [(i, n, p) for i, (n, p) in enumerate(items)]
        elif 0 <= idx < len(items):
            todo = [(idx, items[idx][0], items[idx][1])]
        else:
            print(f"[SKIP] index {idx} ngoài phạm vi 0..{len(items)-1}")
            return

    for i, name, props in todo:
        print(f"[{i}] {name} ...", flush=True)
        row = run_one(name, props, args.max_memory)
        p = write_row(args.outdir, i, row)
        print(f"[{i}] {name}: {row['status']} | mu={row['dipole_debye']} D | "
              f"|g|={row['grad_norm']} | spread={row['E_spread_mH']} mH | "
              f"S2={row['S2']} | {row['walltime_s']}s -> {p}", flush=True)

    if len(todo) > 1 and not args.only and args.index is None:
        do_merge(args.outdir, output_xlsx)


if __name__ == "__main__":
    main()