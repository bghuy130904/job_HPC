#!/usr/bin/env python3
"""
Tính dipole moment ở các mức UHF / UMP2 / OBMP2 / OBDH cho bộ phân tử JSON.

MỌI phương pháp đều chạy trên tham chiếu UHF (unrestricted) — code OBDH
hiện chưa hỗ trợ RHF/ROHF. Đây là lựa chọn có chủ đích, không phải hạn chế
tạm: mục tiêu là chứng minh OBMP2/OBDH ở chế độ U KHÔNG sụp như UMP2.

Bối cảnh (Hait & Head-Gordon, JCTC 2018, 14, 1969, Table 1, tập SP):
    UMP2  54.63 %      <- sụp vì vi phạm N-representability
    RMP2   9.84 %      <- cứu được bằng orbital restricted
    PBE0   5.85 %      (unrestricted, rung 4)
    CCSD   4.80 %
Nếu OBMP2/OBDH ở chế độ U nằm gần RMP2 hơn UMP2 thì tối ưu orbital đã làm
được điều mà MP2 thường phải ép restricted mới có. Đó là lý do cột UMP2
được tính CÙNG một tham chiếu trong cùng một lần chạy — so sánh mới sạch.

Lớp quét nhiều initial guess giữ nguyên như bản CCSD, và nó BẮT BUỘC ở đây:
C2H cho thấy tối ưu orbital KHÔNG cứu được tham chiếu sai trạng thái
(OBMP2 3.269 D, OBDH 3.233 D so với tham chiếu 0.7601 D).

Ví dụ:
    python calc_dipmomt_obdh.py --input sp_inputs.json --outdir /data/.../obdh_sp
    python calc_dipmomt_obdh.py --input sp_inputs.json --outdir /data/.../obdh_sp \
           --only C2H PS --methods uhf ump2 obmp2 obdh
    python calc_dipmomt_obdh.py --input sp_inputs.json --outdir /data/.../obdh_sp --merge
"""

import os
import json
import time
import glob
import argparse

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
GRAD_TOL     = 1e-4
SPREAD_WARN  = 0.1                          # mHartree
INIT_GUESSES = ('minao', 'atom', '1e')      # 'huckel' bị loại: với CN nó chạy
                                            # >25 phút, các guess khác ~25 giây
BASIS        = {"default": "aug-cc-pcvqz", "H": "aug-cc-pvqz"}
ALPHAA       = (0.53, 0.39)                 # tham số OBDH, theo example.py
OBDH_THRESH  = 1e-8
OBDH_NITER   = 300


def _lazy_imports():
    global gto, scf, cc, mp, stabilize_scf, OBDH_CL, OBMP2_CL
    from pyscf import gto, scf, cc, mp                    # noqa: F401
    from pycmf.OBDH.stability import stabilize_scf        # noqa: F401
    from pycmf.OBDH import OBDH_CL, OBMP2_CL              # noqa: F401
    if not hasattr(np.linalg, 'linalg'):
        np.linalg.linalg = np.linalg


# ----------------------------------------------------------------------
def grad_norm(mf):
    dm = mf.make_rdm1()
    return float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock(dm=dm))))


def _one_scf(mol, guess):
    mf = scf.UHF(mol).density_fit(auxbasis="def2-universal-jkfit")
    mf.verbose = 0
    mf.init_guess = guess
    mf.max_cycle = 150
    mf.kernel()
    if not mf.converged:                      # DIIS bò chậm -> bậc hai
        mf = mf.newton()
        mf.kernel(mf.mo_coeff, mf.mo_occ)
    return stabilize_scf(mf, max_macro_cycles=10, verbose=False)


def run_scf_multiguess(mol):
    """
    Bốn lớp, bốn bài toán khác nhau:
      DIIS          -> tìm điểm dừng
      Newton        -> ép về điểm dừng khi DIIS kẹt (PS)
      stabilize_scf -> rời điểm yên ngựa sang cực tiểu (O3, hạ 88 mH)
      nhiều guess   -> thoát trạng thái điện tử sai (C2H, hạ 15 mH)
    stability() chỉ có nghĩa khi đã ở điểm dừng -> Newton phải chạy TRƯỚC.
    """
    results, best = [], None
    for guess in INIT_GUESSES:
        try:
            mf = _one_scf(mol, guess)
            g = grad_norm(mf)
            results.append({"guess": guess, "E": float(mf.e_tot), "grad": g})
            if g <= GRAD_TOL and (best is None or mf.e_tot < best.e_tot - 1e-8):
                best = mf
        except Exception as e:
            results.append({"guess": guess, "err": f"{type(e).__name__}: {e}"})

    if best is None:
        ok = [r for r in results if "E" in r]
        if not ok:
            raise RuntimeError("tất cả initial guess đều lỗi")
        best = _one_scf(mol, min(ok, key=lambda r: r["E"])["guess"])

    en = [r["E"] for r in results if "E" in r]
    spread = (max(en) - min(en)) * 1000.0 if len(en) > 1 else 0.0
    return best, spread, results


# ----------------------------------------------------------------------
def _dip_from_dm(mol, dm):
    """dm có thể là (dm_a, dm_b) hoặc mảng 2D. Trả về vector Debye."""
    v = scf.hf.dip_moment(mol, dm, unit='Debye', verbose=0)
    return np.asarray(v, dtype=float)


def _post_uhf(mol, mf):
    return _dip_from_dm(mol, mf.make_rdm1()), float(mf.e_tot), True


def _post_ump2(mol, mf):
    pt = mp.UMP2(mf)
    pt.verbose = 0
    pt.kernel()
    return _dip_from_dm(mol, pt.make_rdm1(ao_repr=True)), float(pt.e_tot), True


def _post_ob(mol, mf, hybrid):
    """OBDH (hybrid=True) hoặc OBMP2 (hybrid=False), đều trên tham chiếu UHF."""
    solver = OBDH_CL(mf) if hybrid else OBMP2_CL(mf)
    solver.verbose = 0
    solver.alphaa = ALPHAA
    solver.thresh = OBDH_THRESH
    solver.niter = OBDH_NITER
    solver.second_order = True
    solver.mom_select = False
    solver.use_embed = False
    solver.use_cl = False
    solver.run()
    gamma = solver._gamma
    dip = _dip_from_dm(mol, (gamma[0], gamma[1]))
    conv = bool(solver.converged) if solver.converged is not None else None
    return dip, float(solver.ene_tot), conv


METHODS = {
    "uhf":   ("UHF",   lambda mol, mf: _post_uhf(mol, mf)),
    "ump2":  ("UMP2",  lambda mol, mf: _post_ump2(mol, mf)),
    "obmp2": ("OBMP2", lambda mol, mf: _post_ob(mol, mf, hybrid=False)),
    "obdh":  ("OBDH",  lambda mol, mf: _post_ob(mol, mf, hybrid=True)),
}


# ----------------------------------------------------------------------
def run_one(name, props, max_memory, methods):
    t0 = time.time()
    row = {"molecule": name, "charge": props.get("charge"), "spin": props.get("spin"),
           "nao": None, "grad_norm": None, "E_spread_mH": None, "n_guess_ok": None,
           "S2": None, "status": "ok"}
    for key in methods:
        tag = METHODS[key][0]
        row.update({f"E_{tag}": None, f"mu_{tag}": None,
                    f"mux_{tag}": None, f"muy_{tag}": None, f"muz_{tag}": None,
                    f"conv_{tag}": None, f"t_{tag}": None})
    try:
        mol = gto.Mole()
        mol.atom = [(a[0], tuple(a[1])) for a in props["geometry"]]
        mol.charge = props["charge"]
        mol.spin = props["spin"]
        mol.basis = BASIS
        mol.max_memory = max_memory
        mol.verbose = 0
        mol.build()
        row["nao"] = int(mol.nao)

        mf, spread, res = run_scf_multiguess(mol)
        row["E_spread_mH"] = round(float(spread), 6)
        row["n_guess_ok"] = sum(1 for r in res if "E" in r)
        row["grad_norm"] = grad_norm(mf)
        try:
            row["S2"] = float(mf.spin_square()[0])
        except Exception:
            pass

        if row["grad_norm"] > GRAD_TOL:
            row["status"] = "SCF_not_converged"
        elif spread > SPREAD_WARN:
            row["status"] = f"WARN_multi_state(spread={spread:.2f}mH)"

        for key in methods:
            tag, fn = METHODS[key]
            t1 = time.time()
            try:
                dip, e, conv = fn(mol, mf)
                row[f"E_{tag}"] = e
                row[f"mu_{tag}"] = float(np.linalg.norm(dip))
                row[f"mux_{tag}"], row[f"muy_{tag}"], row[f"muz_{tag}"] = map(float, dip)
                row[f"conv_{tag}"] = conv
            except Exception as e:
                row[f"conv_{tag}"] = f"ERR: {type(e).__name__}: {e}"
            row[f"t_{tag}"] = round(time.time() - t1, 1)

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


def do_merge(outdir, out_xlsx, ref_json=None):
    files = sorted(glob.glob(os.path.join(outdir, "rows", "*.csv")))
    if not files:
        print(f"[MERGE] không có file row nào trong {outdir}/rows")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    os.makedirs(os.path.dirname(out_xlsx) or ".", exist_ok=True)
    df.to_csv(os.path.splitext(out_xlsx)[0] + ".csv", index=False)
    try:
        df.to_excel(out_xlsx, index=False)
    except ImportError:
        print("[MERGE] thiếu openpyxl -> chỉ ghi CSV")

    print(f"[MERGE] {len(df)} phân tử -> {out_xlsx}")
    print(f"        status: {dict(df['status'].value_counts())}")
    if "grad_norm" in df:
        print(f"        grad_norm > {GRAD_TOL:g}: {(df.grad_norm > GRAD_TOL).sum()} chất")
    if "E_spread_mH" in df:
        fl = df[df.E_spread_mH > SPREAD_WARN]
        if len(fl):
            print(f"        !! {len(fl)} chất có trạng thái cạnh tranh:")
            for _, r in fl.sort_values("E_spread_mH", ascending=False).iterrows():
                print(f"           {r['molecule']:10} spread = {r['E_spread_mH']:.3f} mH")
        else:
            print("        không chất nào có trạng thái cạnh tranh")

    # thống kê so với tham chiếu, nếu có
    if ref_json and os.path.exists(ref_json):
        ref = json.load(open(ref_json))
        df["ref"] = df.molecule.map(ref)
        sub = df[df.ref.notna()]
        print(f"\n        RMSE regularized so voi {len(sub)} gia tri tham chieu:")
        for tag in ["UHF", "UMP2", "OBMP2", "OBDH"]:
            col = f"mu_{tag}"
            if col not in sub:
                continue
            s = sub[sub[col].notna()]
            err = 100 * (s[col] - s.ref) / np.maximum(s.ref, 1.0)
            print(f"          {tag:6} n={len(s):3}  RMSE = {np.sqrt((err**2).mean()):7.2f} %"
                  f"   ME = {err.mean():+7.2f} %   MAX = {err.abs().max():7.2f} %"
                  f"  ({s.loc[err.abs().idxmax(), 'molecule'] if len(s) else '-'})")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=["uhf", "ump2", "obmp2", "obdh"],
                    choices=list(METHODS))
    ap.add_argument("--max-memory", type=int,
                    default=int(os.environ.get("PYSCF_MAX_MEMORY", "30000")))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--output-xlsx", default=None)
    ap.add_argument("--ref-json", default=None,
                    help="file JSON {tên chất: dipole CCSD(T)} để tính RMSE khi merge")
    args = ap.parse_args()

    out_xlsx = args.output_xlsx or os.path.join(args.outdir, "dipole_obdh.xlsx")

    if args.merge:                       # không cần pyscf/pycmf
        do_merge(args.outdir, out_xlsx, args.ref_json)
        return

    _lazy_imports()

    items = list(json.load(open(args.input)).items())

    if args.only:
        todo = [(i, n, p) for i, (n, p) in enumerate(items) if n in set(args.only)]
        miss = set(args.only) - {n for _, n, _ in todo}
        if miss:
            print(f"[CẢNH BÁO] không có trong input: {sorted(miss)}")
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
        row = run_one(name, props, args.max_memory, args.methods)
        p = write_row(args.outdir, i, row)
        mus = "  ".join(f"{METHODS[k][0]}={row.get('mu_' + METHODS[k][0])}"
                        for k in args.methods)
        print(f"[{i}] {name}: {row['status']} | |g|={row['grad_norm']} | "
              f"spread={row['E_spread_mH']} mH | {mus} | {row['walltime_s']}s -> {p}",
              flush=True)

    if len(todo) > 1 and not args.only and args.index is None:
        do_merge(args.outdir, out_xlsx, args.ref_json)


if __name__ == "__main__":
    main()