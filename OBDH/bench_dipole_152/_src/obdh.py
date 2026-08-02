#!/usr/bin/env python3
"""
Quét alpha_c cho OBDH và chọn hệ số cho RMSE dipole nhỏ nhất (so với CCSD(T)).

Ý tưởng tối ưu:
  - Mỗi SLURM task xử lý ĐÚNG 1 phân tử (theo SLURM_ARRAY_TASK_ID / --index).
  - Trong 1 phân tử: chạy UHF + stabilize_scf MỘT LẦN, rồi lặp toàn bộ alpha_c
    trên cùng tham chiếu SCF đó (khôi phục orbital trước mỗi alpha để mọi alpha
    xuất phát y hệt) -> không lặp lại SCF 99 lần.
  - Ghi 1 file row/phân tử (mọi alpha) -> an toàn cho array, crash/hết giờ vẫn giữ.
  - Bước --merge: gộp tất cả, tính RMSE / MAD / RMSE-trim theo từng alpha_c,
    in ra alpha tốt nhất theo mỗi tiêu chí.

Tham chiếu (ref) dipole CCSD(T): lấy theo tên phân tử, từ
  (a) --ref file  (JSON {ten: gia_tri}  hoặc  CSV cột: molecule,ref), hoặc
  (b) trường "ref_dipole" trong chính input JSON.
Phân tử không có ref vẫn được tính dipole, chỉ bị loại khỏi thống kê RMSE.

Ví dụ:
  # 1 task / phân tử (array)
  python run_obdh_alpha_scan.py --input sp_inputs.json --outdir OUT --ref ref_sp.json
  # gộp + chọn alpha
  python run_obdh_alpha_scan.py --input sp_inputs.json --outdir OUT --ref ref_sp.json --merge
"""

import os, sys, csv, json, time, glob, argparse
import numpy as np
import pandas as pd
from pyscf import gto, scf

from pycmf.OBDH import OBDH_CL
from pycmf.OBDH.stability import stabilize_scf


# ---------------------------------------------------------------- ref
def load_ref(path):
    if not path:
        return {}
    if path.lower().endswith(".json"):
        with open(path) as f:
            return {str(k): float(v) for k, v in json.load(f).items() if v is not None}
    ref = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = r.get("molecule") or r.get("Molecule") or list(r.values())[0]
            val = r.get("ref") or r.get("ref_dipole") or r.get("CCSD(T)")
            try:
                ref[str(key)] = float(val)
            except (TypeError, ValueError):
                pass
    return ref


# ---------------------------------------------------------------- 1 molecule
def build_mol(name, data, max_memory):
    atom = [(sym, tuple(xyz)) for sym, xyz in data["geometry"]]
    mol = gto.Mole()
    mol.atom = atom
    mol.charge = data.get("charge", 0)
    mol.spin = data.get("spin", 0)
    mol.basis = {"default": "aug-cc-pcvqz", "H": "aug-cc-pvqz"}
    mol.max_memory = max_memory
    mol.verbose = 0
    mol.build()
    return mol


def scan_one(name, data, alphas, max_memory, ref_val):
    """Trả về list dict (một dòng / alpha)."""
    base = {"molecule": name, "charge": data.get("charge", 0), "spin": data.get("spin", 0),
            "ref": ref_val}
    rows = []
    t0 = time.time()
    try:
        mol = build_mol(name, data, max_memory)

        mf = scf.UHF(mol).density_fit(auxbasis="def2-universal-jkfit")
        mf.verbose = 0
        mf.kernel()
        mf = stabilize_scf(mf, max_macro_cycles=10, verbose=False)

        # lưu tham chiếu SCF để phục hồi trước mỗi alpha
        mo_c = tuple(c.copy() for c in mf.mo_coeff)
        mo_e = tuple(e.copy() for e in mf.mo_energy)
        mo_o = tuple(o.copy() for o in mf.mo_occ)
        e_uhf = float(mf.e_tot)

        for a in alphas:
            r = dict(base, alpha_c=round(float(a), 2), E_UHF=e_uhf,
                     dipole=None, error=None, obdh_energy=None,
                     converged=None, status="ok")
            try:
                # reset về đúng nghiệm UHF cho mọi alpha
                mf.mo_coeff = tuple(c.copy() for c in mo_c)
                mf.mo_energy = tuple(e.copy() for e in mo_e)
                mf.mo_occ = tuple(o.copy() for o in mo_o)

                calc = OBDH_CL(mf)
                calc.alphaa = (0.53, float(a))
                calc.thresh = 1e-8
                calc.second_order = True
                calc.mom_select = False
                calc.mom_start_cycle = 0
                calc.use_embed = False
                calc.use_cl = False
                calc.verbose = 0
                calc.run()

                r["obdh_energy"] = float(getattr(calc, "ene_tot", np.nan))
                r["converged"] = getattr(calc, "converged", None)
                dip = getattr(calc, "dip_mom", None)
                if dip is not None:
                    r["dipole"] = float(dip)
                    if ref_val is not None:
                        r["error"] = float(dip) - float(ref_val)
            except Exception as e:
                r["status"] = f"ERROR: {type(e).__name__}: {e}"
            rows.append(r)

    except Exception as e:  # hỏng ở SCF -> ghi 1 dòng lỗi cho toàn phân tử
        rows.append(dict(base, alpha_c=None, status=f"SCF_ERROR: {type(e).__name__}: {e}"))

    for r in rows:
        r["walltime_total_s"] = round(time.time() - t0, 2)
    return rows


def write_rows(outdir, idx, name, rows):
    d = os.path.join(outdir, "rows"); os.makedirs(d, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in name)
    p = os.path.join(d, f"{idx:03d}_{safe}.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------- merge / chọn alpha
def do_merge(outdir, out_prefix, trim=3):
    files = sorted(glob.glob(os.path.join(outdir, "rows", "*.csv")))
    if not files:
        print(f"[MERGE] không có row nào trong {outdir}/rows"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(out_prefix + "_all.csv", index=False)

    d = df.dropna(subset=["alpha_c", "error"]).copy()
    d["abserr"] = d["error"].abs()
    n_mol = d["molecule"].nunique()

    stats = []
    for a, g in d.groupby("alpha_c"):
        e = g["error"].to_numpy()
        ae = np.abs(e)
        # RMSE cắt bỏ 'trim' phân tử sai số lớn nhất
        keep = np.sort(ae)[:-trim] if (trim and len(ae) > trim) else ae
        stats.append({"alpha_c": round(a, 2), "N": len(e),
                      "MAD": ae.mean(), "RMSE": np.sqrt((e**2).mean()),
                      f"RMSE_trim{trim}": np.sqrt((keep**2).mean()),
                      "MAX": ae.max(), "bias": e.mean()})
    s = pd.DataFrame(stats).sort_values("alpha_c").reset_index(drop=True)
    s.to_csv(out_prefix + "_alpha_summary.csv", index=False)
    try:
        s.to_excel(out_prefix + "_alpha_summary.xlsx", index=False)
    except Exception:
        pass

    # chỉ so công bằng ở các alpha có đủ phân tử
    full = s[s["N"] == n_mol]
    pick = full if len(full) else s
    best_rmse = pick.loc[pick["RMSE"].idxmin()]
    best_mad  = pick.loc[pick["MAD"].idxmin()]
    best_trim = pick.loc[pick[f"RMSE_trim{trim}"].idxmin()]

    print(f"[MERGE] {n_mol} phân tử, {len(s)} mốc alpha_c -> {out_prefix}_alpha_summary.csv")
    print(f"  (chỉ xét alpha có đủ {n_mol} phân tử: {len(full)}/{len(s)} mốc)")
    print(f"  alpha* theo RMSE       : {best_rmse['alpha_c']:.2f}  (RMSE={best_rmse['RMSE']:.4f}, MAD={best_rmse['MAD']:.4f})")
    print(f"  alpha* theo MAD        : {best_mad['alpha_c']:.2f}  (MAD={best_mad['MAD']:.4f}, RMSE={best_mad['RMSE']:.4f})")
    print(f"  alpha* theo RMSE_trim{trim} : {best_trim['alpha_c']:.2f}  (trim={best_trim[f'RMSE_trim{trim}']:.4f})")
    n_bad = (df["status"] != "ok").sum()
    if n_bad:
        print(f"  [!] {n_bad} dòng lỗi/không hội tụ - xem cột status trong {out_prefix}_all.csv")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ref", default=None, help="JSON {ten:val} hoặc CSV molecule,ref")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--alpha-min", type=float, default=0.01)
    ap.add_argument("--alpha-max", type=float, default=0.99)
    ap.add_argument("--alpha-step", type=float, default=0.01)
    ap.add_argument("--max-memory", type=int,
                    default=int(os.environ.get("PYSCF_MAX_MEMORY", "28000")))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--trim", type=int, default=3, help="số outlier bỏ khi tính RMSE_trim")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    out_prefix = args.out_prefix or os.path.join(args.outdir, "obdh_dipole")

    if args.merge:
        do_merge(args.outdir, out_prefix, trim=args.trim)
        return

    with open(args.input) as f:
        data = json.load(f)
    items = list(data.items())
    ref = load_ref(args.ref)

    alphas = np.round(np.arange(args.alpha_min, args.alpha_max + 1e-9, args.alpha_step), 2)

    idx = args.index
    if idx is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

    def one(i):
        name, props = items[i]
        ref_val = ref.get(name, props.get("ref_dipole"))
        print(f"[{i}] {name} | {len(alphas)} alpha | ref={ref_val}", flush=True)
        rows = scan_one(name, props, alphas, args.max_memory, ref_val)
        p = write_rows(args.outdir, i, name, rows)
        ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"[{i}] {name}: {ok}/{len(rows)} alpha ok -> {p}", flush=True)

    if idx is not None:
        if 0 <= idx < len(items):
            one(idx)
        else:
            print(f"[SKIP] index {idx} ngoài 0..{len(items)-1}")
    else:
        for i in range(len(items)):
            one(i)
        do_merge(args.outdir, out_prefix, trim=args.trim)


if __name__ == "__main__":
    main()