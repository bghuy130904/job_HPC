#!/usr/bin/env python3
"""
Dipole moment ở mức DFT: PBE0, B3LYP (rung 4) và B2PLYP (rung 5, double hybrid).

Cùng lớp quét nhiều initial guess như bản CCSD/OBDH. Lý do vẫn cần dù DFT ít
sụp hơn HF:
  - Bằng chứng "DFT tránh được trạng thái sai" hiện chỉ đến từ MỘT chất (C2H,
    nơi aufbau UHF chọn 2-Pi còn aufbau KS chọn 2-Sigma+ đúng). Một điểm dữ
    liệu không đủ để bỏ lớp kiểm tra.
  - B2PLYP CHỨA số hạng MP2 nên thừa hưởng đúng độ nhạy tham chiếu đã làm
    hỏng OBMP2/OBDH ở C2H. Đây là chất rủi ro cao nhất trong ba, không phải
    thấp nhất.
  - E_spread = 0 trên cả 152 chất là bằng chứng DƯƠNG để viết vào bài.

CÁCH LẤY DIPOLE (khác nhau giữa hai nhóm, có chủ đích):
  PBE0 / B3LYP : giải tích. Mật độ KS CHÍNH LÀ mật độ của phương pháp, nên
                 mf.dip_moment() là chính xác, không cần finite field.
  B2PLYP       : finite field, sai phân trung tâm F = 1e-4 a.u. — đúng như
                 bài báo làm cho mọi double hybrid. Không dùng được mật độ KS
                 vì còn số hạng MP2. Chi phí: 6 lần (UKS + MP2) cho mỗi chất.

LƯU Ý VỀ ORBITAL: mọi thứ chạy UNRESTRICTED (UKS), khớp với bài báo cho rung
1-4. Bài báo dùng orbital RESTRICTED cho double hybrid, nên số B2PLYP ở đây
KHÔNG so trực tiếp được với cột B2PLYP trong Table 1 (5.31% tổng / 7.08% SP).
Nó là số để so với OBDH unrestricted của bạn — cùng điều kiện.

Ví dụ:
    python calc_dipmomt_dft.py --input sp_inputs.json --outdir /data/.../dft_sp
    python calc_dipmomt_dft.py --input sp_inputs.json --outdir /data/.../dft_sp \
           --only C2H PS --funcs pbe0 b3lyp
    python calc_dipmomt_dft.py --outdir /data/.../dft_sp --merge \
           --ref-json ref_ccsdt.json
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
INIT_GUESSES = ('minao', 'atom', '1e')      # 'huckel' bị loại: quá chậm ở CN
BASIS        = {"default": "aug-cc-pcvqz", "H": "aug-cc-pvqz"}
FIELD        = 1e-4                         # a.u., giống bài báo
AU2DEBYE     = 2.541746
CURV_WARN    = 1e-5      # |E+ + E- - 2E0| lon hon -> mot diem truong nhay nghiem

# B2PLYP: 53% HF + 47% B88 exchange ; 73% LYP + 27% PT2 correlation
B2PLYP_XC   = '0.53*HF + 0.47*B88, 0.73*LYP'
B2PLYP_CMP2 = 0.27

FUNCS = {
    "pbe0":   {"xc": "PBE0",     "kind": "hybrid"},
    "b3lyp":  {"xc": "B3LYP",    "kind": "hybrid"},
    "b2plyp": {"xc": B2PLYP_XC,  "kind": "double", "cmp2": B2PLYP_CMP2},
}


def _lazy_imports():
    global gto, dft, mp, scf, stabilize_scf
    from pyscf import gto, dft, mp, scf                   # noqa: F401
    from pycmf.OBDH.stability import stabilize_scf        # noqa: F401
    if not hasattr(np.linalg, 'linalg'):
        np.linalg.linalg = np.linalg


# ----------------------------------------------------------------------
def grad_norm(mf):
    dm = mf.make_rdm1()
    return float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock(dm=dm))))


def _build_uks(mol, xc, hcore=None):
    mf = dft.UKS(mol)
    mf.xc = xc
    mf = mf.density_fit()
    mf.verbose = 0
    mf.max_cycle = 200
    if hcore is not None:                     # cho finite field
        mf.get_hcore = lambda *a, **k: hcore
    return mf


def _scf_once(mol, xc, guess, hcore=None):
    mf = _build_uks(mol, xc, hcore)
    mf.init_guess = guess
    mf.kernel()
    if not mf.converged:                      # DIIS kẹt -> bậc hai
        mf = mf.newton()
        mf.kernel(mf.mo_coeff, mf.mo_occ)
    return stabilize_scf(mf, max_macro_cycles=10, verbose=False)


def scf_multiguess(mol, xc):
    """
    Bốn lớp, bốn bài toán khác nhau:
      DIIS          -> tìm điểm dừng
      Newton        -> ép về điểm dừng khi DIIS kẹt
      stabilize_scf -> rời điểm yên ngựa sang cực tiểu thật
      nhiều guess   -> thoát khỏi TRẠNG THÁI ĐIỆN TỬ sai
    stability() chỉ có nghĩa khi đã ở điểm dừng -> Newton chạy TRƯỚC.
    """
    res, best = [], None
    for g in INIT_GUESSES:
        try:
            mf = _scf_once(mol, xc, g)
            gn = grad_norm(mf)
            res.append({"guess": g, "E": float(mf.e_tot), "grad": gn})
            if gn <= GRAD_TOL and (best is None or mf.e_tot < best.e_tot - 1e-8):
                best = mf
        except Exception as e:
            res.append({"guess": g, "err": f"{type(e).__name__}: {e}"})
    if best is None:
        ok = [r for r in res if "E" in r]
        if not ok:
            raise RuntimeError("tất cả initial guess đều lỗi")
        best = _scf_once(mol, xc, min(ok, key=lambda r: r["E"])["guess"])
    en = [r["E"] for r in res if "E" in r]
    spread = (max(en) - min(en)) * 1000.0 if len(en) > 1 else 0.0
    return best, spread, res


# ----------------------------------------------------------------------
def _b2plyp_energy(mol, xc, cmp2, hcore, mo_ref=None):
    """E(B2PLYP) = E_UKS[xc] + cmp2 * E_corr(MP2 trên chính orbital KS đó)."""
    mf = _build_uks(mol, xc, hcore)
    if mo_ref is not None:                    # bám theo nghiệm đã chọn
        mf.kernel(dm0=mo_ref)
    else:
        mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.mo_coeff, mf.mo_occ)
    pt = mp.UMP2(mf)
    pt.verbose = 0
    pt.kernel()
    return float(mf.e_tot + cmp2 * pt.e_corr)


def dipole_finite_field(mol, xc, cmp2, dm0=None, e0=None):
    """mu = mu_nuc - dE/dF, sai phân trung tâm — đúng cách bài báo làm.

    Trả về (vector Debye, độ cong lớn nhất). Độ cong |E+ + E- - 2E0| ~ F^2 *
    độ phân cực (~1e-8 Eh). Nếu nó lớn thì một điểm trường đã rơi vào nghiệm
    KS khác -> số liệu chất đó không dùng được. B2PLYP là phương pháp DUY NHẤT
    ở đây dùng finite field, và số hạng MP2 khuếch đại tham chiếu sai (xem C2H),
    nên chốt kiểm tra này cần thiết.
    """
    mf0 = _build_uks(mol, xc)
    h0 = mf0.get_hcore()
    with mol.with_common_orig((0, 0, 0)):
        r = mol.intor('int1e_r', comp=3)
    d = np.zeros(3)
    curv = 0.0
    for x in range(3):
        f = np.zeros(3)
        f[x] = FIELD
        ep = _b2plyp_energy(mol, xc, cmp2, h0 + np.einsum('i,iuv->uv', f, r), dm0)
        em = _b2plyp_energy(mol, xc, cmp2, h0 - np.einsum('i,iuv->uv', f, r), dm0)
        d[x] = (ep - em) / (2 * FIELD)
        if e0 is not None:
            curv = max(curv, abs(ep + em - 2 * e0))
    nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    return (nuc - d) * AU2DEBYE, curv


# ----------------------------------------------------------------------
def run_one(name, props, max_memory, funcs, mode='dm'):
    t0 = time.time()
    row = {"molecule": name, "charge": props.get("charge"), "spin": props.get("spin"),
           "nao": None, "status": "ok"}
    for k in funcs:
        F = k.upper()
        row.update({f"E_{F}": None,
                    # hai dai luong, hai cot. KHONG con cot "mu_{F}" gop chung:
                    # truoc day no chua mat do hay finite field tuy theo --dipole,
                    # va hang du lieu khong ghi lai la cai nao.
                    f"mu_dm_{F}": None,        # mat do (giai tich voi PBE0/B3LYP)
                    f"mu_ff_{F}": None,        # -dE/dF, chi B2PLYP moi can
                    f"diff_{F}": None,         # % lech dm vs ff
                    f"mux_dm_{F}": None, f"muy_dm_{F}": None, f"muz_dm_{F}": None,
                    f"mux_ff_{F}": None, f"muy_ff_{F}": None, f"muz_ff_{F}": None,
                    f"curv_{F}": None,
                    f"grad_{F}": None, f"spread_{F}": None, f"S2_{F}": None,
                    f"t_{F}": None})
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

        flags = []
        for k in funcs:
            F, cfg = k.upper(), FUNCS[k]
            t1 = time.time()
            try:
                mf, spread, _ = scf_multiguess(mol, cfg["xc"])
                gn = grad_norm(mf)
                row[f"grad_{F}"] = gn
                row[f"spread_{F}"] = round(float(spread), 6)
                try:
                    row[f"S2_{F}"] = float(mf.spin_square()[0])
                except Exception:
                    pass

                if cfg["kind"] == "hybrid":
                    # Mat do KS CHINH LA mat do cua phuong phap (Hohenberg-Kohn),
                    # nen Hellmann-Feynman ap dung: giai tich la chinh xac, va
                    # mu_ff = mu_dm theo dung dinh nghia, khong can chay 6 diem.
                    dip = np.asarray(mf.dip_moment(unit='Debye', verbose=0), float)
                    row[f"E_{F}"] = float(mf.e_tot)
                    row[f"mu_dm_{F}"] = float(np.linalg.norm(dip))
                    row[f"mux_dm_{F}"], row[f"muy_dm_{F}"], row[f"muz_dm_{F}"] = \
                        map(float, dip)
                    row[f"mu_ff_{F}"] = row[f"mu_dm_{F}"]
                    row[f"mux_ff_{F}"], row[f"muy_ff_{F}"], row[f"muz_ff_{F}"] = \
                        map(float, dip)
                else:
                    # Double hybrid co so hang MP2:
                    #   dm = rho_KS + c*(d_oo + d_vv) -> mat do unrelaxed, re,
                    #        cung dinh nghia ma Tran (PCCP 2022) dung cho OBMP2.
                    #        Viet duoi dang (1-c)*rho_KS + c*pt.make_rdm1() la
                    #        tuong duong, vi phan tham chieu cua pt.make_rdm1()
                    #        CHINH LA rho_KS (dinh thuc tren orbital KS).
                    #   ff = sai phan huu han -> cach cua Hait & Head-Gordon
                    # Chenh lech do o cc-pVDZ: C2H -0.7 %, CH +1.9 %, HCO +12.4 %,
                    # O3 +13.4 %, CN +47.8 %.
                    dm0 = mf.make_rdm1()
                    pt = mp.UMP2(mf); pt.verbose = 0; pt.kernel()
                    c = cfg["cmp2"]
                    e0 = float(mf.e_tot + c * pt.e_corr)
                    row[f"E_{F}"] = e0

                    dmp = pt.make_rdm1(ao_repr=True)
                    dm_mix = tuple((1.0 - c) * np.asarray(dm0[i]) + c * np.asarray(dmp[i])
                                   for i in (0, 1))
                    dip_dm = np.asarray(scf.hf.dip_moment(mol, dm_mix, unit='Debye',
                                                          verbose=0), float)
                    row[f"mu_dm_{F}"] = float(np.linalg.norm(dip_dm))
                    row[f"mux_dm_{F}"], row[f"muy_dm_{F}"], row[f"muz_dm_{F}"] = \
                        map(float, dip_dm)

                    if mode in ("ff", "both"):
                        dip_ff, curv = dipole_finite_field(mol, cfg["xc"], c, dm0, e0)
                        row[f"curv_{F}"] = float(curv)
                        if curv > CURV_WARN:
                            flags.append(f"{F}:FF_jump({curv:.1e})")
                        nff = float(np.linalg.norm(dip_ff))
                        row[f"mu_ff_{F}"] = nff
                        row[f"mux_ff_{F}"], row[f"muy_ff_{F}"], row[f"muz_ff_{F}"] = \
                            map(float, dip_ff)
                        if nff > 1e-8:
                            row[f"diff_{F}"] = 100.0 * (row[f"mu_dm_{F}"] - nff) / nff

                if gn > GRAD_TOL:
                    flags.append(f"{F}:SCF")
                elif spread > SPREAD_WARN:
                    flags.append(f"{F}:spread={spread:.2f}mH")
            except Exception as e:
                flags.append(f"{F}:ERR({type(e).__name__})")
            row[f"t_{F}"] = round(time.time() - t1, 1)

        if flags:
            row["status"] = "WARN " + " ".join(flags)

    except Exception as e:
        row["status"] = f"ERROR: {type(e).__name__}: {e}"

    row["walltime_s"] = round(time.time() - t0, 2)
    return row


# ----------------------------------------------------------------------
def _fmt_dip(row, F):
    """Chuoi tom tat cho MOT phiem ham: chi liet ke dai luong da thuc su tinh
    duoc. Phiem ham khong chay (hoac chay loi) tra ve chuoi rong -> khong xuat
    hien trong dong tom tat."""
    parts = [f"{lab}={row[c]:.4f}"
             for c, lab in ((f"mu_dm_{F}", "dm"), (f"mu_ff_{F}", "ff"))
             if row.get(c) is not None and np.isfinite(row[c])]
    return f"{F}[{' '.join(parts)}]" if parts else ""


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
    bad = df[df.status != 'ok']
    print(f"        chất có cảnh báo: {len(bad)}")
    for _, r in bad.iterrows():
        print(f"          {r['molecule']:10} {r['status']}")

    for F in ["PBE0", "B3LYP", "B2PLYP"]:
        c = f"spread_{F}"
        if c in df:
            fl = df[df[c] > SPREAD_WARN]
            print(f"        {F:7} spread>{SPREAD_WARN} mH: {len(fl)} chất"
                  + (f" -> {list(fl.molecule)}" if len(fl) else "")
                  + f"   (max {df[c].max():.3f} mH)")

    if ref_json and os.path.exists(ref_json):
        ref = json.load(open(ref_json))
        df["ref"] = df.molecule.map(ref)
        sub = df[df.ref.notna()]
        print(f"\n        RMSE regularized so với {len(sub)} giá trị CCSD(T)/CBS:")
        for pre, lab in (("mu_dm", "mat do"), ("mu_ff", "finite field (-dE/dF)")):
            print(f"\n          --- {lab} ---")
            for F in ["PBE0", "B3LYP", "B2PLYP"]:
                c = f"{pre}_{F}"
                if c not in sub:
                    continue
                t = sub[sub[c].notna()]
                if not len(t):
                    continue
                err = 100 * (t[c] - t.ref) / np.maximum(t.ref, 1.0)
                print(f"          {F:7} n={len(t):3}  RMSE = {np.sqrt((err**2).mean()):7.2f} %"
                      f"   ME = {err.mean():+7.2f} %"
                      f"   MAX = {err.abs().max():7.2f} % ({t.loc[err.abs().idxmax(),'molecule']})")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="file JSON hình học (không cần khi --merge)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--funcs", nargs="+", default=["pbe0", "b3lyp", "b2plyp"],
                    choices=list(FUNCS))
    ap.add_argument("--dipole", choices=["dm", "ff", "both"], default="dm",
                    help="CHI anh huong toi B2PLYP. dm (mac dinh) = chi mat do; "
                         "ff / both = chay them finite field (6 lan UKS+MP2 moi "
                         "chat). PBE0/B3LYP bien phan nen mat do CHINH LA -dE/dF: "
                         "mu_dm va mu_ff cua chung luon bang nhau, khong ton them "
                         "gi. Cot nao khong duoc yeu cau se de trong, KHONG bi dien "
                         "bang gia tri cua cot khac.")
    ap.add_argument("--max-memory", type=int,
                    default=int(os.environ.get("PYSCF_MAX_MEMORY", "30000")))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--output-xlsx", default=None)
    ap.add_argument("--ref-json", default=None)
    args = ap.parse_args()

    out_xlsx = args.output_xlsx or os.path.join(args.outdir, "dipole_dft.xlsx")

    if args.merge:                            # không cần pyscf, không cần --input
        do_merge(args.outdir, out_xlsx, args.ref_json)
        return

    if not args.input:
        ap.error("--input là bắt buộc khi không dùng --merge")

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
        row = run_one(name, props, args.max_memory, args.funcs, args.dipole)
        p = write_row(args.outdir, i, row)
        mus = "  ".join(x for x in (_fmt_dip(row, k.upper()) for k in args.funcs) if x)
        print(f"[{i}] {name}: {row['status']} | {mus} | {row['walltime_s']}s -> {p}",
              flush=True)

    if len(todo) > 1 and not args.only and args.index is None:
        do_merge(args.outdir, out_xlsx, args.ref_json)


if __name__ == "__main__":
    main()