#!/usr/bin/env python3
"""
Dipole moment mức (U)CCSD bằng FINITE FIELD — đúng cách Hait & Head-Gordon
(JCTC 2018, 14, 1969) tính cho MP2/CCSD/CCSD(T): sai phân trung tâm với điện
trường F = 1e-4 a.u. Bài báo chỉ lấy dipole giải tích cho HF và DFT rung 1-4,
vì hai thứ đó biến phân; mọi phương pháp tương quan và double hybrid đều dùng
finite field.

VÌ SAO
    Năng lượng CCSD KHÔNG dừng đối với phép quay orbital, nên Hellmann-Feynman
    không áp dụng và mật độ không cho dipole đúng. make_rdm1() của pyscf bỏ hẳn
    khối occupied-virtual của orbital response. Đo ở cc-pVDZ, chênh lệch giữa
    mật độ unrelaxed và finite field:
        CH +0.37%   C2H +0.55%   O3 -3.93%   HCO -4.54%   CN -6.34%
    Nhỏ hơn MP2 rất nhiều (O3 tới +65%) vì amplitude T1 của CCSD chính là một
    phép quay orbital nên tự hấp thụ phần lớn hồi phục. Nhưng vẫn cùng cỡ với
    sai số CCSD-vs-CCSD(T) (4.80% trên tập SP), nên không bỏ qua được nếu muốn
    đặt cạnh cột CCSD của Table 1.

    Cột mu_dm (mật độ unrelaxed) vẫn được ghi để bạn đo trực tiếp chênh lệch
    trên chính bộ dữ liệu của mình.

CHI PHÍ: 7 lần CCSD mỗi chất (1 trường 0 + 6 trường). Tập SP trước đây mất
    59.9 giờ CPU -> ước tính ~420 giờ. Dùng --dipole dm để quay lại cách cũ.

CÁC LỚP SCF (giữ nguyên): DIIS -> Newton -> stabilize_scf -> nhiều guess.
    Các điểm trường DÙNG LẠI nghiệm trường 0 làm guess -> không nhảy trạng
    thái, và nhanh hơn nhiều.

Ví dụ:
    python calc_dipmomt_ccsd.py --input sp_inputs.json --outdir /data/.../sp
    python calc_dipmomt_ccsd.py --input sp_inputs.json --outdir /data/.../sp \
           --only C2H PS O3 CN HCO
    python calc_dipmomt_ccsd.py --outdir /data/.../sp --merge --ref-json ref_ccsdt.json
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
INIT_GUESSES = ('minao', 'atom', '1e')      # 'huckel' bị loại: >25 phút ở CN
BASIS        = {"default": "aug-cc-pcvqz", "H": "aug-cc-pvqz"}
FIELD        = 1e-4                         # a.u., giống bài báo
AU2DEBYE     = 2.541746
CURV_WARN    = 1e-5                         # Hartree; |E+ + E- - 2E0| lớn hơn
                                            # mức này -> một điểm trường đã
                                            # nhảy sang trạng thái khác


def _lazy_imports():
    global gto, scf, cc, stabilize_scf
    from pyscf import gto, scf, cc                        # noqa: F401
    from pycmf.OBDH.stability import stabilize_scf        # noqa: F401
    if not hasattr(np.linalg, 'linalg'):
        np.linalg.linalg = np.linalg


# ----------------------------------------------------------------------
def _nuc_dip(mol):
    return np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())


def _dip_from_dm(mol, dm):
    return np.asarray(scf.hf.dip_moment(mol, dm, unit='Debye', verbose=0), float)


def symmetry_flag(mol, dip, tol=1e-3):
    """Truc nao co mu_nuc = 0 thi mu tong cung phai = 0 (doi xung diem).

    Dieu kien CHUNG cho moi chat, khong phai ban va rieng. Vi pham = nghiem SCF
    pha doi xung KHONG GIAN. CHI GHI LAI, khong doi nghiem, khong doi so lieu.
    """
    nuc = _nuc_dip(mol) * AU2DEBYE
    return "".join(ax for ax, (n, m) in zip("xyz", zip(nuc, np.asarray(dip, float)))
                   if abs(n) < tol and abs(m) > tol)


def ccsd_density(mol, mycc):
    """Mat do CCSD tu mycc.make_rdm1() -- dung dai luong Tran (PCCP 2022) dung
    lam moc, va ho ghi ro "its density matrix is not relaxed".

    Luu y ve phan loai: khoi ov cua no KHAC 0, den tu bien do don
    (pyscf uccsd_rdm: dova = l1a), khong phai tu Z-vector. Nen no khong nam
    cung o voi cot 'unrel' cua MP2/OBMP2 (o do dov = 0) -- co ov nhung van
    chua phai relaxed. Vi vay giu ten rieng la mu_dm.

    Tra ve (dipole Debye, (occ_max, occ_min, n_bad)).
    """
    d_mo = mycc.make_rdm1()                       # co so MO
    C = mycc.mo_coeff
    dip = _dip_from_dm(mol, tuple(C[i] @ d_mo[i] @ C[i].conj().T for i in (0, 1)))
    o = np.concatenate([np.linalg.eigvalsh(d) for d in d_mo])
    return dip, (float(o.max()), float(o.min()),
                 int(((o < -1e-6) | (o > 1 + 1e-6)).sum()))


def grad_norm(mf):
    dm = mf.make_rdm1()
    return float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock(dm=dm))))


def _mk_uhf(mol, hcore=None):
    mf = scf.UHF(mol).density_fit(auxbasis="def2-universal-jkfit")
    mf.verbose = 0
    mf.max_cycle = 150
    if hcore is not None:
        mf.get_hcore = lambda *a, **k: hcore
    return mf


def _scf_once(mol, guess=None, hcore=None, dm0=None):
    mf = _mk_uhf(mol, hcore)
    if guess is not None:
        mf.init_guess = guess
    mf.kernel(dm0=dm0)
    if not mf.converged:                    # DIIS bò chậm -> bậc hai
        mf = mf.newton()
        mf.kernel(mf.mo_coeff, mf.mo_occ)
    return mf


def run_scf_multiguess(mol):
    """
    Bốn lớp, bốn bài toán KHÁC nhau, không thay thế nhau được:
      DIIS          -> tìm điểm dừng
      Newton        -> ép về điểm dừng khi DIIS kẹt (PS: |g| 3.6e-3 -> 8e-6)
      stabilize_scf -> rời điểm yên ngựa sang cực tiểu thật (O3: hạ 88 mH)
      nhiều guess   -> thoát TRẠNG THÁI ĐIỆN TỬ sai (C2H: hạ 15 mH)
    stability() chỉ có nghĩa khi đã ở điểm dừng -> Newton chạy TRƯỚC.
    """
    res, best = [], None
    for g in INIT_GUESSES:
        try:
            mf = stabilize_scf(_scf_once(mol, guess=g), max_macro_cycles=10, verbose=False)
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
        best = stabilize_scf(_scf_once(mol, guess=min(ok, key=lambda r: r["E"])["guess"]),
                             max_macro_cycles=10, verbose=False)
    en = [r["E"] for r in res if "E" in r]
    spread = (max(en) - min(en)) * 1000.0 if len(en) > 1 else 0.0
    return best, spread, res


# ----------------------------------------------------------------------
def _ccsd_in_field(mol, hcore, dm0):
    """E_CCSD với h1 = hcore0 + F.r. Dùng dm0 của trường 0 làm guess để KHÔNG
    nhảy sang trạng thái điện tử khác."""
    mf = _scf_once(mol, hcore=hcore, dm0=dm0)
    g = grad_norm(mf)
    mycc = cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    return float(mycc.e_tot), g, bool(mycc.converged)


def dipole_finite_field(mol, mf0, e0):
    """
    mu = mu_nuc - dE/dF, sai phân trung tâm.
    Trả về (vector Debye, max|grad| các điểm trường, max độ cong, số lần CCSD
    không hội tụ).
    Độ cong |E+ + E- - 2E0| ~ F^2 * polarizability nên rất nhỏ. Nếu nó lớn thì
    một điểm trường đã nhảy sang trạng thái khác -> kết quả không dùng được.
    """
    h0 = _mk_uhf(mol).get_hcore()
    with mol.with_common_orig((0, 0, 0)):
        r = mol.intor('int1e_r', comp=3)
    dm0 = mf0.make_rdm1()

    d = np.zeros(3)
    gmax, curv, nbad = 0.0, 0.0, 0
    for x in range(3):
        f = np.zeros(3)
        f[x] = FIELD
        pert = np.einsum('i,iuv->uv', f, r)
        ep, gp, cp = _ccsd_in_field(mol, h0 + pert, dm0)
        em, gm, cm = _ccsd_in_field(mol, h0 - pert, dm0)
        d[x] = (ep - em) / (2 * FIELD)
        gmax = max(gmax, gp, gm)
        curv = max(curv, abs(ep + em - 2 * e0))
        nbad += (not cp) + (not cm)

    nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    return (nuc - d) * AU2DEBYE, gmax, curv, nbad


# ----------------------------------------------------------------------
def run_one(name, props, max_memory, mode):
    t0 = time.time()
    row = {"molecule": name, "charge": props.get("charge"), "spin": props.get("spin"),
           "nao": None, "converged": None, "grad_norm": None, "E_spread_mH": None,
           "n_guess_ok": None, "S2": None, "sym_break": None,
           "E_UHF": None, "E_UCCSD": None,
           "mu_ff_CCSD": None, "mux_ff": None, "muy_ff": None, "muz_ff": None,
           "mu_dm_CCSD": None, "d_dm_ff_CCSD": None,
           "nocc_max_CCSD": None, "nocc_min_CCSD": None, "n_bad_CCSD": None,
           "ff_grad_max": None, "ff_curv": None, "ff_ccsd_bad": None,
           "walltime_s": None, "status": "ok"}
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
        row["E_UHF"] = float(mf.e_tot)
        row["converged"] = bool(mf.converged)
        row["grad_norm"] = grad_norm(mf)
        try:
            row["S2"] = float(mf.spin_square()[0])
        except Exception:
            pass

        try:
            row["sym_break"] = symmetry_flag(mol, _dip_from_dm(mol, mf.make_rdm1()))
        except Exception:
            pass

        flags = []
        if row["sym_break"]:
            flags.append(f"sym_break({row['sym_break']})")
        if row["grad_norm"] > GRAD_TOL:
            flags.append("SCF_not_converged")
        elif spread > SPREAD_WARN:
            flags.append(f"multi_state(spread={spread:.2f}mH)")

        # trường 0: lấy E_CCSD và (nếu cần) mật độ unrelaxed
        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        mycc.kernel()
        row["E_UCCSD"] = float(mycc.e_tot)
        if not mycc.converged:
            flags.append("CCSD_not_converged")

        # mat do LUON duoc tinh (mien phi: dung lai Lambda da giai)
        dip_dm, occ = ccsd_density(mol, mycc)
        row["mu_dm_CCSD"] = float(np.linalg.norm(dip_dm))
        row["nocc_max_CCSD"], row["nocc_min_CCSD"], row["n_bad_CCSD"] = occ
        if row["n_bad_CCSD"]:
            flags.append(f"N_repr({row['n_bad_CCSD']})")
        print(f"    CCSD   so chiem tu nhien: max {occ[0]:.6f}  min {occ[1]:+.3e}"
              f"  ngoai khoang: {occ[2]}"
              f"{'  << NGOAI [0,1]' if occ[2] else ''}", flush=True)

        if mode == "ff":
            dip, gmax, curv, nbad = dipole_finite_field(mol, mf, row["E_UCCSD"])
            row["ff_grad_max"] = float(gmax)
            row["ff_curv"] = float(curv)
            row["ff_ccsd_bad"] = int(nbad)
            if gmax > GRAD_TOL:
                flags.append(f"FF_SCF(|g|={gmax:.1e})")
            if curv > CURV_WARN:
                flags.append(f"FF_state_jump(curv={curv:.1e})")
            if nbad:
                flags.append(f"FF_CCSD_bad={nbad}")
        else:
            dip = dip_dm                 # bo qua finite field
        row["mu_ff_CCSD"] = float(np.linalg.norm(dip))
        row["mux_ff"], row["muy_ff"], row["muz_ff"] = map(float, dip)
        if row["mu_ff_CCSD"] > 1e-8:
            row["d_dm_ff_CCSD"] = 100.0 * (row["mu_dm_CCSD"] - row["mu_ff_CCSD"]) \
                                  / row["mu_ff_CCSD"]

        if flags:
            row["status"] = "WARN " + " ".join(flags)

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
    bad = df[df.status != 'ok']
    print(f"        chất có cảnh báo: {len(bad)}")
    for _, r in bad.iterrows():
        print(f"          {r['molecule']:10} {r['status']}")
    if "ff_curv" in df and df.ff_curv.notna().any():
        print(f"        độ cong finite-field max: {df.ff_curv.max():.2e} "
              f"({df.loc[df.ff_curv.idxmax(), 'molecule']})")

    if "sym_break" in df:
        sb = df[df.sym_break.fillna("").astype(str) != ""]
        if len(sb):
            print(f"        !! {len(sb)} chất có nghiệm UHF phá đối xứng không gian:")
            for _, r in sb.iterrows():
                print(f"           {r['molecule']:10} trục {r['sym_break']}")
        else:
            print(f"        không chất nào phá đối xứng không gian ({len(df)} chất)")

    if "n_bad_CCSD" in df and df.n_bad_CCSD.notna().any():
        v = df[df.n_bad_CCSD.fillna(0) > 0]
        if len(v):
            print(f"        !! {len(v)} chất có số chiếm CCSD ngoài [0,1]:")
            for _, r in v.sort_values("nocc_max_CCSD", ascending=False).iterrows():
                print(f"           {r['molecule']:10} max = {r['nocc_max_CCSD']:.6f}"
                      f"   min = {r['nocc_min_CCSD']:+.3e}   ({int(r['n_bad_CCSD'])} orbital)")
        else:
            print(f"        mật độ CCSD N-representable trên cả {len(df)} chất")

    c = "d_dm_ff_CCSD"
    if c in df and df[c].notna().any():
        t = df[df[c].notna()]
        print(f"        mật độ (make_rdm1) so với finite field: "
              f"RMS {np.sqrt((t[c]**2).mean()):.2f} %   "
              f"max {t[c].abs().max():.2f} % ({t.loc[t[c].abs().idxmax(),'molecule']})")

    if ref_json and os.path.exists(ref_json):
        ref = json.load(open(ref_json))
        df["ref"] = df.molecule.map(ref)
        sub = df[df.ref.notna()]
        print(f"\n        RMSE regularized so với {len(sub)} giá trị CCSD(T)/CBS:")
        for col, tag in [("mu_ff_CCSD", "finite field (-dE/dF)"),
                         ("mu_dm_CCSD", "mật độ (make_rdm1)")]:
            if col not in sub:
                continue
            s = sub[sub[col].notna()]
            if not len(s):
                continue
            err = 100 * (s[col] - s.ref) / np.maximum(s.ref, 1.0)
            print(f"          {tag:22} n={len(s):3}  RMSE = {np.sqrt((err**2).mean()):6.2f} %"
                  f"   ME = {err.mean():+6.2f} %"
                  f"   MAX = {err.abs().max():6.2f} % ({s.loc[err.abs().idxmax(),'molecule']})")
        print("          (bài báo, CCSD trên tập SP: RMSE 4.80 %)")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="file JSON hình học (không cần khi --merge)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--dipole", choices=["ff", "dm"], default="ff",
                    help="ff = chạy finite field (mặc định, cột chính); "
                         "dm = BỎ QUA finite field cho rẻ gấp 7 lần. Ba cột mật "
                         "độ (det, unrel, lambda) LUÔN được tính trong cả hai chế độ.")
    ap.add_argument("--max-memory", type=int,
                    default=int(os.environ.get("PYSCF_MAX_MEMORY", "30000")))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--output-xlsx", default=None)
    ap.add_argument("--ref-json", default=None)
    args = ap.parse_args()

    out_xlsx = args.output_xlsx or os.path.join(args.outdir, "dipole_results.xlsx")

    if args.merge:                    # không cần pyscf, không cần --input
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
        row = run_one(name, props, args.max_memory, args.dipole)
        p = write_row(args.outdir, i, row)
        print(f"[{i}] {name}: {row['status']} | ff={row['mu_ff_CCSD']} "
              f"dm={row['mu_dm_CCSD']} lệch={row['d_dm_ff_CCSD']}% | "
              f"|g|={row['grad_norm']} | spread={row['E_spread_mH']} mH | "
              f"{row['walltime_s']}s -> {p}", flush=True)

    if len(todo) > 1 and not args.only and args.index is None:
        do_merge(args.outdir, out_xlsx, args.ref_json)


if __name__ == "__main__":
    main()