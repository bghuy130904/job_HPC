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
FIELD        = 1e-4      # a.u., giong bai bao (sai phan trung tam hai diem)
AU2DEBYE     = 2.541746
CURV_WARN    = 1e-5      # |E+ + E- - 2E0| lon hon -> diem truong nhay nghiem


def _lazy_imports():
    global gto, scf, cc, mp, df, stabilize_scf, OBDH_CL, OBMP2_CL
    global attach
    from pyscf import gto, scf, cc, mp, df                # noqa: F401
    from pycmf.OBDH.stability import stabilize_scf        # noqa: F401
    from pycmf.OBDH import OBDH_CL, OBMP2_CL              # noqa: F401
    # obmp2_rdm1.py phai nam cung thu muc voi script nay (hoac tren PYTHONPATH)
    from pycmf.OBDH.obdh_rdm1 import attach                         # noqa: F401
    if not hasattr(np.linalg, 'linalg'):
        np.linalg.linalg = np.linalg


# ----------------------------------------------------------------------
def grad_norm(mf):
    dm = mf.make_rdm1()
    return float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock(dm=dm))))


def _mk_uhf(mol, hcore=None):
    # AUXBASIS -- doc ky truoc khi doi.
    #
    # DFOBMP2.__init__ (dfobmp2.py:650) MUON lai mf.with_df cho bien do MP2:
    #     if getattr(mf, 'with_df', None): self.with_df = mf.with_df
    # nen bo auxbasis chon o day di thang vao eqn (4). def2-universal-jkfit
    # duoc thiet ke de khop J/K o co hoa tri def2, khong co ham khop mat do
    # cap loi-loi va loi-hoa tri. Voi aug-cc-pCVQZ + frozen=0 (tuong quan ca
    # loi) no qua nho: naux 152 cho 218 AO, trong khi RI-MP2 can 3-4 lan nao.
    #
    # Do tren CN / aug-cc-pCVTZ, sai so E_corr so voi ERI chinh xac:
    #     def2-universal-jkfit   naux 152   +6.798 mEh   (frozen=2: -0.493)
    #     aug-cc-pvtz-ri         naux 212   +4.002 mEh   (frozen=2: -0.028)
    #     def2-qzvppd-ri         naux 314   +0.168 mEh
    #     make_auxbasis(mp2fit)  naux 489   -0.014 mEh
    # Toan bo sai so den tu tuong quan loi -- dong bang loi thi bo nao cung on.
    # Khong ton tai bo khop co ten cho aug-cc-pCVQZ (BSE khong co
    # aug-cc-pcvqz-ri lan -jkfit), nen make_auxbasis(mp2fit=True) roi ve
    # even-tempered tu sinh; do chinh la lua chon tot nhat o day.
    #
    # O muc SCF thi jkfit hoan toan on (+0.044 mEh, -0.043 mD tren CN/
    # aug-cc-pCVTZ) -- vi the calc_dipmomt_ccsd_ff.py va calc_dipmomt_dft.py
    # GIU NGUYEN jkfit: cc.UCCSD va mp.UMP2 KHONG ke thua mf.with_df, chung
    # dung ERI chinh xac. Chi rieng solver OBMP2/OBDH ke thua.
    mf = scf.UHF(mol).density_fit(auxbasis=df.make_auxbasis(mol, mp2fit=True))
    mf.verbose = 0
    mf.max_cycle = 150
    if hcore is not None:
        mf.get_hcore = lambda *a, **k: hcore
    return mf


def _one_scf(mol, guess):
    mf = _mk_uhf(mol)
    mf.init_guess = guess
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






# ----------------------------------------------------------------------
def _nuc_dip(mol):
    return np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())


def symmetry_flag(mol, dip, tol=1e-3):
    """Truc nao co mu_nuc = 0 thi mu tong cung phai = 0 (doi xung diem).

    Day la dieu kien CHUNG, ap cho ca 71 chat, khong phai ban va cho mot chat.
    Phan tu khong doi xung khong co truc nao bang 0 -> tu dong thoa.

    Vi pham nghia la nghiem SCF pha doi xung KHONG GIAN (khac voi pha doi xung
    spin). Vi du ClO2 (C2v quanh truc z): sau stabilize_scf, UHF cho
    mu = (0, 0.859, 2.077) D -- thanh phan y le ra phai bang 0. UMP2 khuech dai
    no thanh -5.347 D.

    CHI GHI LAI, khong doi nghiem, khong doi so lieu. Quy trinh giu nguyen cho
    moi chat; cot nay de biet mot con so bat thuong den tu dau.
    """
    nuc = _nuc_dip(mol) * AU2DEBYE
    bad = [ax for ax, (n, m) in zip("xyz", zip(nuc, np.asarray(dip, float)))
           if abs(n) < tol and abs(m) > tol]
    return "".join(bad)


def _energy_in_field(mol, key, hcore, dm0):
    """Nang luong cua mot phuong phap voi h1 = hcore0 + F.r."""
    mf = _mk_uhf(mol, hcore)
    mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.mo_coeff, mf.mo_occ)

    # Voi phuong phap toi uu orbital, diem truong PHAI duoc on dinh hoa.
    # Tham chieu pha doi xung (S2 ~ 0.8) co hai nghiem gan suy bien; dm0 cua
    # truong 0 khong con nam trong luu vuc hut dung khi them +-F, DIIS ve diem
    # YEN NGUA (grad_norm van tot nen khong lo ra), roi vong OO khuech dai sai
    # lech. Do o FH r=1.74: khong stabilize -> mu = 34.68 D; co stabilize ->
    # mu = 2.88 D. UHF/UMP2 khong co vong OO nen khong can.
    if key in ("obmp2", "obdh"):
        mf = stabilize_scf(mf, max_macro_cycles=10, verbose=False)

    if key == "uhf":
        return float(mf.e_tot)
    if key == "ump2":
        pt = mp.UMP2(mf); pt.verbose = 0; pt.kernel()
        return float(pt.e_tot)
    s = _mk_solver(mf, hybrid=(key == "obdh"))
    s.run()
    return float(s.ene_tot)


def dipole_ff(mol, key, mf0, e0):
    """
    mu = mu_nuc - dE/dF, sai phan trung tam F = 1e-4 a.u. (nhu bai bao).
    Diem truong DUNG LAI nghiem truong 0 lam guess -> khong nhay trang thai.
    Tra ve (vector Debye, do cong lon nhat).
    Do cong |E+ + E- - 2E0| ~ F^2 * do phan cuc (~1e-8 Eh). Neu lon thi mot
    diem truong da roi vao nghiem khac -> so lieu chat do khong dung duoc.
    """
    h0 = _mk_uhf(mol).get_hcore()
    with mol.with_common_orig((0, 0, 0)):
        r = mol.intor('int1e_r', comp=3)
    dm0 = mf0.make_rdm1()
    d = np.zeros(3); curv = 0.0
    for x in range(3):
        f = np.zeros(3); f[x] = FIELD
        pert = np.einsum('i,iuv->uv', f, r)
        ep = _energy_in_field(mol, key, h0 + pert, dm0)
        em = _energy_in_field(mol, key, h0 - pert, dm0)
        d[x] = (ep - em) / (2 * FIELD)
        curv = max(curv, abs(ep + em - 2 * e0))
    return (_nuc_dip(mol) - d) * AU2DEBYE, curv


def _mk_solver(mf, hybrid):
    s = OBDH_CL(mf) if hybrid else OBMP2_CL(mf)
    s.verbose = 0
    s.alphaa = ALPHAA
    s.thresh = OBDH_THRESH
    s.niter = OBDH_NITER
    s.second_order = True
    s.mom_select = False
    s.use_embed = False
    s.use_cl = False
    return s


def _energy_zero_field(mol, key, mf, alpha_c=1.0, want_unrel=False):
    """Nang luong truong 0 + cac mat do co the dung + chan doan.

    Tra ve (E, dip_det, dip_unrel, diag):
      dip_det    mat do DINH THUC tren orbital da toi uu (uobdh_solver.py:650).
                 So chiem cung 1/0 -> THIEU dong gop amplitude.
      dip_unrel  I + d_oo + d_vv dung bien do eqn (4) cua Tran, PCCP 2022:
                 "evaluated using the T2 amplitude (eqn (4)) as in standard
                 MP2".  Khoi ov = 0.
      diag       dict chan doan, xem duoi.

    diag['dE_HF_mH'] = E_HF[orbital OBMP2] - E_HF[orbital HF], mHartree.
        Luon >= 0 vi orbital HF cuc tieu E_HF.  Do "khoang cach nang luong"
        giua hai bo orbital, tuc muc do hoi phuc orbital ma vong OO da lam.
        O CN no la 24 mH, di kem goc quay 12.6/17.1 do va <S2> tut tu 1.15
        xuong 0.77.  Chat nao co so nay lon thi OBMP2 lech xa MP2 chuan.

    diag['occ_max'], ['occ_min'], ['n_bad'] = pho so chiem tu nhien cua mat do
        unrelaxed.  N-representable khi 0 <= n <= 1.  n_bad dem so orbital ra
        ngoai khoang do; khac 0 la dau hieu vi pham (Kurlancheek &
        Head-Gordon, Mol. Phys. 107, 1223).  Luu y Tr = N van dung ke ca khi
        vi pham, nen hai chot kiem tra nay doc lap.

    Voi UHF/UMP2 thi dip_unrel = dip_det va diag rong.
    """
    nan3 = np.array([np.nan] * 3)

    if key == "uhf":
        # Mot dinh thuc, khong co tuong quan -> det VA unrel la cung mot vat.
        v = np.asarray(_dip_from_dm(mol, mf.make_rdm1()), float)
        return float(mf.e_tot), v, v.copy(), {}

    if key == "ump2":
        pt = mp.UMP2(mf); pt.verbose = 0; pt.kernel()
        # UMP2 khong toi uu orbital -> dinh thuc tham chieu CHINH LA dinh thuc
        # UHF, nen mu_det_UMP2 == mu_det_UHF theo dung dinh nghia. Giu lai de
        # cot co nghia thong nhat giua cac phuong phap, khong phai trung hop.
        v_det = np.asarray(_dip_from_dm(mol, mf.make_rdm1()), float)
        # pt.make_rdm1 = gamma_ref + d_oo + d_vv, khoi ov = 0 -> DUNG la unrelaxed
        v_unrel = (np.asarray(_dip_from_dm(mol, pt.make_rdm1(ao_repr=True)), float)
                   if want_unrel else nan3)
        return float(pt.e_tot), v_det, v_unrel, {}

    s = _mk_solver(mf, hybrid=(key == "obdh"))
    s.run()
    g = s._gamma
    dip_det = _dip_from_dm(mol, (g[0], g[1]))

    ac = alpha_c if key == "obdh" else 1.0
    diag = {}
    dip_unrel = nan3
    if want_unrel:
        try:
            attach(s, alpha_c=ac)
            dip_unrel = np.asarray(
                _dip_from_dm(mol, s.rdm1_unrelaxed(ao_repr=True, alpha_c=ac)), float)
            occ = np.concatenate(s.natural_occupations(alpha_c=ac))
            diag["occ_max"] = float(occ.max())
            diag["occ_min"] = float(occ.min())
            diag["n_bad"] = int(((occ < -1e-6) | (occ > 1 + 1e-6)).sum())
        except Exception as e:
            dip_unrel = nan3
            diag["occ_err"] = f"{type(e).__name__}: {e}"

    try:
        dm_ob = scf.uhf.make_rdm1(s.mo_coeff, mf.mo_occ)
        diag["dE_HF_mH"] = (mf.energy_tot(dm=dm_ob) - mf.e_tot) * 1000.0
    except Exception:
        pass
    return float(s.ene_tot), dip_det, dip_unrel, diag


METHODS = {"uhf": "UHF", "ump2": "UMP2", "obmp2": "OBMP2", "obdh": "OBDH"}
ANALYTIC = {"uhf"}          # UHF bien phan -> mat do cho dipole dung


# ----------------------------------------------------------------------
def run_one(name, props, max_memory, methods, mode='det'):
    """mode: 'det' | 'unrel' | 'ff' | 'both'  -- xem help cua --dipole."""
    want_unrel = mode in ("unrel", "both")
    want_ff    = mode in ("ff", "both")

    t0 = time.time()
    row = {"molecule": name, "charge": props.get("charge"), "spin": props.get("spin"),
           "nao": None, "grad_norm": None, "E_spread_mH": None, "n_guess_ok": None,
           "S2": None, "sym_break": None, "status": "ok"}
    for k in methods:
        F = METHODS[k]
        row.update({f"E_{F}": None,
                    # --- ba dai luong dipole, moi cai mot nguon, khong dung chung o --
                    f"mu_det_{F}": None,       # mat do DINH THUC (so chiem 0/1)
                    f"mu_unrel_{F}": None,     # I + d_oo + d_vv  (= PCCP 2022)
                    f"mu_ff_{F}": None,        # -dE/dF, sai phan trung tam
                    # thanh phan vector duoc ghi rieng theo tung nguon
                    f"mux_det_{F}": None, f"muy_det_{F}": None, f"muz_det_{F}": None,
                    f"mux_ff_{F}": None, f"muy_ff_{F}": None, f"muz_ff_{F}": None,
                    f"d_det_ff_{F}": None,     # % lech det vs ff
                    f"d_unrel_ff_{F}": None,   # % lech unrel vs ff
                    f"dE_HF_mH_{F}": None,
                    f"nocc_max_{F}": None, f"nocc_min_{F}": None,
                    f"n_bad_{F}": None,
                    f"curv_{F}": None, f"t_{F}": None})
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

        for k in methods:
            F = METHODS[k]
            t1 = time.time()
            try:
                e0, dip_det, dip_unrel, diag = _energy_zero_field(
                    mol, k, mf, alpha_c=ALPHAA[1], want_unrel=want_unrel)
                row[f"E_{F}"] = e0
                row[f"mu_det_{F}"] = float(np.linalg.norm(dip_det))
                row[f"mux_det_{F}"], row[f"muy_det_{F}"], row[f"muz_det_{F}"] = \
                    map(float, dip_det)
                if np.all(np.isfinite(dip_unrel)):
                    row[f"mu_unrel_{F}"] = float(np.linalg.norm(dip_unrel))
                if "dE_HF_mH" in diag:
                    row[f"dE_HF_mH_{F}"] = round(float(diag["dE_HF_mH"]), 4)
                if "occ_max" in diag:
                    row[f"nocc_max_{F}"] = round(diag["occ_max"], 8)
                    row[f"nocc_min_{F}"] = round(diag["occ_min"], 10)
                    row[f"n_bad_{F}"] = diag["n_bad"]
                    if diag["n_bad"]:
                        flags.append(f"{F}:N_repr({diag['n_bad']})")
                    bad = "  << NGOAI [0,1]" if diag["n_bad"] else ""
                    print(f"    {F:6} so chiem tu nhien: max {diag['occ_max']:.6f}"
                          f"  min {diag['occ_min']:+.3e}"
                          f"  ngoai khoang: {diag['n_bad']}{bad}", flush=True)
                elif "occ_err" in diag:
                    print(f"    {F:6} so chiem: {diag['occ_err']}", flush=True)

                # mu_ff CHI duoc ghi bang mot so thuc su la -dE/dF. Khong bao gio
                # do mat do dinh thuc vao day: cot phai giu dung nhan cua no.
                dip_ff = None
                if k in ANALYTIC:
                    # UHF bien phan -> Hellmann-Feynman ap dung chinh xac, mat do
                    # dinh thuc LA -dE/dF. Khong can chay finite field.
                    dip_ff, curv = dip_det, 0.0
                elif want_ff:
                    dip_ff, curv = dipole_ff(mol, k, mf, e0)
                    row[f"curv_{F}"] = float(curv)
                    if curv > CURV_WARN:
                        flags.append(f"{F}:FF_jump({curv:.1e})")

                if dip_ff is not None:
                    ref = float(np.linalg.norm(dip_ff))
                    row[f"mu_ff_{F}"] = ref
                    row[f"mux_ff_{F}"], row[f"muy_ff_{F}"], row[f"muz_ff_{F}"] = \
                        map(float, dip_ff)
                    if ref > 1e-8:
                        row[f"d_det_ff_{F}"] = \
                            100.0 * (row[f"mu_det_{F}"] - ref) / ref
                        if row[f"mu_unrel_{F}"] is not None:
                            row[f"d_unrel_ff_{F}"] = \
                                100.0 * (row[f"mu_unrel_{F}"] - ref) / ref
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
    """Chuoi tom tat cho MOT phuong phap: chi liet ke dai luong da thuc su
    tinh duoc. Phuong phap khong chay (hoac chay loi) tra ve chuoi rong ->
    khong xuat hien trong dong tom tat."""
    parts = [f"{lab}={row[key]:.4f}"
             for key, lab in ((f"mu_det_{F}", "det"),
                              (f"mu_unrel_{F}", "unrel"),
                              (f"mu_ff_{F}", "ff"))
             if row.get(key) is not None and np.isfinite(row[key])]
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

    # nghiem SCF pha doi xung khong gian
    if "sym_break" in df:
        sb = df[df.sym_break.fillna("").astype(str) != ""]
        if len(sb):
            print(f"        !! {len(sb)} chat co nghiem UHF pha doi xung khong gian:")
            for _, r in sb.iterrows():
                print(f"           {r['molecule']:10} truc {r['sym_break']}")
        else:
            print(f"        khong chat nao pha doi xung khong gian ({len(df)} chat)")

    # N-representability cua mat do unrelaxed, tren ca tap
    for tag in ["OBMP2", "OBDH"]:
        col = f"n_bad_{tag}"
        if col not in df or df[col].isna().all():
            continue
        v = df[df[col].fillna(0) > 0]
        if len(v):
            print(f"        !! {tag}: {len(v)} chat co so chiem ngoai [0,1]:")
            for _, r in v.sort_values(f"nocc_max_{tag}", ascending=False).iterrows():
                print(f"           {r['molecule']:10} max = {r[f'nocc_max_{tag}']:.6f}"
                      f"   min = {r[f'nocc_min_{tag}']:+.3e}"
                      f"   ({int(r[col])} orbital)")
        else:
            print(f"        {tag}: mat do unrelaxed N-representable tren ca {len(df)} chat")

    # thống kê so với tham chiếu, nếu có
    if ref_json and os.path.exists(ref_json):
        ref = json.load(open(ref_json))
        df["ref"] = df.molecule.map(ref)
        sub = df[df.ref.notna()]
        print(f"\n        RMSE regularized so voi {len(sub)} gia tri tham chieu:")
        labels = {"mu_det": "mat do dinh thuc",
                  "mu_unrel": "unrelaxed I+d_oo+d_vv (= PCCP)",
                  "mu_ff": "finite field  (-dE/dF)"}
        for pre, lab in labels.items():
            print(f"\n          --- {lab} ---")
            for tag in ["UHF", "UMP2", "OBMP2", "OBDH"]:
                col = f"{pre}_{tag}"
                if col not in sub:
                    continue
                s = sub[sub[col].notna()]
                if not len(s):
                    continue
                err = 100 * (s[col] - s.ref) / np.maximum(s.ref, 1.0)
                print(f"          {tag:6} n={len(s):3}  RMSE = {np.sqrt((err**2).mean()):7.2f} %"
                      f"   ME = {err.mean():+7.2f} %   MAX = {err.abs().max():7.2f} %"
                      f"  ({s.loc[err.abs().idxmax(), 'molecule']})")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="file JSON hình học (không cần khi --merge)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=["uhf", "ump2", "obmp2", "obdh"],
                    choices=list(METHODS))
    ap.add_argument("--dipole", choices=["det", "unrel", "ff", "both"], default="det",
                    help="dai luong dipole nao duoc tinh. mu_det (mat do dinh thuc "
                         "tren orbital da toi uu) LUON co vi no mien phi sau khi "
                         "solver chay xong. "
                         "det (mac dinh) = chi mu_det; "
                         "unrel = them mu_unrel (I+d_oo+d_vv, = PCCP 2022) va pho "
                         "so chiem tu nhien; "
                         "ff = them mu_ff (-dE/dF, sai phan trung tam, DAT: 6 diem "
                         "truong x 1 lan chay day du moi diem); "
                         "both = ca mu_unrel lan mu_ff. "
                         "Cot nao khong duoc yeu cau se de trong, KHONG bi dien "
                         "bang gia tri cua cot khac.")
    ap.add_argument("--max-memory", type=int,
                    default=int(os.environ.get("PYSCF_MAX_MEMORY", "30000")))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--output-xlsx", default=None)
    ap.add_argument("--ref-json", default=None,
                    help="file JSON {tên chất: dipole CCSD(T)} để tính RMSE khi merge")
    args = ap.parse_args()

    out_xlsx = args.output_xlsx or os.path.join(args.outdir, "dipole_obdh.xlsx")

    if args.merge:                       # không cần pyscf/pycmf, không cần --input
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
        row = run_one(name, props, args.max_memory, args.methods, args.dipole)
        p = write_row(args.outdir, i, row)
        mus = "  ".join(x for x in (_fmt_dip(row, METHODS[k]) for k in args.methods) if x)
        print(f"[{i}] {name}: {row['status']} | |g|={row['grad_norm']} | "
              f"spread={row['E_spread_mH']} mH | {mus} | {row['walltime_s']}s -> {p}",
              flush=True)

    if len(todo) > 1 and not args.only and args.index is None:
        do_merge(args.outdir, out_xlsx, args.ref_json)


if __name__ == "__main__":
    main()