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

CCSD(T) (--levels ccsd ccsd_t, mặc định bật)
    Thêm cột E_CCSD_T / mu_dm_CCSD_T / mu_ff_CCSD_T để tự dựng giá trị tham
    chiếu thay vì mượn của nhóm.

    --dipole ff : rẻ. CCSD và CCSD(T) DÙNG CHUNG 6 điểm trường — mỗi điểm chạy
        SCF + CCSD một lần rồi lấy ra hai năng lượng, nên (T) chỉ thêm 6 lần
        tính (T), không nhân đôi số lần CCSD.
    --dipole dm : ĐẮT. Mật độ (T) cần phương trình Lambda có số hạng (T)
        (uccsd_t_lambda), mỗi vòng lặp Lambda phải dựng lại phần (T) -> khoảng
        10-30 lần chi phí một lần (T) đơn lẻ. Ở aug-cc-pCVQZ đây là bước đắt
        nhất của cả script. Nếu chỉ cần năng lượng thì --levels ccsd_t với
        --dipole ff rẻ hơn nhiều.

    CẢNH BÁO VỀ TỪ "THAM CHIẾU": Hait & Head-Gordon dùng CCSD(T)/CBS (ngoại suy
    tới giới hạn bộ cơ sở). CCSD(T)/aug-cc-pCVQZ ở đây là MỘT bộ cơ sở, không
    phải CBS — hai đại lượng khác nhau. Nếu thay giá trị của nhóm bằng cột này
    thì phải nói rõ trong bài, và nên chạy thêm ít nhất một bộ (aug-cc-pCVTZ)
    để ước lượng phần còn thiếu tới CBS.

    Mật độ (T) cùng hạng với mật độ CCSD (không có Z-vector), nên vẫn gọi là
    mu_dm chứ không phải mu_unrel. Đo ở OH/6-31G: CCSD lệch -0.25 % so với
    finite field, CCSD(T) chỉ lệch +0.07 % — Lambda có (T) hấp thụ thêm phần
    hồi phục.

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
    global uccsd_t, uccsd_t_lambda, uccsd_t_rdm
    from pyscf import gto, scf, cc                        # noqa: F401
    from pyscf.cc import uccsd_t, uccsd_t_lambda, uccsd_t_rdm   # noqa: F401
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


LEVELS = {"ccsd": "CCSD", "ccsd_t": "CCSD_T"}


def _t_lambda(mycc, eris):
    """Giai phuong trinh Lambda co so hang (T).

    DAT: moi vong lap Lambda phai dung lai phan (T), nen chi phi vao khoang
    10-30 lan mot lan tinh (T) don le. Day la buoc dat nhat cua ca script khi
    chay --dipole dm o co aug-cc-pCVQZ.
    """
    conv, l1, l2 = uccsd_t_lambda.kernel(mycc, eris, mycc.t1, mycc.t2, verbose=0)
    return bool(conv), l1, l2


def ccsd_density(mol, mycc, l=None, eris=None):
    """Mat do CCSD tu mycc.make_rdm1() -- dung dai luong Tran (PCCP 2022) dung
    lam moc, va ho ghi ro "its density matrix is not relaxed".

    Luu y ve phan loai: khoi ov cua no KHAC 0, den tu bien do don
    (pyscf uccsd_rdm: dova = l1a), khong phai tu Z-vector. Nen no khong nam
    cung o voi cot 'unrel' cua MP2/OBMP2 (o do dov = 0) -- co ov nhung van
    chua phai relaxed. Vi vay giu ten rieng la mu_dm.

    Tra ve (dipole Debye, (occ_max, occ_min, n_bad)).
    """
    if l is None:
        d_mo = mycc.make_rdm1()                   # CCSD: Lambda cua chinh no
    else:
        # CCSD(T): mat do dap ung voi Lambda co so hang (T).
        # Cung hang voi mat do CCSD (khong co Z-vector), nen van goi la mu_dm.
        d_mo = uccsd_t_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l[0], l[1],
                                     eris=eris, ao_repr=False)
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
def _ccsd_in_field(mol, hcore, dm0, want_t):
    """E_CCSD (va E_CCSD(T) neu want_t) voi h1 = hcore0 + F.r. Dung dm0 cua
    truong 0 lam guess de KHONG nhay sang trang thai dien tu khac.

    Tra ve (dict {tag: E}, |grad| SCF, CCSD hoi tu?).
    (T) o day chi la NANG LUONG -- khong can Lambda, nen re: mot lan (T) moi
    diem truong."""
    mf = _scf_once(mol, hcore=hcore, dm0=dm0)
    g = grad_norm(mf)
    mycc = cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    e = {"CCSD": float(mycc.e_tot)}
    if want_t:
        eris = mycc.ao2mo()
        e["CCSD_T"] = float(mycc.e_tot + uccsd_t.kernel(mycc, eris, mycc.t1,
                                                        mycc.t2, verbose=0))
    return e, g, bool(mycc.converged)


def dipole_finite_field(mol, mf0, e0, tags):
    """
    mu = mu_nuc - dE/dF, sai phan trung tam. CCSD va CCSD(T) dung CHUNG cac
    diem truong: mot lan SCF + CCSD moi diem, roi lay hai nang luong ra. Nen
    them (T) chi ton them 6 lan tinh (T), khong nhan doi so lan CCSD.

    Tra ve ({tag: vector Debye}, max|grad|, {tag: max do cong}, so lan CCSD
    khong hoi tu).
    Do cong |E+ + E- - 2E0| ~ F^2 * do phan cuc nen rat nho. Neu no lon thi mot
    diem truong da nhay sang trang thai khac -> ket qua khong dung duoc.
    """
    h0 = _mk_uhf(mol).get_hcore()
    with mol.with_common_orig((0, 0, 0)):
        r = mol.intor('int1e_r', comp=3)
    dm0 = mf0.make_rdm1()
    want_t = "CCSD_T" in tags

    d = {t: np.zeros(3) for t in tags}
    curv = {t: 0.0 for t in tags}
    gmax, nbad = 0.0, 0
    for x in range(3):
        f = np.zeros(3)
        f[x] = FIELD
        pert = np.einsum('i,iuv->uv', f, r)
        ep, gp, cp = _ccsd_in_field(mol, h0 + pert, dm0, want_t)
        em, gm, cm = _ccsd_in_field(mol, h0 - pert, dm0, want_t)
        gmax = max(gmax, gp, gm)
        nbad += (not cp) + (not cm)
        for t in tags:
            d[t][x] = (ep[t] - em[t]) / (2 * FIELD)
            curv[t] = max(curv[t], abs(ep[t] + em[t] - 2 * e0[t]))

    nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    return {t: (nuc - d[t]) * AU2DEBYE for t in tags}, gmax, curv, nbad


# ----------------------------------------------------------------------
def run_one(name, props, max_memory, mode, levels=("ccsd", "ccsd_t")):
    tags = [LEVELS[k] for k in levels]
    want_t = "CCSD_T" in tags

    t0 = time.time()
    row = {"molecule": name, "charge": props.get("charge"), "spin": props.get("spin"),
           "nao": None, "converged": None, "grad_norm": None, "E_spread_mH": None,
           "n_guess_ok": None, "S2": None, "sym_break": None, "E_UHF": None,
           "ff_grad_max": None, "ff_ccsd_bad": None,
           "walltime_s": None, "status": "ok"}
    for T in ("CCSD", "CCSD_T"):
        row.update({f"E_{T}": None,
                    f"mu_dm_{T}": None, f"mu_ff_{T}": None, f"d_dm_ff_{T}": None,
                    f"mux_dm_{T}": None, f"muy_dm_{T}": None, f"muz_dm_{T}": None,
                    f"mux_ff_{T}": None, f"muy_ff_{T}": None, f"muz_ff_{T}": None,
                    f"nocc_max_{T}": None, f"nocc_min_{T}": None, f"n_bad_{T}": None,
                    f"lambda_conv_{T}": None, f"curv_{T}": None})
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

        # ---------- truong 0 ----------
        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        mycc.kernel()
        if not mycc.converged:
            flags.append("CCSD_not_converged")
        row["E_CCSD"] = float(mycc.e_tot)

        eris = mycc.ao2mo() if want_t else None
        if want_t:
            et = uccsd_t.kernel(mycc, eris, mycc.t1, mycc.t2, verbose=0)
            row["E_CCSD_T"] = float(mycc.e_tot + et)

        # ---------- mat do ----------
        # CCSD mien phi (Lambda cua no duoc giai san). CCSD(T) thi KHONG:
        # no can Lambda co so hang (T), dat gap hang chuc lan. Vi vay mat do
        # (T) chi duoc tinh khi thuc su xin muc do do.
        for T in tags:
            if T == "CCSD":
                dip_dm, occ = ccsd_density(mol, mycc)
                row["lambda_conv_CCSD"] = True
            else:
                conv, l1, l2 = _t_lambda(mycc, eris)
                row["lambda_conv_CCSD_T"] = conv
                if not conv:
                    flags.append("T_lambda_not_converged")
                dip_dm, occ = ccsd_density(mol, mycc, l=(l1, l2), eris=eris)
            row[f"mu_dm_{T}"] = float(np.linalg.norm(dip_dm))
            row[f"mux_dm_{T}"], row[f"muy_dm_{T}"], row[f"muz_dm_{T}"] = map(float, dip_dm)
            row[f"nocc_max_{T}"], row[f"nocc_min_{T}"], row[f"n_bad_{T}"] = occ
            if occ[2]:
                flags.append(f"{T}:N_repr({occ[2]})")
            print(f"    {T:7} so chiem tu nhien: max {occ[0]:.6f}  min {occ[1]:+.3e}"
                  f"  ngoai khoang: {occ[2]}"
                  f"{'  << NGOAI [0,1]' if occ[2] else ''}", flush=True)

        # ---------- finite field ----------
        # mu_ff CHI duoc ghi bang mot so thuc su la -dE/dF. Khong bao gio do
        # mu_dm vao day: cot phai giu dung nhan cua no. O mode 'dm' thi
        # mu_ff_* de trong, va cot mu_dm_* da co san gia tri mat do.
        if mode == "ff":
            e0 = {T: row[f"E_{T}"] for T in tags}
            dips, gmax, curv, nbad = dipole_finite_field(mol, mf, e0, tags)
            row["ff_grad_max"] = float(gmax)
            row["ff_ccsd_bad"] = int(nbad)
            if gmax > GRAD_TOL:
                flags.append(f"FF_SCF(|g|={gmax:.1e})")
            if nbad:
                flags.append(f"FF_CCSD_bad={nbad}")
            for T in tags:
                row[f"curv_{T}"] = float(curv[T])
                if curv[T] > CURV_WARN:
                    flags.append(f"{T}:FF_state_jump({curv[T]:.1e})")
                ref = float(np.linalg.norm(dips[T]))
                row[f"mu_ff_{T}"] = ref
                row[f"mux_ff_{T}"], row[f"muy_ff_{T}"], row[f"muz_ff_{T}"] = \
                    map(float, dips[T])
                if ref > 1e-8:
                    row[f"d_dm_ff_{T}"] = 100.0 * (row[f"mu_dm_{T}"] - ref) / ref

        if flags:
            row["status"] = "WARN " + " ".join(flags)

    except Exception as e:
        row["status"] = f"ERROR: {type(e).__name__}: {e}"

    row["walltime_s"] = round(time.time() - t0, 2)
    return row


# ----------------------------------------------------------------------
def _fmt_dip(row, T):
    """Chuoi tom tat cho MOT muc ly thuyet: chi liet ke dai luong da thuc su
    tinh duoc. Muc khong chay (hoac chay loi) tra ve chuoi rong."""
    parts = [f"{lab}={row[c]:.4f}"
             for c, lab in ((f"mu_dm_{T}", "dm"), (f"mu_ff_{T}", "ff"))
             if row.get(c) is not None and np.isfinite(row[c])]
    if row.get(f"d_dm_ff_{T}") is not None:
        parts.append(f"lech={row[f'd_dm_ff_{T}']:+.2f}%")
    return f"{T}[{' '.join(parts)}]" if parts else ""


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
    for T in ("CCSD", "CCSD_T"):
        c = f"curv_{T}"
        if c in df and df[c].notna().any():
            print(f"        {T:7} độ cong finite-field max: {df[c].max():.2e} "
                  f"({df.loc[df[c].idxmax(), 'molecule']})")

    if "sym_break" in df:
        sb = df[df.sym_break.fillna("").astype(str) != ""]
        if len(sb):
            print(f"        !! {len(sb)} chất có nghiệm UHF phá đối xứng không gian:")
            for _, r in sb.iterrows():
                print(f"           {r['molecule']:10} trục {r['sym_break']}")
        else:
            print(f"        không chất nào phá đối xứng không gian ({len(df)} chất)")

    for T in ("CCSD", "CCSD_T"):
        cb = f"n_bad_{T}"
        if cb in df and df[cb].notna().any():
            v = df[df[cb].fillna(0) > 0]
            if len(v):
                print(f"        !! {len(v)} chất có số chiếm {T} ngoài [0,1]:")
                for _, r in v.sort_values(f"nocc_max_{T}", ascending=False).iterrows():
                    print(f"           {r['molecule']:10} max = {r[f'nocc_max_{T}']:.6f}"
                          f"   min = {r[f'nocc_min_{T}']:+.3e}   ({int(r[cb])} orbital)")
            else:
                print(f"        mật độ {T} N-representable trên cả {len(df)} chất")

        cl = f"lambda_conv_{T}"
        if cl in df and df[cl].notna().any():
            nb = (~df[cl].fillna(True).astype(bool)).sum()
            if nb:
                print(f"        !! {T}: {nb} chất có Lambda không hội tụ")

        c = f"d_dm_ff_{T}"
        if c in df and df[c].notna().any():
            t = df[df[c].notna()]
            print(f"        {T:7} mật độ so với finite field: "
                  f"RMS {np.sqrt((t[c]**2).mean()):.2f} %   "
                  f"max {t[c].abs().max():.2f} % ({t.loc[t[c].abs().idxmax(),'molecule']})")

    if ref_json and os.path.exists(ref_json):
        ref = json.load(open(ref_json))
        df["ref"] = df.molecule.map(ref)
        sub = df[df.ref.notna()]
        print(f"\n        RMSE regularized so với {len(sub)} giá trị CCSD(T)/CBS:")
        for pre, lab in (("mu_dm", "mật độ"), ("mu_ff", "finite field (-dE/dF)")):
            for T in ("CCSD", "CCSD_T"):
                col = f"{pre}_{T}"
                if col not in sub:
                    continue
                t = sub[sub[col].notna()]
                if not len(t):
                    continue
                err = 100 * (t[col] - t.ref) / np.maximum(t.ref, 1.0)
                print(f"          {T:7} {lab:24} n={len(t):3}  "
                      f"RMSE = {np.sqrt((err**2).mean()):6.2f} %"
                      f"   ME = {err.mean():+6.2f} %"
                      f"   MAX = {err.abs().max():6.2f} % "
                      f"({t.loc[err.abs().idxmax(),'molecule']})")
        print("          (bài báo, CCSD trên tập SP: RMSE 4.80 %)")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="file JSON hình học (không cần khi --merge)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--levels", nargs="+", default=["ccsd", "ccsd_t"],
                    choices=list(LEVELS),
                    help="muc ly thuyet nao duoc tinh. ccsd_t them cot CCSD(T) "
                         "de tu dung lam tham chieu thay cho gia tri cua nhom. "
                         "Chi phi: o --dipole ff chi them 6 lan tinh (T) (dung "
                         "chung diem truong voi CCSD); o --dipole dm thi them "
                         "phuong trinh Lambda co so hang (T), DAT gap hang chuc "
                         "lan mot lan (T) don le.")
    ap.add_argument("--dipole", choices=["ff", "dm"], default="ff",
                    help="ff = chạy finite field (mặc định, cột chính); "
                         "dm = BỎ QUA finite field, rẻ gấp 7 lần. Cột mu_dm_CCSD "
                         "(mycc.make_rdm1) LUÔN được tính trong cả hai chế độ vì "
                         "nó miễn phí — Lambda đã được giải sẵn. Ở chế độ dm thì "
                         "mu_ff_CCSD để trống, KHÔNG bị điền bằng mu_dm.")
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
        row = run_one(name, props, args.max_memory, args.dipole, args.levels)
        p = write_row(args.outdir, i, row)
        print(f"[{i}] {name}: {row['status']} | "
              f"{'  '.join(x for x in (_fmt_dip(row, LEVELS[k]) for k in args.levels) if x)} | "
              f"|g|={row['grad_norm']} | spread={row['E_spread_mH']} mH | "
              f"{row['walltime_s']}s -> {p}", flush=True)

    if len(todo) > 1 and not args.only and args.index is None:
        do_merge(args.outdir, out_xlsx, args.ref_json)


if __name__ == "__main__":
    main()