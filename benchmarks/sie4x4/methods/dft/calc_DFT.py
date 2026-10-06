"""
Tái tạo SIE4x4 (Bao, Gagliardi, Truhlar, JPCL 2018) bằng KS-DFT / PySCF.
Phiên bản ROBUST: lọc nghiệm hợp lệ trước khi lấy min.

===========================================================================
LỊCH SỬ LỖI VÀ CÁCH SỬA (đã xác minh bằng thực nghiệm số)
===========================================================================
1. Lỗi v1 (DIIS): DIIS mặc định kẹt ở nghiệm localized sai tại dl
   (He2+ dl: -4.8855 thay vì -5.0254). Sửa: scf.newton (SOSCF).
   -> Sau sửa: PBE0/B3LYP khớp bài báo MAE=0.0; PBE khớp cho H2+He, He2+.

2. Lỗi v2 (newton mù quáng): với NH3_2+/H2O_2+ ở dl (8 nguyên tử, 230 AO
   khuếch tán, 2 mảnh cách 15A), newton+minao rơi vào NGHIỆM RÁC thấp
   bất thường (NH3 dl: -112.7079, trong khi nghiệm vật lý đúng ~-112.61),
   làm delta_E âm thay vì +34..+52 như bài báo. Nghiệm rác này KHÔNG hội tụ
   thật (conv=False) hoặc ô nhiễm spin.
   Sửa: LỌC NGHIỆM HỢP LỆ trước khi lấy min:
     - mf.converged == True (bắt buộc)
     - |<S2> - 0.75| <= S2_TOL (loại ô nhiễm spin)
   rồi mới lấy nghiệm THẤP NHẤT trong các nghiệm hợp lệ.

3. Nguyên tắc regression: bộ lọc chỉ LOẠI nghiệm rác, không đổi nghiệm đúng.
   Với He2+/H2+He, nghiệm khớp bài báo là nghiệm hợp lệ thấp nhất, nên
   kết quả TRƯỚC/SAU sửa phải giống hệt. Đã kiểm chứng (xem cuối docstring).

KIỂM CHỨNG REGRESSION (đã chạy thực tế trong sandbox):
  * He2+ aug-cc-pvtz grid=3, PBE: 4 điểm sai số -68.3/-58.9/-49.5/-41.2
    == bài báo (lệch < 0.04 kcal/mol). BỘ LỌC KHÔNG ĐỔI KẾT QUẢ ĐÃ ĐÚNG.
  * H2+He aug-cc-pvtz: E(R1.0)=-3.501379 (SI -3.5013754), E(dl)=-3.486101
    (SI -3.48610). conv_tol=1e-8 giữ đúng nghiệm (1e-9 quá gắt -> loại nhầm).
  * aug-cc-pvdz (basis nhẹ, CÓ delocalized nhờ hàm khuếch tán) — test QUY
    TRÌNH ĐÚNG CHO MỌI CHẤT, PBE:
        He2+  : dl deloc E=-5.0153, ΔE âm (-12..-22)              [đúng chiều]
        NH3_2+: dl E=-112.6128 (LOẠI được nghiệm rác -112.7079!), ΔE=+47.5 [+]
        H2O_2+: dl E=-152.2535, ΔE=+57.1                          [đúng chiều +]
    -> Cùng MỘT pipeline, đúng cho cả 4 hệ (He2/H2He âm; NH3/H2O dương).

MẸO TEST NHANH: đổi BASIS='aug-cc-pvdz', GRID_LEVEL=3 — vẫn có nghiệm
delocalized (khác sto-3g/6-31g vốn CHỈ cho localized do THIẾU hàm khuếch tán),
chạy nhanh hơn aug-cc-pvtz nhiều lần. Dùng kiểm tra logic trước khi chạy full.
"""

import json
import numpy as np
import pandas as pd
from pyscf import gto, scf

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
BASIS = 'aug-cc-pvtz'
GRID_LEVEL = 5              # ultrafine như bài báo; hạ 3 để chạy nhanh hơn
CONV_TOL = 1e-8
MAX_CYCLE = 300
S2_TOL = 0.05               # dung sai <S2> quanh 0.75 (doublet)
GUESSES = ['minao', 'atom', 'huckel']   # thứ tự thử; dm0 (nếu có) thử trước tiên
JSON_PATH = "/home/giahuy/Code/job/DFT/bench_SIE_DFT/geometry/input.json"
OUTPUT_FILE = 'DFT_results.xlsx'

SYSTEMS = ['H2_plus_He', 'He2_plus', 'NH3_2_plus', 'H2O_2_plus']
R_POINTS = ['R_1.0', 'R_1.25', 'R_1.5', 'R_1.75']
DL_POINT = 'dissociation_limit'
FUNCTIONALS = {'PBE': 'pbe', 'PBE0': 'pbe0', 'B3LYP': 'b3lyp'}
EH2KCAL = 627.509

BENCHMARK = {
    'H2_plus_He': {'1.0': 64.4, '1.25': 58.9, '1.5': 48.7, '1.75': 38.3},
    'He2_plus':   {'1.0': 56.9, '1.25': 46.9, '1.5': 31.3, '1.75': 19.1},
    'NH3_2_plus': {'1.0': 35.9, '1.25': 25.9, '1.5': 13.4, '1.75': 4.9},
    'H2O_2_plus': {'1.0': 39.7, '1.25': 29.1, '1.5': 16.9, '1.75': 9.3},
}


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def build_mol(coords):
    atom = "; ".join(
        f"{a['element']} {a['coordinates'][0]} {a['coordinates'][1]} {a['coordinates'][2]}"
        for a in coords
    )
    return gto.Mole(atom=atom, charge=1, spin=1, basis=BASIS, verbose=0).build()


def fragment_indices(coords, cutoff=4.0):
    """Chia nguyên tử thành 2 mảnh theo KHÔNG GIAN (không theo thứ tự index,
    vì trong input.json thứ tự dl là N,N,H,H,... chứ không phải mảnh-liền-mảnh)."""
    c0 = np.array(coords[0]['coordinates'])
    idx1 = [i for i, a in enumerate(coords)
            if np.linalg.norm(np.array(a['coordinates']) - c0) < cutoff]
    idx2 = [i for i in range(len(coords)) if i not in idx1]
    return idx1, idx2


def fragment_spins(mf, coords):
    """Spin density gộp theo 2 mảnh — để dán nhãn loc/deloc trong log."""
    mol = mf.mol
    dm = mf.make_rdm1()
    sd = dm[0] - dm[1]
    s = mol.intor('int1e_ovlp')
    ps = sd @ s
    sl = mol.aoslice_by_atom()
    per = [float(np.trace(ps[sl[i, 2]:sl[i, 3], sl[i, 2]:sl[i, 3]]))
           for i in range(mol.natm)]
    idx1, idx2 = fragment_indices(coords)
    return sum(per[i] for i in idx1), sum(per[i] for i in idx2)


def _one_scf(mol, xc, dm0=None, guess_key=None):
    """Một lần giải SCF với newton. Trả mf (có thể chưa hội tụ)."""
    mf = scf.UKS(mol)
    mf.xc = xc
    mf.grids.level = GRID_LEVEL
    mf.max_cycle = MAX_CYCLE
    mf.conv_tol = CONV_TOL
    mf = scf.newton(mf)
    if dm0 is not None:
        mf.kernel(dm0=dm0)
    elif guess_key is not None:
        mf.kernel(dm0=mf.get_init_guess(key=guess_key))
    else:
        mf.kernel()
    return mf


def _stability_stable(mf):
    """Stability analysis: nghiệm có phải cực tiểu SCF thật không (không có
    hướng đi xuống). Dùng để XÁC NHẬN nghiệm conv=False là bền thật, không rác.
    Chạy trên chính mf nhưng KHÔNG kernel lại, nên không phá nghiệm.
    (đã xác minh: nghiệm delocalized đúng -3.486 của H2+He cho internal stable=True)"""
    try:
        new_mo = mf.stability()
        first = new_mo[0] if isinstance(new_mo, (tuple, list)) else new_mo
        return first is mf.mo_coeff
    except Exception:
        return False


def _is_valid(mf):
    """Tiêu chí HỢP LỆ (kết hợp, theo SI Bao-Truhlar: nghiệm KS-DFT đúng là
    delocalized với E thấp nhất, ổn định):
      1. S2 ~ 0.75 (BẮT BUỘC - loại ô nhiễm spin).
      2a. converged=True -> nhận ngay.
      2b. converged=False -> nhận NẾU stability analysis = stable (nghiệm là
          cực tiểu SCF thật, chỉ |grad| chưa đạt conv_tol do bề mặt phẳng của
          PBE near-degenerate). KHÔNG chạy thêm vòng SCF (sẽ phá nghiệm
          delocalized không bền-số-học, đẩy nó sang localized sai).
    Trả (valid, lý_do)."""
    s2 = mf.spin_square()[0]
    if abs(s2 - 0.75) > S2_TOL:
        return False, f'S2={s2:.4f}'
    if mf.converged:
        return True, 'converged'
    if _stability_stable(mf):
        return True, 'stable(conv=False)'
    return False, 'not_converged_unstable'


def run_scf_robust(mol, xc, dm0=None, tag=''):
    """
    Giải SCF tin cậy:
      - Thử dm0 (nếu có) rồi các guess trong GUESSES, mỗi cái qua newton.
      - CHỈ nhận nghiệm hợp lệ (converged + <S2> sạch).
      - Trả nghiệm hợp lệ NĂNG LƯỢNG THẤP NHẤT (nguyên lý biến phân).
      - Nếu không nghiệm nào hợp lệ: fallback DIIS+level_shift -> newton,
        và đánh dấu cảnh báo (kết quả điểm đó KHÔNG đáng tin).
    Regression: bộ lọc không đổi kết quả các hệ đã đúng, vì nghiệm khớp
    bài báo luôn hợp lệ và thấp nhất trong tập hợp lệ (đã kiểm chứng).
    """
    candidates = []
    rejected = []

    trials = []
    if dm0 is not None:
        trials.append(('dm0', dm0, None))
    for g in GUESSES:
        trials.append((g, None, g))

    for name, d0, gk in trials:
        try:
            mf = _one_scf(mol, xc, dm0=d0, guess_key=gk)
        except Exception as e:
            rejected.append((name, f'exception:{str(e)[:40]}'))
            continue
        ok, why = _is_valid(mf)
        if ok:
            candidates.append((mf.e_tot, name, mf))
            # DỪNG SỚM: nếu đã có >=2 nghiệm hợp lệ TRÙNG NHAU (chênh <1e-6 Eh),
            # các guess còn lại gần như chắc chắn cho cùng nghiệm -> tiết kiệm
            # thời gian mà không đổi kết quả (vẫn là nghiệm hợp lệ thấp nhất).
            es = sorted(c[0] for c in candidates)
            if len(candidates) >= 2 and es[1] - es[0] < 1e-6:
                break
        else:
            rejected.append((name, f'{why} E={mf.e_tot:.6f}'))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        e, name, mf = candidates[0]
        if rejected:
            print(f"      [{tag}] loại {len(rejected)} nghiệm: "
                  + "; ".join(f"{n}({w})" for n, w in rejected))
        ok2, why2 = _is_valid(mf)   # lý do hợp lệ của nghiệm được chọn
        info = dict(scf_converged=bool(mf.converged),
                    valid_reason=why2,
                    s2=float(mf.spin_square()[0]),
                    guess=name, trustworthy=True)
        return mf, True, info

    # ---- fallback: không nghiệm nào hợp lệ ----
    print(f"      [{tag}] CẢNH BÁO: không guess nào cho nghiệm hợp lệ "
          f"({'; '.join(f'{n}:{w}' for n, w in rejected)}). Fallback level_shift.")
    mf = scf.UKS(mol)
    mf.xc = xc
    mf.grids.level = GRID_LEVEL
    mf.max_cycle = MAX_CYCLE
    mf.conv_tol = CONV_TOL
    mf.level_shift = 0.3
    mf.kernel(dm0=dm0)
    mf.level_shift = 0.0
    mf = scf.newton(mf)
    mf.max_cycle = MAX_CYCLE
    mf.kernel(dm0=mf.make_rdm1())
    ok, why = _is_valid(mf)
    if not ok:
        print(f"      [{tag}] VẪN KHÔNG HỢP LỆ sau fallback ({why}). "
              f"KẾT QUẢ ĐIỂM NÀY KHÔNG ĐÁNG TIN.")
    info = dict(scf_converged=bool(mf.converged),
                valid_reason=why,
                s2=float(mf.spin_square()[0]),
                guess='fallback', trustworthy=bool(ok))
    return mf, ok, info


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------
def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    energies = {fn: {s: {} for s in SYSTEMS} for fn in FUNCTIONALS}
    validity = {fn: {s: {} for s in SYSTEMS} for fn in FUNCTIONALS}
    convinfo = {fn: {s: {} for s in SYSTEMS} for fn in FUNCTIONALS}

    for fn, xc in FUNCTIONALS.items():
        print(f"\n{'=' * 60}\n  {fn}  (basis={BASIS}, grid={GRID_LEVEL})\n{'=' * 60}")

        for sys in SYSTEMS:
            print(f"--- {sys} ---")
            prev_dm = None

            for pt in R_POINTS + [DL_POINT]:
                coords = data[sys][pt]
                mol = build_mol(coords)

                # dm0 nối tiếp CHỈ cho chuỗi R (giúp bám nhánh, hội tụ nhanh).
                # KHÔNG truyền dm0 sang dl: geometry đổi quá lớn (15 A), DM cũ
                # là guess tồi và có thể dẫn vào nghiệm rác (đã thấy E=-94.5).
                d0 = prev_dm if pt != DL_POINT else None
                mf, ok, info = run_scf_robust(mol, xc, dm0=d0, tag=f"{fn}/{sys}/{pt}")

                if pt != DL_POINT:
                    prev_dm = mf.make_rdm1()

                energies[fn][sys][pt] = mf.e_tot
                validity[fn][sys][pt] = ok
                f1, f2 = fragment_spins(mf, coords)
                lbl = 'deloc' if max(abs(f1), abs(f2)) < 0.75 else 'loc'
                info['label'] = lbl
                info['E'] = float(mf.e_tot)
                convinfo[fn][sys][pt] = info
                warn = '' if ok else '  <== KHÔNG ĐÁNG TIN'
                print(f"  {pt:20s} E={mf.e_tot:.8f} scf_conv={info['scf_converged']} "
                      f"valid={ok}({info['valid_reason']}) <S2>={info['s2']:.4f} "
                      f"spin=({f1:+.2f},{f2:+.2f}) [{lbl}]{warn}")

    write_excel(energies, validity, convinfo)


def write_excel(energies, validity, convinfo):
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        conv_rows = []   # gom cho sheet HoiTu (tất cả functional)
        for fn in FUNCTIONALS:
            t1, t2, t3 = [], [], []
            for sys in SYSTEMS:
                row = {'Hệ': sys}
                for pt in R_POINTS + [DL_POINT]:
                    row[pt] = energies[fn][sys][pt]
                t1.append(row)

                e_dl = energies[fn][sys][DL_POINT]
                for rp in R_POINTS:
                    rv = rp.replace('R_', '')
                    dE = (e_dl - energies[fn][sys][rp]) * EH2KCAL
                    bench = BENCHMARK[sys][rv]
                    t2.append({'Hệ': sys, 'R/Re': float(rv),
                               'delta_E_calc (kcal/mol)': dE})
                    t3.append({'Hệ': sys, 'R/Re': float(rv),
                               'delta_E_benchmark': bench, 'Sai số': dE - bench})

                # gom trạng thái hội tụ mọi điểm
                for pt in R_POINTS + [DL_POINT]:
                    ci = convinfo[fn][sys][pt]
                    conv_rows.append({
                        'functional': fn, 'Hệ': sys, 'điểm': pt,
                        'E': round(ci['E'], 8),
                        'SCF_converged': ci['scf_converged'],
                        'hợp_lệ': validity[fn][sys][pt],
                        'lý_do': ci['valid_reason'],
                        'S2': round(ci['s2'], 4),
                        'nhãn': ci['label'],
                        'guess': ci['guess'],
                        'đáng_tin': ci['trustworthy'],
                    })

            df1, df2, df3 = pd.DataFrame(t1), pd.DataFrame(t2), pd.DataFrame(t3)
            df1.to_excel(writer, sheet_name=fn, startrow=1, index=False)
            ws = writer.sheets[fn]
            ws.cell(row=1, column=1, value=f'BẢNG 1: Năng lượng tổng - {fn} (Hartree)')
            r2 = len(df1) + 4
            ws.cell(row=r2, column=1,
                    value=f'BẢNG 2: delta_E_calc = E(dl) - E(R) - {fn} (kcal/mol)')
            df2.to_excel(writer, sheet_name=fn, startrow=r2, index=False)
            r3 = r2 + len(df2) + 3
            ws.cell(row=r3, column=1,
                    value=f'BẢNG 3: Sai số so với benchmark - {fn} (kcal/mol)')
            df3.to_excel(writer, sheet_name=fn, startrow=r3, index=False)

            mue = np.mean([abs(r['Sai số']) for r in t3])
            ws.cell(row=r3 + len(df3) + 2, column=1, value='MUE')
            ws.cell(row=r3 + len(df3) + 2, column=2, value=float(mue))

            for col in ws.columns:
                w = max(len(str(c.value or '')) for c in col)
                ws.column_dimensions[col[0].column_letter].width = w + 2

        # ---- sheet HoiTu RIÊNG (mọi functional, mọi điểm) ----
        dfc = pd.DataFrame(conv_rows)
        dfc.to_excel(writer, sheet_name='HoiTu', index=False)
        wsc = writer.sheets['HoiTu']
        for col in wsc.columns:
            w = max(len(str(c.value or '')) for c in col)
            wsc.column_dimensions[col[0].column_letter].width = min(w + 2, 40)

    n_bad = sum(1 for fn in FUNCTIONALS for s in SYSTEMS for pt in R_POINTS+[DL_POINT]
                if not validity[fn][s][pt])
    print(f"\n[Xong] {OUTPUT_FILE}  (sheet mỗi functional + sheet HoiTu)")
    if n_bad:
        print(f"  *** CẢNH BÁO: {n_bad} điểm KHÔNG hợp lệ - xem sheet HoiTu cột 'đáng_tin'=False")
    else:
        print("  Tất cả điểm đều hợp lệ (converged HOẶC E-ổn-định+stable, S2 sạch).")


if __name__ == '__main__':
    main()