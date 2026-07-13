"""
Benchmark SIE4x4 cho OBDH (orbital-optimized double hybrid = B2PLYP self-consistent, pyCMF).
DUAL-INIT (UHF & UKS), 2 nhánh loc/deloc, có KIỂM TRA HỘI TỤ ĐẦY ĐỦ.

Yêu cầu ghi ra Excel — 4 sheet TÁCH BIỆT:
  1. NangLuong : năng lượng tổng OBDH (Hartree), 2 init.
  2. SaiSo     : De và sai số so benchmark, 2 init.
  3. Nghiem    : nhãn loc/deloc của nghiệm được chọn, 2 init.
  4. HoiTu     : trạng thái hội tụ — SCF nền (UHF/UKS) VÀ bước OBDH — cho
                 CẢ hai chiến lược guess (sym/broken), từng init. Cột
                 'chiến_lược_guess' = điểm xuất phát; cột 'nhãn' = nghiệm THỰC
                 TẾ (loc/deloc đọc từ spin). Hai cột này KHÁC nhau: vd NH3 cả
                 hai chiến lược đều cho nhãn 'loc'. Đây là bảng kiểm
                 tra độ tin cậy; mọi ô phải True thì kết quả mới đáng tin.

Bản pyCMF đã sửa để obmp2_iter lưu mp.converged (mp.converged = conv), nên
OBDH_CL(mf).converged đọc được sau .run(). SCF nền đọc mf.converged như thường.

KHÁC BIỆT CÓ CHỦ ĐÍCH so với calc_DFT (xem docstring localized_guess để biết
chi tiết + bằng chứng số): calc_DFT KHÔNG tách phân tử để ép localized vì PBE
(0% HF) cho multi-guess toàn cục tự phủ cả 2 basin. OBDH dùng functional lai
53% HF -> multi-guess toàn cục CHỈ chạm 1 basin (đúng cho cả UHF lẫn UKS init),
nên PHẢI cưỡng bức localized bằng broken-symmetry guess (localized_guess).

Chạy nặng: NH3/H2O aug-cc-pvtz rất tốn. Test nhanh: BASIS='aug-cc-pvdz' (vẫn có
delocalized nhờ hàm khuếch tán), GRID=3.
"""

import json
import numpy as np
import pandas as pd
from pyscf import gto, scf
from pycmf.OBDH import OBDH_CL

# ---------------------------------------------------------------------------
BASIS = 'aug-cc-pvtz'
GRID = 5
ALPHAA = (0.53, 0.39)
XC_BASE = f'{ALPHAA[0]}*HF + {1-ALPHAA[0]}*B88, {1-ALPHAA[1]}*LYP'
OBDH_THRESH = 1e-6
OBDH_NITER = 500              # tăng trần để tránh chạm trần chưa hội tụ
SCF_CONV_TOL = 1e-9
SCF_MAX_CYCLE = 300
S2_TOL = 0.15   # dung sai <S2> quanh 0.75. Nới từ 0.05 -> 0.15 vì UHF cho hệ
                # nhiều electron (NH3+/H2O+) có spin contamination tự nhiên
                # S2~0.80 (nghiệm ĐÚNG), không phải rác. Nghiệm rác thường S2>1.0.
GUESSES = ['minao', 'atom', 'huckel']   # multi-guess cho nhánh deloc (đồng nhất calc_DFT)
INITS = ['UHF', 'UKS']
JSON_PATH = "/home/giahuy/Code/job/OBDH/bench_SIE_DFT/geometry/input.json"
OUTPUT_FILE = 'OBDH_results.xlsx'

SYSTEMS = ['H2_plus_He', 'He2_plus', 'NH3_2_plus', 'H2O_2_plus']
R_POINTS = ['R_1.0', 'R_1.25', 'R_1.5', 'R_1.75']
DL_POINT = 'dissociation_limit'
EH2KCAL = 627.509

BENCHMARK = {
    'H2_plus_He': {'1.0': 64.4, '1.25': 58.9, '1.5': 48.7, '1.75': 38.3},
    'He2_plus':   {'1.0': 56.9, '1.25': 46.9, '1.5': 31.3, '1.75': 19.1},
    'NH3_2_plus': {'1.0': 35.9, '1.25': 25.9, '1.5': 13.4, '1.75': 4.9},
    'H2O_2_plus': {'1.0': 39.7, '1.25': 29.1, '1.5': 16.9, '1.75': 9.3},
}

FRAG_SPEC = {
    'He2_plus':   [('He', 1, 1), ('He', 0, 0)],
    'H2_plus_He': [('H', 0, 1), ('H', 1, 0), ('He', 0, 0)],
    'NH3_2_plus': None,
    'H2O_2_plus': None,
}


# ---------------------------------------------------------------------------
def build_mol(coords):
    a = "; ".join(f"{x['element']} {x['coordinates'][0]} {x['coordinates'][1]} {x['coordinates'][2]}"
                  for x in coords)
    return gto.Mole(atom=a, charge=1, spin=1, basis=BASIS, verbose=0).build()


def make_mf(mol, init):
    if init == 'UHF':
        mf = scf.UHF(mol)
    else:
        mf = scf.UKS(mol)
        mf.xc = XC_BASE
        mf.grids.level = GRID
    mf.max_cycle = SCF_MAX_CYCLE
    mf.conv_tol = SCF_CONV_TOL
    return mf


def atom_spins(mf):
    mol = mf.mol
    dm = mf.make_rdm1()
    sd = dm[0] - dm[1]
    s = mol.intor('int1e_ovlp')
    ps = sd @ s
    sl = mol.aoslice_by_atom()
    return [float(np.trace(ps[sl[i, 2]:sl[i, 3], sl[i, 2]:sl[i, 3]])) for i in range(mol.natm)]


def label_solution(mf):
    sp = atom_spins(mf)
    return 'loc' if max(abs(x) for x in sp) > 0.75 else 'deloc'


def run_obdh(mf):
    """Chạy OBDH; trả (ene_tot, converged). converged đọc từ ob.converged
    (pyCMF đã sửa để lưu). mf.converged phải True trước khi gọi."""
    ob = OBDH_CL(mf)
    ob.alphaa = ALPHAA
    ob.thresh = OBDH_THRESH
    ob.niter = OBDH_NITER
    ob.second_order = True
    ob.use_embed = False
    ob.use_cl = False
    ob.verbose = 0
    ob.run()
    conv = getattr(ob, 'converged', None)
    if conv is None:
        print("      *** CẢNH BÁO: ob.converged=None -> bản pyCMF CHƯA lưu "
              "converged. Cập nhật pyCMF (obmp2_iter: mp.converged = conv). "
              "Không thể kiểm hội tụ OBDH! ***")
    return ob.ene_tot, bool(conv) if conv is not None else None


def _frag_dm(atom_str, charge, spin, n, offset, init):
    dm = np.zeros((2, n, n))
    m = gto.Mole(atom=atom_str, charge=charge, spin=spin, basis=BASIS, verbose=0).build()
    if m.nelectron > 0:
        mf = make_mf(m, init); mf.kernel()
        d = mf.make_rdm1()
        if np.array(d).ndim == 2:
            d = np.array([d / 2, d / 2])
        dm[0][offset:offset+m.nao, offset:offset+m.nao] = d[0]
        dm[1][offset:offset+m.nao, offset:offset+m.nao] = d[1]
    return dm, m.nao


def _frag_dm_atomwise(frag_atoms, charge, spin, init):
    """Tính dm cho 1 mảnh (list các dict nguyên tử), trả dm theo thứ tự nguyên
    tử của mảnh đó."""
    astr = "; ".join(f"{a['element']} {a['coordinates'][0]} {a['coordinates'][1]} {a['coordinates'][2]}"
                     for a in frag_atoms)
    m = gto.Mole(atom=astr, charge=charge, spin=spin, basis=BASIS, verbose=0).build()
    if m.nelectron == 0:
        return m, np.zeros((2, m.nao, m.nao))
    mf = make_mf(m, init); mf.kernel()
    d = mf.make_rdm1()
    if np.array(d).ndim == 2:
        d = np.array([d / 2, d / 2])
    return m, np.array(d)


def localized_guess(sys, coords, mol, init):
    """Dựng dm bất đối xứng (hole định xứ trên 1 mảnh) cho nhánh localized.

    ===================================================================
    TẠI SAO OBDH CẦN localized_guess MÀ calc_DFT KHÔNG CẦN?
    ===================================================================
    Đây là điểm KHÁC BIỆT có chủ đích giữa hai file (không phải bất nhất tùy
    tiện). Lý do là FUNCTIONAL, không phải init UHF/UKS. Đã kiểm chứng bằng số
    (He2+ dl, aug-cc-pvdz, UKS+newton, thử mọi guess minao/atom/huckel/1e):

      - PBE (0% HF, dùng trong calc_DFT): multi-guess toàn cục TỰ PHỦ CẢ HAI
        basin — minao/atom -> delocalized (-5.015), huckel/1e -> localized
        (-4.876). Nên calc_DFT chỉ cần multi-guess toàn cục rồi ĐO spin để dán
        nhãn loc/deloc; KHÔNG cần tách phân tử.

      - Functional OBDH (0.53*HF + 0.47*B88, LYP): CÙNG UKS+newton+mọi guess
        vẫn CHỈ ra delocalized (-4.954). Bề mặt SCF của hybrid nặng HF không cho
        multi-guess toàn cục chạm tới localized basin.

    Kiểm thêm: cả UHF-init lẫn UKS-init của OBDH đều KHÔNG tự phủ 2 basin
    (mỗi init chỉ ra 1 basin: He2+ -> deloc cả hai; H2+He -> UHF loc, UKS deloc).
    scf.newton cũng KHÔNG giúp phủ thêm. => nguyên nhân là functional lai HF.

    HỆ QUẢ: với OBDH, localized basin chỉ chạm tới được bằng cách CƯỠNG BỨC —
    dựng guess bất đối xứng từ (cation + neutral) rồi ghép. Và basin này đôi khi
    THẤP HƠN deloc (He2+ dl UHF: loc -4.850 < deloc -4.826), nên bỏ sót nó =
    lấy nhầm nghiệm cao hơn, sai về biến phân. Vì vậy localized_guess LÀ CẦN
    THIẾT cho OBDH (cả hai init), khác với calc_DFT.

    ---------------------------------------------------------------
    LỖI CŨ (đã sửa) trong chính hàm này:
      (a) chia mảnh theo NỬA DANH SÁCH index -> sai cho NH3/H2O ở dl vì thứ tự
          là N,N,H,... (nguyên tử nặng đứng trước), mảnh bị trộn -> crash.
      (b) chia theo cutoff cố định 4.0 A -> ở R ngắn hai mảnh gần nhau, gom hết
          vào 1 mảnh, mảnh kia rỗng -> crash.
      SỬA: chia theo NGUYÊN TỬ NẶNG gần nhất (mỗi N/O là 1 tâm, mỗi H gán về tâm
      gần nhất), ghép dm theo AO slice của TỪNG nguyên tử (vì nguyên tử 2 mảnh
      xen kẽ trong thứ tự AO tổng).
    """
    n = mol.nao
    dm = np.zeros((2, n, n))
    sl = mol.aoslice_by_atom()

    if FRAG_SPEC[sys] is not None:
        # hệ nhỏ (He2+/H2+He): thứ tự đã là mảnh-liền-mảnh, ghép tuần tự OK
        offset = 0
        for (el, ch, sp), at in zip(FRAG_SPEC[sys], coords):
            c = at['coordinates']
            d, na = _frag_dm(f"{el} {c[0]} {c[1]} {c[2]}", ch, sp, n, offset, init)
            dm += d; offset += na
        return dm

    # NH3/H2O: chia theo NGUYÊN TỬ NẶNG gần nhất (không dùng cutoff cố định,
    # vì ở R ngắn hai mảnh gần nhau, cutoff cố định gom hết vào 1 mảnh -> crash).
    # Hai nguyên tử nặng (N hoặc O) là 2 tâm; mỗi H gán về nguyên tử nặng gần nhất.
    heavy = [i for i, a in enumerate(coords) if a['element'] in ('N', 'O')]
    if len(heavy) != 2:
        raise ValueError(f'{sys}: cần đúng 2 nguyên tử nặng, thấy {len(heavy)}')
    h1, h2 = heavy
    p1 = np.array(coords[h1]['coordinates'])
    p2 = np.array(coords[h2]['coordinates'])
    idx1, idx2 = [], []
    for i, a in enumerate(coords):
        pi = np.array(a['coordinates'])
        (idx1 if np.linalg.norm(pi - p1) <= np.linalg.norm(pi - p2) else idx2).append(i)

    for idx_list, ch, sp in [(idx1, 1, 1), (idx2, 0, 0)]:
        frag_atoms = [coords[i] for i in idx_list]
        m, d = _frag_dm_atomwise(frag_atoms, ch, sp, init)
        # ghép d (theo thứ tự nguyên tử trong mảnh) vào dm tổng theo AO slice
        # của từng nguyên tử tương ứng trong mol tổng
        frag_sl = m.aoslice_by_atom()
        for k, i_glob in enumerate(idx_list):
            a0, a1 = sl[i_glob, 2], sl[i_glob, 3]       # AO range nguyên tử trong mol tổng
            f0, f1 = frag_sl[k, 2], frag_sl[k, 3]       # AO range trong mảnh
            dm[0][a0:a1, a0:a1] = d[0][f0:f1, f0:f1]
            dm[1][a0:a1, a0:a1] = d[1][f0:f1, f0:f1]
    return dm


def _scf_valid(mf):
    """Nghiệm SCF nền hợp lệ = hội tụ + spin sạch (cùng tiêu chí calc_DFT)."""
    if not mf.converged:
        return False
    return abs(mf.spin_square()[0] - 0.75) <= S2_TOL


def solve_branch(sys, coords, init, branch):
    """Giải 1 nhánh cho 1 init: thử guess, LỌC nghiệm hợp lệ, lấy SCF hợp lệ E
    thấp nhất, rồi chạy OBDH.

    QUAN TRỌNG - phân biệt CHIẾN LƯỢC GUESS với NHÃN NGHIỆM:
      branch='sym'    : xuất phát từ guess đối xứng (minao/atom/huckel).
      branch='broken' : xuất phát từ guess phá đối xứng (localized_guess).
    Đây chỉ là ĐIỂM XUẤT PHÁT, KHÔNG phải nhãn nghiệm. Nghiệm cuối ra loc hay
    deloc được đọc từ SPIN thực tế qua label_solution() (field 'label'), độc lập
    với tên branch. Ví dụ: ở NH3 dl, CẢ branch 'sym' lẫn 'broken' đều hội tụ về
    nghiệm LOCALIZED (label='loc') - vì OBDH nặng HF/MP2 nên nghiệm vật lý đúng
    của NH3+ là localized (hole trên n(N)), không có nghiệm deloc bền. Đó KHÔNG
    phải lỗi; là kết quả vật lý. Hai branch chỉ để THĂM DÒ, việc dán nhãn dựa
    trên nghiệm thu được.

    Regression (He2+ aug-cc-pvdz): multi-guess ra kết quả GIỐNG HỆT guess mặc
    định khi nghiệm đã đúng (lệch 0.0). Multi-guess/broken chỉ CỨU khi guess
    mặc định bỏ sót basin (He2+: broken tìm loc -4.850 < sym deloc -4.826).

    Trả dict: E, scf_conv, scf_s2_ok, obdh_conv, label(nghiệm thực tế), note.
    """
    mol = build_mol(coords)

    # ---- bước 1: thu thập các SCF nền hợp lệ ----
    scf_candidates = []   # (e_scf, mf)
    if branch == 'sym':
        for g in GUESSES:
            mf = make_mf(mol, init)
            try:
                mf.kernel(dm0=mf.get_init_guess(key=g))
            except Exception:
                continue
            if _scf_valid(mf):
                scf_candidates.append((mf.e_tot, mf))
                es = sorted(c[0] for c in scf_candidates)
                if len(scf_candidates) >= 2 and es[1] - es[0] < 1e-7:
                    break
    else:
        try:
            dm0 = localized_guess(sys, coords, mol, init)
            mf = make_mf(mol, init)
            mf.kernel(dm0=dm0)
            if _scf_valid(mf):
                scf_candidates.append((mf.e_tot, mf))
        except Exception:
            return dict(E=None, scf_conv=False, scf_s2_ok=False,
                        obdh_conv=None, label=None, note='broken_guess_fail')

    # ---- bước 2: nếu không SCF nào hợp lệ, báo rõ ----
    if not scf_candidates:
        mf = make_mf(mol, init)
        if branch == 'sym':
            mf.kernel()
        else:
            try:
                mf.kernel(dm0=localized_guess(sys, coords, mol, init))
            except Exception:
                return dict(E=None, scf_conv=False, scf_s2_ok=False,
                            obdh_conv=None, label=None, note='broken_guess_fail')
        s2 = mf.spin_square()[0]
        return dict(E=None, scf_conv=bool(mf.converged),
                    scf_s2_ok=abs(s2 - 0.75) <= S2_TOL,
                    obdh_conv=None, label=label_solution(mf), note='scf_no_valid')

    # ---- bước 3: lấy SCF nền hợp lệ E thấp nhất, chạy OBDH ----
    scf_candidates.sort(key=lambda x: x[0])
    _, mf = scf_candidates[0]
    scf_conv = bool(mf.converged)
    s2 = mf.spin_square()[0]
    s2_ok = abs(s2 - 0.75) <= S2_TOL

    try:
        E, obdh_conv = run_obdh(mf)
    except Exception as e:
        return dict(E=None, scf_conv=scf_conv, scf_s2_ok=s2_ok,
                    obdh_conv=False, label=label_solution(mf),
                    note=f'obdh_exc:{str(e)[:30]}')

    return dict(E=E, scf_conv=scf_conv, scf_s2_ok=s2_ok,
                obdh_conv=obdh_conv, label=label_solution(mf), note='')


def solve(sys, coords, init):
    """2 chiến lược guess (sym + broken); trả nghiệm hợp lệ thấp nhất + chi tiết.
    Lưu ý: hai chiến lược có thể ra CÙNG nhãn nghiệm (vd NH3: cả hai -> loc).
    Đó là bình thường - ta lấy nghiệm variational thấp nhất bất kể nhãn."""
    d = solve_branch(sys, coords, init, 'sym')
    l = solve_branch(sys, coords, init, 'broken')

    def valid(b):
        return (b['E'] is not None and b['scf_conv'] and b['scf_s2_ok']
                and b['obdh_conv'] is True)

    cands = [b for b in (d, l) if valid(b)]
    if cands:
        chosen = min(cands, key=lambda b: b['E'])
    else:
        withE = [b for b in (d, l) if b['E'] is not None]
        chosen = None
        if withE:
            withE.sort(key=lambda b: (not (b['obdh_conv'] is True), b['E']))
            chosen = withE[0]

    return dict(chosen=chosen, sym=d, broken=l)


# ---------------------------------------------------------------------------
def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    E = {ini: {s: {} for s in SYSTEMS} for ini in INITS}       # năng lượng chọn
    LBL = {ini: {s: {} for s in SYSTEMS} for ini in INITS}     # nhãn chọn
    CONV = {ini: {s: {} for s in SYSTEMS} for ini in INITS}    # chi tiết hội tụ 2 nhánh

    print(f"OBDH dual-init | basis={BASIS} grid={GRID} alphaa={ALPHAA} niter={OBDH_NITER}\n")

    for sys in SYSTEMS:
        print(f"===== {sys} =====")
        for pt in R_POINTS + [DL_POINT]:
            line = f"  {pt:20s}"
            for ini in INITS:
                r = solve(sys, data[sys][pt], ini)
                ch = r['chosen']
                E[ini][sys][pt] = ch['E'] if ch else None
                LBL[ini][sys][pt] = ch['label'] if ch else 'FAIL'
                CONV[ini][sys][pt] = r
                if ch and ch['E'] is not None:
                    warn = '' if (ch['scf_conv'] and ch['obdh_conv'] is True and ch['scf_s2_ok']) else ' !'
                    line += f" | {ini}:{ch['E']:.5f}({ch['label']}){warn}"
                else:
                    line += f" | {ini}:FAIL"
            print(line)
        print()

    write_excel(E, LBL, CONV)


def write_excel(E, LBL, CONV):
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # ---- Sheet 1: NangLuong ----
        t_en = []
        for sys in SYSTEMS:
            row = {'Hệ': sys}
            for pt in R_POINTS + [DL_POINT]:
                for ini in INITS:
                    row[f'{pt}_{ini}'] = E[ini][sys][pt]
            t_en.append(row)

        # ---- Sheet 2: SaiSo ----
        t_err = []
        for sys in SYSTEMS:
            for rp in R_POINTS:
                rv = rp.replace('R_', '')
                bench = BENCHMARK[sys][rv]
                row = {'Hệ': sys, 'R/Re': float(rv), 'benchmark': bench}
                for ini in INITS:
                    e_dl = E[ini][sys][DL_POINT]
                    e_r = E[ini][sys][rp]
                    if e_dl is not None and e_r is not None:
                        dE = (e_dl - e_r) * EH2KCAL
                        row[f'De_{ini}'] = round(dE, 2)
                        row[f'err_{ini}'] = round(dE - bench, 2)
                    else:
                        row[f'De_{ini}'] = None
                        row[f'err_{ini}'] = None
                t_err.append(row)

        # ---- Sheet 3: Nghiem (loc/deloc của nghiệm được chọn) ----
        t_lbl = []
        for sys in SYSTEMS:
            row = {'Hệ': sys}
            for pt in R_POINTS + [DL_POINT]:
                for ini in INITS:
                    row[f'{pt}_{ini}'] = LBL[ini][sys][pt]
            t_lbl.append(row)

        # ---- Sheet 4: HoiTu (bảng kiểm tra hội tụ TÁCH RIÊNG) ----
        # mỗi dòng = 1 (hệ, điểm, init, nhánh); cột trạng thái SCF + OBDH
        t_conv = []
        for sys in SYSTEMS:
            for pt in R_POINTS + [DL_POINT]:
                for ini in INITS:
                    r = CONV[ini][sys][pt]
                    for branch in ['sym', 'broken']:
                        b = r[branch]
                        chosen = (r['chosen'] is b)
                        t_conv.append({
                            'Hệ': sys, 'điểm': pt, 'init': ini, 'chiến_lược_guess': branch,
                            'SCF_hội_tụ': b['scf_conv'],
                            'SCF_S2_ok': b['scf_s2_ok'],
                            'OBDH_hội_tụ': b['obdh_conv'],
                            'nhãn': b['label'],
                            'E': None if b['E'] is None else round(b['E'], 6),
                            'ĐƯỢC_CHỌN': chosen,
                            'ghi_chú': b['note'],
                        })

        pd.DataFrame(t_en).to_excel(writer, sheet_name='NangLuong', index=False)
        pd.DataFrame(t_err).to_excel(writer, sheet_name='SaiSo', index=False)
        pd.DataFrame(t_lbl).to_excel(writer, sheet_name='Nghiem', index=False)
        pd.DataFrame(t_conv).to_excel(writer, sheet_name='HoiTu', index=False)

        for sh in ['NangLuong', 'SaiSo', 'Nghiem', 'HoiTu']:
            ws = writer.sheets[sh]
            for col in ws.columns:
                w = max(len(str(c.value or '')) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(w + 2, 38)

    # cảnh báo tóm tắt ra stdout
    n_bad = sum(1 for sys in SYSTEMS for pt in R_POINTS+[DL_POINT] for ini in INITS
                if (lambda ch: ch is None or not ch['scf_conv']
                    or ch['obdh_conv'] is not True or not ch['scf_s2_ok'])
                (CONV[ini][sys][pt]['chosen']))
    print(f"[Xong] {OUTPUT_FILE}  (4 sheet: NangLuong, SaiSo, Nghiem, HoiTu)")
    if n_bad:
        print(f"  *** CẢNH BÁO: {n_bad} nghiệm được chọn CHƯA hội tụ đầy đủ "
              f"- xem sheet HoiTu (cột ĐƯỢC_CHỌN=True mà có False).")
    else:
        print("  Tất cả nghiệm được chọn đều hội tụ đầy đủ (SCF + OBDH + S2).")


if __name__ == '__main__':
    main()